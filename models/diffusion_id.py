import torch
import torch.nn as nn
import torch.nn.functional as F

# --- tiny helpers ---
def sinusoidal_time_embed(timesteps, dim):
    # timesteps: [B], returns [B, dim]
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * -(torch.log(torch.tensor(10000.0)) / (half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class SimpleDenoiser(nn.Module):
    """
    Minimal per-token denoiser: x_t (H) + time emb (H) + cond (H) -> eps_hat (H)
    Runs independently at each time step (fast & simple).
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(),
            nn.Linear(hidden_dim*4, hidden_dim)
        )

    def forward(self, x_t, t, cond):
        # x_t: [B, L, H], t: [B], cond: [B, L, H]
        B, L, H = x_t.shape
        t_emb = sinusoidal_time_embed(t, H)            # [B, H]
        t_emb = self.time_mlp(t_emb).unsqueeze(1)      # [B, 1, H] -> broadcast
        c_emb = self.cond_mlp(cond)                    # [B, L, H]
        h = x_t + t_emb + c_emb
        return self.net(h)                             # eps_hat: [B, L, H]

class GaussianDiffusionHead(nn.Module):
    """
    DDPM on hidden states (size H) over a length-L sequence.
    Condition: plan vector + future temporal embeds.
    """
    def __init__(self, hidden_dim, token_len, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_len = token_len
        self.T = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0/alphas))

        self.eps_net = SimpleDenoiser(hidden_dim)

        # project concatenated condition (plan + temporal) to H
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim), nn.GELU()
        )

    @torch.no_grad()
    def sample(self, cond_hidden, steps=50):
        """
        cond_hidden: dict with
          - 'plan': [B, H]
          - 'future_temporal': [B, L, H]  (your dec_embeds)
        returns: x_0 sample [B, L, H]
        """
        plan = cond_hidden['plan']                           # [B, H]
        dec = cond_hidden['future_temporal']                 # [B, L, H]
        B, L, H = dec.shape

        plan_seq = plan.unsqueeze(1).expand(B, L, H)         # [B, L, H]
        cond = self.cond_proj(torch.cat([plan_seq, dec], -1))# [B, L, H]

        # start from pure noise
        x_t = torch.randn(B, L, H, device=dec.device)

        # Use strided timesteps for fast sampling
        ts = torch.linspace(self.T-1, 0, steps, device=dec.device).long()

        for i in range(steps):
            t = ts[i]
            t_batch = torch.full((B,), t, device=dec.device, dtype=torch.long)

            eps_hat = self.eps_net(x_t, t_batch, cond)       # [B, L, H]

            beta_t = self.betas[t]
            sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
            sqrt_one_minus_acp_t = self.sqrt_one_minus_alphas_cumprod[t]
            acp_t = self.alphas_cumprod[t]

            # predict x0
            x0_pred = (x_t - sqrt_one_minus_acp_t * eps_hat) / (acp_t.sqrt() + 1e-8)

            # DDPM posterior mean (simplified, DDIM-like step)
            x_t = acp_t.sqrt() * x0_pred + (1 - acp_t).sqrt() * eps_hat
            if t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + beta_t.sqrt() * noise

        return x0_pred  # final denoised hidden sequence

    def training_loss(self, x0_target, cond_hidden):
        """
        Standard noise-prediction loss.
        x0_target: [B, L, H] teacher-forced hidden targets (use your LLM-hidden or enc-projected targets)
        """
        B, L, H = x0_target.shape
        t = torch.randint(0, self.T, (B,), device=x0_target.device).long()
        noise = torch.randn_like(x0_target)
        x_t = (self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2) * x0_target +
               self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2) * noise)

        plan = cond_hidden['plan']                 # [B, H]
        dec  = cond_hidden['future_temporal']      # [B, L, H]
        plan_seq = plan.unsqueeze(1).expand(B, L, H)
        cond = self.cond_proj(torch.cat([plan_seq, dec], -1))

        eps_hat = self.eps_net(x_t, t, cond)
        return F.mse_loss(eps_hat, noise)
    
    
    def _shared_diffusion_step(self, x0_target, cond_hidden, t_override=None, clamp_t=False):
        B, L, H = x0_target.shape
        device = x0_target.device

        if t_override is None:
            t = torch.randint(0, self.T, (B,), device=device).long()
        else:
            t = t_override.to(device=device).long()
            if clamp_t:
                t = t.clamp(0, self.T - 1)

        noise = torch.randn_like(x0_target)
        sqrt_acp = self.sqrt_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)
        sqrt_om_acp = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1).unsqueeze(2)

        x_t = sqrt_acp * x0_target + sqrt_om_acp * noise

        plan = cond_hidden['plan']            # [B, H]
        dec  = cond_hidden['future_temporal'] # [B, L, H]
        plan_seq = plan.unsqueeze(1).expand(B, L, H)
        cond = self.cond_proj(torch.cat([plan_seq, dec], -1))

        eps_hat = self.eps_net(x_t, t, cond)
        return {
            "t": t,
            "noise": noise,
            "x_t": x_t,
            "cond": cond,
            "eps_hat": eps_hat,
            "sqrt_acp": sqrt_acp,
            "sqrt_om_acp": sqrt_om_acp,
        }

    def _masked_mse(self, pred, target, loss_mask=None):
        """MSE with optional token mask. loss_mask: [B, L], 1 for valid tokens."""
        if loss_mask is None:
            return F.mse_loss(pred, target)
        mask = loss_mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
        denom = (mask.sum() * pred.size(-1)).clamp_min(1.0)
        return ((pred - target) ** 2 * mask).sum() / denom

    def training_loss_new(self, x0_target, cond_hidden, t_override=None, loss_mask=None):
        """
        Standard noise-prediction loss.
        x0_target: [B, L, H]
        t_override: Optional LongTensor [B] specifying diffusion timestep per sample.
        loss_mask: Optional [B, L] mask; only valid future slots contribute.
        """
        step = self._shared_diffusion_step(
            x0_target,
            cond_hidden,
            t_override=t_override,
            clamp_t=True,
        )
        return self._masked_mse(step["eps_hat"], step["noise"], loss_mask=loss_mask)

    def denoise_and_training_loss(self, x0_target, cond_hidden, t_override=None, loss_mask=None):
        """
        Compute one shared diffusion step and return both x0_hat and noise-prediction loss.
        loss_mask: Optional [B, L] mask; only valid future slots contribute to diff_loss.
        """
        step = self._shared_diffusion_step(
            x0_target,
            cond_hidden,
            t_override=t_override,
            clamp_t=True,
        )
        x0_hat = (step["x_t"] - step["sqrt_om_acp"] * step["eps_hat"]) / (step["sqrt_acp"] + 1e-8)
        diff_loss = self._masked_mse(step["eps_hat"], step["noise"], loss_mask=loss_mask)
        return x0_hat, diff_loss

    def denoise_once(self, x0_target, cond_hidden, t_override=None):
        step = self._shared_diffusion_step(
            x0_target,
            cond_hidden,
            t_override=t_override,
            clamp_t=False,
        )
        x0_hat = (step["x_t"] - step["sqrt_om_acp"] * step["eps_hat"]) / (step["sqrt_acp"] + 1e-8)
        return x0_hat

