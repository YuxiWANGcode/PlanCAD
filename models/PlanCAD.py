import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import math

from .diffusion_id import GaussianDiffusionHead

class PlanHead(nn.Module):
    """
    Learnable plan generator (Route 2):
    hist_prompts: [B, 7, H]  (frozen LLM embeddings from past 7 days)
    -> plans_all: [B, 3, H]  (plan tokens for t, t+1, t+2)
    """
    def __init__(self, hidden_dim: int, plan_horizon: int = 3, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"PlanHead: hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.plan_horizon = plan_horizon
        self.hidden_dim = hidden_dim

        # 3 learnable queries -> produce 3 plan tokens
        self.plan_queries = nn.Parameter(torch.randn(plan_horizon, hidden_dim) * 0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hist_prompts: torch.Tensor) -> torch.Tensor:
        """
        hist_prompts: [B, 7, H]
        return:      [B, 3, H]
        """
        if hist_prompts.dim() != 3:
            raise ValueError(f"PlanHead expects hist_prompts [B,7,H], got {tuple(hist_prompts.shape)}")
        B, T, H = hist_prompts.shape
        if T != 7:
            # allow flexibility, but your design is 7 days
            pass
        if H != self.hidden_dim:
            raise ValueError(f"PlanHead hidden mismatch: got H={H}, expected {self.hidden_dim}")

        q = self.plan_queries.unsqueeze(0).expand(B, -1, -1)  # [B,3,H]
        attn_out, _ = self.attn(query=q, key=hist_prompts, value=hist_prompts)  # [B,3,H]
        x = self.ln(q + attn_out)
        x = x + self.ffn(x)
        return x

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.plan_weight_beta = float(getattr(configs, "plan_weight_beta", 1.0))
        # Basic initialization (same as before)
        self.token_len = configs.token_len
        self.token_num = int(configs.seq_len / configs.token_len) # Number of days in historical data
        
        # Device configuration
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
            
        # Load LLaMA
        self.llama = AutoModelForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            token=configs.token
        )
        
        # Freeze LLaMA
        for param in self.llama.parameters():
            param.requires_grad = False
            
        self.hidden_dim = self.llama.config.hidden_size
        
        user_embeds_size = 128
        times_embeds_size = 128
        latlon_emb_dim = 128
        place_embeds_size = 256
        drop_rate = 0.2
        # Embeddings
        self.user_embed = nn.Embedding(configs.num_users, user_embeds_size)
        self.tod_embed = nn.Embedding(48, times_embeds_size)
        self.dow_embed = nn.Embedding(7, times_embeds_size)
        self.place_embed = nn.Embedding(configs.num_classes, place_embeds_size)
        
        self.act = nn.GELU()
        # Lat/lon projection
        self.latlon_proj = nn.Sequential(
            nn.Linear(6, latlon_emb_dim),
            nn.LayerNorm(latlon_emb_dim),
            self.act,
            nn.Dropout(drop_rate)
        )
        
        # Calculate total embedding dimension
        self.struct_emb_dim = user_embeds_size + times_embeds_size * 2 + latlon_emb_dim + place_embeds_size # 640
        
        # Projection to LLaMA hidden dimension
        self.struct2hidden = nn.Sequential(
            nn.Linear(self.struct_emb_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.act,
            nn.Dropout(drop_rate)
        )
        self.dec_feature_dim = user_embeds_size + times_embeds_size * 2
        self.dec2hidden = nn.Sequential(
            nn.Linear(self.dec_feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.act,
            nn.Dropout(drop_rate)
        )
        
        self.full_embed_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            self.act,
            nn.Dropout(drop_rate),
        )
            
        
        # Day-level processing
        self.day_processor = DayLevelProcessor(self.hidden_dim)
        
        # Aggregation attention
        self.agg_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.dropout = nn.Dropout(drop_rate)
        self.pool_proj = nn.Linear(48, 1) 
        # Output classifier
        self.num_places = configs.num_classes
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.num_places),
        )
        self.hierarchical_attention = HierarchicalAttention(self.hidden_dim)

        self.use_diffusion = configs.use_diffusion
        self.sample_steps = int(getattr(configs, "sample_steps", 50))
        if self.use_diffusion:
            
            self.diffusion = GaussianDiffusionHead(
                hidden_dim=self.hidden_dim,
                token_len=self.token_len * int(getattr(configs, "plan_horizon", 3)),
                timesteps=int(getattr(configs, "diffusion_steps", 1000)),
                beta_start=float(getattr(configs, "beta_start", 1e-4)),
                beta_end=float(getattr(configs, "beta_end", 2e-2)),
            )
        # === global plan projection ===
        self.plan_linear = nn.Linear(self.token_num * self.hidden_dim,
                             self.hidden_dim)
        
        self.plan_attn = nn.Linear(self.hidden_dim, 1)
        # Plan prompt embedding dimension (fixed by precomputed LLM embeddings)
        self.prompt_dim = 2048

        # Horizon length for plan aggregation (must match rollout horizon when used)
        # self.plan_horizon = getattr(args, "plan_horizon", 3)
        self.plan_horizon = 3 #change here for 3 days or 7 days

        # Horizon plan aggregator: concat (K*H) -> H
        self.plan_agg = torch.nn.Sequential(
            torch.nn.Linear(self.plan_horizon * self.prompt_dim, self.prompt_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.prompt_dim, self.prompt_dim),
        )
        # hist_prompts [B,7,H] -> plans_all [B,3,H]
        # ---------------------------------------------------------
        plan_heads = int(getattr(configs, "plan_heads", 8))
        self.plan_head = PlanHead(
            hidden_dim=self.hidden_dim,
            plan_horizon=self.plan_horizon,  # 3
            num_heads=plan_heads,
            dropout=0.1
        )        

        # ---------------------------------------------------------
        # Horizon-aware Actor: Day-1 queries attend plan_tokens
        # queries: [B,48,H], plan_tokens: [B,K,H] where K in {1,2,3}
        # ---------------------------------------------------------
        plan_heads = int(getattr(configs, "plan_heads", 8))
        if self.hidden_dim % plan_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} must be divisible by plan_heads={plan_heads}")

        self.plan_cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=plan_heads,
            dropout=0.1,
            batch_first=True
        )
        self.post_attn_ln = nn.LayerNorm(self.hidden_dim)
        self.post_attn_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(0.1),
        )
        

    def aggregate_day(self, enc_embeds):
        """
        Optimized day-level aggregation with parallel processing
        Args:
            enc_embeds (torch.Tensor): [B, 336, hidden_dim]
        Returns:
            torch.Tensor: [B, 7, hidden_dim] aggregated day embeddings
        """
        B, T, H = enc_embeds.size()
        assert T == 336, f"Expected T=336, got T={T}"
        
        # Reshape to [B, 7, 48, H]
        daily_sequences = enc_embeds.view(B, 7, 48, H)
        
        # Process all days at once by reshaping
        # Reshape to [B * 7, 48, H]
        reshaped_sequences = daily_sequences.reshape(-1, 48, H)
        
        # Process all sequences in parallel
        processed_sequences = self.day_processor(reshaped_sequences)  # [B * 7, 48, H]
        
        # Compute attention weights for all days in parallel
        attention_scale = math.sqrt(H) + 1e-6
        attention_logits = torch.matmul(processed_sequences, processed_sequences.transpose(-2, -1)) / attention_scale
        attention_weights = torch.softmax(attention_logits, dim=-1)  # [B * 7, 48, 48]
        
        # Apply attention and mean pooling
        attended_sequences = torch.matmul(attention_weights, processed_sequences)  # [B * 7, 48, H]
        day_embeddings = self.pool_proj(attended_sequences.transpose(-1, -2)).squeeze(-1)
        
        # Reshape back to [B, 7, H]
        day_embeds = day_embeddings.view(B, 7, H)
        
        # Final inter-day attention
        day_embeds, _ = self.agg_attention(day_embeds, day_embeds, day_embeds)
        
        return day_embeds

    def encode_struct(self, x_enc):
        """
        Encode historical data x_enc.

        Args:
            x_enc (torch.Tensor): [B, 336, 7] tensor containing 
                                  [user_id, time_of_day, day_of_week, ?, lat, lon, place_id]

        Returns:
            torch.Tensor: [B, 336, hidden_dim] encoded embeddings.
        """
        user_id = x_enc[..., 0].long()
        tod     = x_enc[..., 1].long()
        dow     = x_enc[..., 2].long()
        latlon  = x_enc[..., 4:6].float()
        sin, cos = torch.sin(latlon), torch.cos(latlon)
        latlon = torch.cat([sin, cos, latlon], dim=-1)
        place   = x_enc[..., 6].long()

        # Embeddings
        emb_user   = self.user_embed(user_id)         # [B, 336, user_embeds_size]
        emb_tod    = self.tod_embed(tod)              # [B, 336, times_embeds_size]
        emb_dow    = self.dow_embed(dow)              # [B, 336, times_embeds_size]
        emb_place  = self.place_embed(place)          # [B, 336, place_embeds_size]
        emb_latlon = self.latlon_proj(latlon)         # [B, 336, latlon_emb_dim]

        # Concatenate and project to hidden_dim
        struct_concat = torch.cat([emb_user, emb_tod, emb_dow, emb_latlon, emb_place], dim=-1)  # [B, 336, struct_emb_dim]
        struct_embeds = self.struct2hidden(struct_concat)  # [B, 336, hidden_dim]
        return struct_embeds

    def encode_future_struct(self, x_dec_f):
        """
        Encode future day data x_dec_f.

        Args:
            x_dec_f (torch.Tensor): [B, 48, 3] tensor containing 
                                     [user_id, time_of_day, day_of_week]

        Returns:
            torch.Tensor: [B, 48, hidden_dim] encoded embeddings.
        """
        user_id = x_dec_f[..., 0].long()
        tod     = x_dec_f[..., 1].long()
        dow     = x_dec_f[..., 2].long()

        # Embeddings
        emb_user = self.user_embed(user_id)          # [B, 48, user_embeds_size]
        emb_tod  = self.tod_embed(tod)               # [B, 48, times_embeds_size]
        emb_dow  = self.dow_embed(dow)               # [B, 48, times_embeds_size]

        # Concatenate and project to hidden_dim
        concat_future = torch.cat([emb_user, emb_tod, emb_dow], dim=-1)  # [B, 48, dec_feature_dim]
        dec_embeds = self.dec2hidden(concat_future)  # [B, 48, hidden_dim]
        return dec_embeds
    
    def build_weighted_plan_vec(self, plan_tokens, plan_offset=0, beta=1.0):
        """
        plan_tokens: [B, K, H]
        plan_offset: int, current day offset within selected segment
        beta: larger -> stronger emphasis on current aligned token
        returns: [B, H]
        """
        B, K, H = plan_tokens.shape

        offset = max(0, min(int(plan_offset), K - 1))

        idx = torch.arange(K, device=plan_tokens.device, dtype=plan_tokens.dtype)  # [K]
        weights = torch.exp(-beta * torch.abs(idx - float(offset)))                # [K]
        weights = weights / (weights.sum() + 1e-8)

        plan_vec = (plan_tokens * weights.view(1, K, 1)).sum(dim=1)                # [B, H]
        return plan_vec

    def build_plan_vec_from_tokens(self, plan_tokens):
        """
        Mean-pool only nonzero plan tokens. This supports masked plan sequences such as [P1, P2, 0].
        plan_tokens: [B, K, H]
        returns: [B, H]
        """
        token_mask = (plan_tokens.abs().sum(dim=-1) > 0).to(plan_tokens.dtype)  # [B,K]
        denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)              # [B,1]
        return (plan_tokens * token_mask.unsqueeze(-1)).sum(dim=1) / denom

    def build_plan_per_slot(self, plan_tokens, future_len):
        """
        Align daily plan tokens to fine-grained future slots.
        plan_tokens: [B, K, H], where token j corresponds to future day j.
        returns: [B, future_len, H]. Pads with zeros if future_len > K*token_len.
        """
        B, K, H = plan_tokens.shape
        plan_per_slot = plan_tokens.repeat_interleave(self.token_len, dim=1)  # [B,K*48,H]
        if plan_per_slot.size(1) < future_len:
            pad_len = future_len - plan_per_slot.size(1)
            pad = plan_per_slot.new_zeros(B, pad_len, H)
            plan_per_slot = torch.cat([plan_per_slot, pad], dim=1)
        return plan_per_slot[:, :future_len, :]
    
    def forward(self, x_enc, x_mark_enc, x_dec_f, x_dec=None, x_mark_dec=None, plan_offset=0, loss_mask=None):
        """
        Forward pass of the model.

        Args:
            x_enc (torch.Tensor): [B, 336, 7] historical data.
            x_mark_enc (torch.Tensor): [B, 7, hidden_dim] daily prompt embeddings.
            x_dec_f (torch.Tensor): [B, L, 3/4] future segment features.
            x_dec (torch.Tensor, optional): Not used. Included for compatibility.
            x_mark_dec (torch.Tensor): [B, hidden_dim] task description prompt.

        Returns:
            torch.Tensor: [B, L, num_places] logits for place prediction.
        """

        # ------------------------------------------------------------------
        # 1) Encode historical data -> [B, 336, hidden_dim]
        # ------------------------------------------------------------------
        enc_embeds = self.encode_struct(x_enc)  # [B, 336, hidden_dim]
        enc_embeds = self.hierarchical_attention(enc_embeds)
        
        # ------------------------------------------------------------------
        # 2) Aggregate into day-level tokens -> [B, 7, hidden_dim]
        # ------------------------------------------------------------------
        day_embeds = self.aggregate_day(enc_embeds)  # [B, 7, hidden_dim]

        # ---------------------------------------------------------
        # Route 2: build plan tokens ONLY from past-7-day hist prompts
        # x_mark_enc: [B,7,H] frozen LLM embeddings (past 7 days)
        # plans_all:  [B,3,H] = [P_t, P_{t+1}, P_{t+2}]
        # ---------------------------------------------------------
        hist_prompts = x_mark_enc  # [B,7,H]
        
        # Decide what to use as "plan_tokens" for the rest of forward
        # - If caller passes x_mark_dec as [B,K,H], use it (typically prefix of plans_all).
        # - Else default to K=1 from plans_all.
        if x_mark_dec is None:
            plans_all = self.plan_head(F.normalize(hist_prompts, dim=-1))  # [B,3,H]
            plan_tokens = plans_all[:, :1, :]  # [B,1,H]
        elif x_mark_dec.dim() == 3:
            plan_tokens = x_mark_dec           # [B,K,H]
        elif x_mark_dec.dim() == 2:
            plan_tokens = x_mark_dec.unsqueeze(1)  # [B,1,H] legacy
        else:
            raise ValueError(f"Unexpected x_mark_dec shape: {tuple(x_mark_dec.shape)}")

        # Segment-level plan summary. Masked zero tokens are ignored.
        plan_vec = self.build_plan_vec_from_tokens(plan_tokens)  # [B,H]

        # ------------------------------------------------------------------
        # 3) Add x_mark_enc to day embeddings -> [B, 7, hidden_dim]
        # ------------------------------------------------------------------
        day_embeds = F.normalize(day_embeds, dim=-1)
        x_mark_enc = F.normalize(x_mark_enc, dim=-1)
        
        day_embeds = day_embeds + x_mark_enc  # [B, 7, hidden_dim]
        
        # ------------------------------------------------------------------
        # 4) Encode future segment features -> [B, L, hidden_dim]
        # ------------------------------------------------------------------
        future_len = x_dec_f.size(1)
        dec_embeds = self.encode_future_struct(x_dec_f)  # [B, L, hidden_dim]

        # ------------------------------------------------------------------
        # 5) Align daily plan tokens to future slots.
        #    For a 3-day canvas, P1 conditions slots 1--48, P2 slots 49--96, etc.
        #    Unselected plan tokens can be zero-masked by the caller.
        # ------------------------------------------------------------------
        plan_vec = F.normalize(plan_vec, dim=-1)                         # [B,H]
        plan_per_slot = self.build_plan_per_slot(plan_tokens, future_len) # [B,L,H]
        plan_per_slot = F.normalize(plan_per_slot, dim=-1)

        dec_embeds = F.normalize(dec_embeds, dim=-1)                    # [B,L,H]
        dec_embeds = dec_embeds + plan_per_slot
        
        # ---------------------------------------------------------
        # Actor cross-attention (Option A):
        # Future segment queries attend plan_tokens (K tokens).
        # This is what makes logits(C1) != logits(C3).
        # ---------------------------------------------------------
        # Ensure plan_tokens is [B,K,H] (already decided above)
        if plan_tokens.dim() != 3:
            raise ValueError(f"plan_tokens must be [B,K,H], got {tuple(plan_tokens.shape)}")

        # Cross-attn: z = LN(q + Attn(q, plan_tokens)) + FFN
        q = F.normalize(dec_embeds, dim=-1)                 # [B,L,H]
        kv = F.normalize(plan_tokens, dim=-1)               # [B,K,H]
        attn_out, _ = self.plan_cross_attn(query=q, key=kv, value=kv)  # [B,L,H]
        z = self.post_attn_ln(q + attn_out)
        z = z + self.post_attn_ffn(z)                       # [B,L,H]

        if self.use_diffusion:
            # print("---------Using diffusion model.-------------------")
            # ------------------------------------------------------------------
            # 6) Concatenate historical and future embeddings -> [B, 55, hidden_dim]
            # ------------------------------------------------------------------
            full_embeds = torch.cat([day_embeds, dec_embeds], dim=1)  # [B, 7 + L, hidden_dim]
            full_embeds = self.full_embed_mlp(full_embeds) + full_embeds  # [B, 7 + L, hidden_dim]

            # ------------------------------------------------------------------
            # 7) Pass through LLaMA
            # ------------------------------------------------------------------
            outputs = self.llama.model(inputs_embeds=full_embeds)
            # hidden_states = outputs.last_hidden_state
            hidden_states = outputs.last_hidden_state.detach() # detach to avoid gradient flow from LLaMA 

            # ------------------------------------------------------------------
            # 8) Extract future hidden states -> [B, L, hidden_dim]
            # ------------------------------------------------------------------
            next_day_states = hidden_states[:, -future_len:, :]  # [B, L, hidden_dim]
            teacher_future = next_day_states
            
            # teacher_logits = self.classifier(self.dropout(next_day_states))

            # ------------------------------------------------------------------
            # 9) Apply diffusion classification head -> [B, 48, num_places]
            # ----------------------------------------------------------
            # Option A Route 2:
            # plan comes from plan_tokens (generated from frozen hist prompts),
            # NOT from LLaMA hidden pooling.
            # ----------------------------------------------------------
            
            plan = plan_vec
            B = plan.size(0)


            if not hasattr(self, "_step_count"):
                self._step_count = 0
            self._step_count += 1

            # tune these thresholds when needed
            if self._step_count < 1000:
                t_logits = torch.zeros(B, device=teacher_future.device, dtype=torch.long)
            elif self._step_count < 2000:
                t_max = 10
                t_logits = torch.randint(0, t_max, (B,), device=teacher_future.device, dtype=torch.long)
            elif self._step_count < 4000:
                t_max = 50
                t_logits = torch.randint(0, t_max, (B,), device=teacher_future.device, dtype=torch.long)
            elif self._step_count < 8000:
                t_max = 100
                t_logits = torch.randint(0, t_max, (B,), device=teacher_future.device, dtype=torch.long)
            elif self._step_count < 16000:
                t_max = 200
                t_logits = torch.randint(0, t_max, (B,), device=teacher_future.device, dtype=torch.long)
            else:
                t_max = min(300, self.diffusion.T)   # safe cap
                t_logits = torch.randint(0, t_max, (B,), device=teacher_future.device, dtype=torch.long)

            cond_hidden = {
                "plan": plan,          # [B,H]
                "future_temporal": z    # [B,L,H]  (cross-attended)
            }

            # hidden_seq = self.diffusion.sample(cond_hidden, steps=self.sample_steps) # [B, self.token_len, H]
            # hidden_seq = torch.nn.functional.layer_norm(hidden_seq, hidden_seq.shape[-1:])
            # feat_loss = ((hidden_seq - teacher_future.detach())**2).mean()

            # student_logits = self.classifier(self.dropout(hidden_seq))  # [B, 48, num_places]
            x0_hat, diff_loss = self.diffusion.denoise_and_training_loss(
                teacher_future,   # teacher-forcing hidden targets
                cond_hidden,
                t_logits,
                loss_mask=loss_mask
            )

            student_logits = self.classifier(self.dropout(x0_hat))
            
            # ------------------------------------------------------------------
            # 10) Apply diffusion model to generate diff_loss
            # ------------------------------------------------------------------
            
            mse_x0 = torch.nn.functional.mse_loss(x0_hat, teacher_future, reduction="mean")
            rmse_x0 = torch.sqrt(mse_x0 + 1e-8)
            # beta = 1
            # diff_loss = diff_loss + beta * feat_loss
            # print("diff_loss:", diff_loss.item())
            # print("feat_loss:", feat_loss.item())

            # return student_logits,diff_loss,mse_x0,rmse_x0
            return student_logits, diff_loss
        else:
            # ------------------------------------------------------------------
            # 6) Concatenate historical and future embeddings -> [B, 55, hidden_dim]
            # ------------------------------------------------------------------
            full_embeds = torch.cat([day_embeds, dec_embeds], dim=1)  # [B, 7 + L, hidden_dim]
            full_embeds = self.full_embed_mlp(full_embeds) + full_embeds  # [B, 7 + L, hidden_dim]

            # ------------------------------------------------------------------
            # 7) Pass through LLaMA
            # ------------------------------------------------------------------
            outputs = self.llama.model(inputs_embeds=full_embeds)
            hidden_states = outputs.last_hidden_state

            # ------------------------------------------------------------------
            # 8) Extract future hidden states -> [B, L, hidden_dim]
            # ------------------------------------------------------------------
            next_day_states = hidden_states[:, -future_len:, :]  # [B, L, hidden_dim]

            # ------------------------------------------------------------------
            # 9) Apply classification head -> [B, L, num_places]
            # ------------------------------------------------------------------
            logits = self.classifier(self.dropout(next_day_states))  # [B, L, num_places]

            return logits,None

class DayLevelProcessor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.act = nn.GELU()
        # Temporal convolution for local patterns
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            self.act,
        )
        self.feature_norm = nn.LayerNorm(hidden_dim)
        # Self-attention for capturing temporal dependencies
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final processing
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            self.act,
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.temporal_pos = nn.Parameter(torch.zeros(1, 48, hidden_dim))
        nn.init.xavier_uniform_(self.temporal_pos)
        
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: [B, L, H]. Historical day blocks usually have L=48;
        # support other lengths defensively to avoid shape errors.
        B, L, H = x.shape

        if self.temporal_pos.size(1) == L:
            pos = self.temporal_pos
        elif self.temporal_pos.size(1) > L:
            pos = self.temporal_pos[:, :L, :]
        else:
            repeat_times = (L + self.temporal_pos.size(1) - 1) // self.temporal_pos.size(1)
            pos = self.temporal_pos.repeat(1, repeat_times, 1)[:, :L, :]

        x = x + pos
        
        # Temporal convolution
        x_conv = x.transpose(1, 2)  # [B, H, 48]
        x_conv = self.temporal_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)  # [B, 48, H]
        x_conv = self.feature_norm(x_conv)
        
        # Self-attention with skip connection
        x_attn, _ = self.self_attention(x_conv, x_conv, x_conv)
        gate_weights = self.gate(torch.cat([x, x_attn], dim=-1))
        x = x + gate_weights * x_attn
        
        x = self.norm1(x)  # Norm AFTER residual
        x = self.norm2(x + self.mlp(x))
        
        return x

class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.local_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # Local attention within each day
        B, T, H = x.shape
        daily_x = x.view(B * 7, 48, H)
        local_out = self.local_attention(daily_x, daily_x, daily_x)[0]
        # local_out = local_out.view(B, T, H) 
        local_out = local_out.reshape(B, T, H)
        x = self.norm1(x + local_out)
        
        # Global attention across all timesteps
        global_out = self.global_attention(x, x, x)[0]
        return self.norm2(x + global_out)

class PromptFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, sequence, prompt):
        # sequence: [B, seq_len, H]
        # prompt: [B, prompt_len, H]
        attn_out, _ = self.cross_attn(
            query=sequence,
            key=prompt,
            value=prompt
        )
        return self.norm(sequence + attn_out)
