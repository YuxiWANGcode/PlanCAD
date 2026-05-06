from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import wandb
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import csv
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

smooth_fn = SmoothingFunction().method1

def id_to_xy(pid):
    pid = int(pid)
    if pid < 0 or pid >= 40000:
        return (0, 0)
    x = pid // 200 + 1
    y = pid % 200 + 1
    return (x, y)

def compute_bleu(pred_ids, true_ids):
    pred_tokens = [str(int(x)) for x in pred_ids]
    true_tokens = [str(int(x)) for x in true_ids]
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return 0.0
    return sentence_bleu(
        [true_tokens],
        pred_tokens,
        smoothing_function=smooth_fn
    )

def compute_dtw(pred_ids, true_ids):
    pred_traj = [id_to_xy(x) for x in pred_ids]
    true_traj = [id_to_xy(x) for x in true_ids]
    distance, _ = fastdtw(pred_traj, true_traj, dist=euclidean)
    return float(distance)

def compute_rhythm_dtw(pred_ids, true_ids, padding_value=40000):
    pred = np.array(pred_ids)
    true = np.array(true_ids)

    valid_mask = true != padding_value
    pred_valid = pred[valid_mask]
    true_valid = true[valid_mask]

    if len(true_valid) == 0:
        return 0.0

    true_rows = true_valid // 200
    true_cols = true_valid % 200
    pred_rows = pred_valid // 200
    pred_cols = pred_valid % 200

    distances = np.sqrt((true_rows - pred_rows) ** 2 + (true_cols - pred_cols) ** 2) * 500
    return float(np.mean(distances))


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN found in {name}")

def topk_mrr_from_logits(
    logits: torch.Tensor,   # [N, C]
    targets: torch.Tensor,  # [N]
    padding_idx: int,
    ks=(1, 3, 5, 10)
):
    """
    Compute top-k accuracy and MRR from logits. For one day or multiple days.
    """
    device = logits.device
    valid_mask = targets != padding_idx
    logits = logits[valid_mask]
    targets = targets[valid_mask]

    total = targets.numel()
    if total == 0:
        return {f'acc@{k}': 0.0 for k in ks} | {'MRR': 0.0}, 0

    maxk = max(ks)
    _, topk = logits.topk(maxk, dim=1, sorted=True)      # [N, maxk]
    correct = topk.eq(targets.unsqueeze(1))              # [N, maxk]

    # acc@k
    out = {}
    for k in ks:
        out[f'acc@{k}'] = (correct[:, :k].any(dim=1).float().mean().item())

    # MRR: rank of the first hit
    ranks = torch.arange(1, maxk + 1, device=device).view(1, -1)     # [1, maxk]
    hit_ranks = correct.float() * ranks                               # [N, maxk]
    hit_ranks[hit_ranks == 0] = float('inf')
    first_rank = hit_ranks.min(dim=1).values                          # [N]
    out['MRR'] = (1.0 / first_rank).mean().item()

    return out, total

class HorizonCritic(nn.Module):
    def __init__(self, in_dim=4, hidden_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat):   # feat: [..., 10]
        return self.net(feat).squeeze(-1)   # [...]

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        wandb.init(
            project="hummob",  # Change project name as needed
            config={                              # Log hyperparameters
                "learning_rate": self.args.learning_rate,
                "batch_size": self.args.batch_size,
                "epochs": self.args.train_epochs,
                "model": self.args.model,
            },
            # mode="disabled" 
            mode="online"
        )
        train_data, train_loader = self._get_data(flag='train')
        self.args.num_classes = train_data.get_num_class()
        self.num_classes = train_data.get_num_class()

        critic_feature_mode = getattr(self.args, "critic_feature_mode", "k_planvecnorm")
        if critic_feature_mode == "k_planvecnorm":
            critic_in_dim = 2
        elif critic_feature_mode == "plan_only":
            critic_in_dim = 4
        elif critic_feature_mode == "traj_only":
            critic_in_dim = 5
        elif critic_feature_mode == "hybrid":
            critic_in_dim = 8
        else:
            raise ValueError(f"Unknown critic_feature_mode: {critic_feature_mode}")

        self.critic = HorizonCritic(
            in_dim=critic_in_dim,
            hidden_dim=getattr(self.args, "critic_hidden_dim", 32),
            dropout=getattr(self.args, "critic_dropout", 0.1)
        ).to(self.device)
        
        print(f"Dataset: {self.args.data}, City: {self.args.city}, Num Users: {self.args.num_users}, Num Classes: {self.args.num_classes}")

        
        
    def _build_model(self):
        train_data, train_loader = self._get_data(flag='train')
        self.args.num_classes = train_data.get_num_class()
        self.args.num_users = train_data.get_num_users()
        
        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.path != '':
            model.load_state_dict(torch.load(self.args.path),strict=False)
            
        if self.args.use_multi_gpu:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        else:
            self.device = torch.device(f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu')
            model = model.to(self.device)
        return model

    def _get_data(self, flag, **kwargs):
        data_set, data_loader = data_provider(self.args, flag, **kwargs)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            p_list.append(p)
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print(n, p.dtype, p.shape)

        for n, p in self.critic.named_parameters():
            if not p.requires_grad:
                continue
            p_list.append(p)
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print(f"[critic] {n}", p.dtype, p.shape)

        model_optim = optim.AdamW(
            [{'params': p_list}],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))

        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        if self.args.data == 'yj':
            criterion = nn.CrossEntropyLoss(ignore_index=40000, label_smoothing=self.args.label_smoothing)
        elif self.args.data == 'us':
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        return criterion

    
    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = 0
        total_samples = 0
        correct_counts = torch.zeros(10, device=self.device)  # Track top-1 through top-10 in one tensor
        mrr_sum = 0
        
        self.model.eval()
        self.critic.eval()
        with torch.no_grad():
            # for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
            for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark, batch_future_full_7d) in enumerate(vali_loader):
                # Move data to device all at once
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y_f = batch_y_f.float().to(self.device, non_blocking=True)
                batch_y = batch_y.long().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                batch_future_full_7d = batch_future_full_7d.float().to(self.device, non_blocking=True)  # [B,336,7] or empty
                # Forward pass
                with torch.cuda.amp.autocast():
                    # outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
                    if self.args.rollout_days > 1:
                        selector_mode = "oracle"
                        if is_test:
                            selector_mode = getattr(self.args, "critic_infer_mode", "oracle")
                        else:
                            if getattr(self.args, "save_best_with_critic", False):
                                selector_mode = getattr(self.args, "critic_infer_mode", "oracle")

                        outputs, batch_y_roll = self.rollout_predict_highfreq(
                            vali_data,
                            batch_x,
                            batch_x_mark,
                            batch_y_mark,
                            batch_future_full_7d,
                            rollout_days=self.args.rollout_days,
                            plan_mode=getattr(self.args, "plan_mode", "horizon_tokens"),
                            K_eval=getattr(self.args, "plan_horizon", 3),
                            selector_mode=selector_mode,
                        )
                        batch_y = batch_y_roll.view(-1)
                    else:
                        # plan_mode
                        if getattr(self.args, "plan_mode", "horizon_tokens") == "horizon_tokens":
                            with torch.no_grad():
                                plans_all = self.model.plan_head(batch_x_mark)     
                                plan_tokens = plans_all[:, :getattr(self.args, "plan_horizon", 3), :]
                            outputs, _ = self.model(batch_x, batch_x_mark, batch_y_f, None, plan_tokens)
                        else:
                            outputs, _ = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)


                # Handle test vs validation sequence lengths
                if is_test and self.args.rollout_days == 1:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:]
                
                # Reshape for computation
                batch_y = batch_y.view(-1)
                outputs = outputs.view(-1, outputs.size(-1))

                # Filter valid samples (handle padding)
                padding_idx = 40000 if self.args.data == 'yj' else 0
                valid_mask = batch_y != padding_idx
                valid_outputs = outputs[valid_mask]
                valid_targets = batch_y[valid_mask]
                
                # Calculate loss efficiently
                loss = criterion(valid_outputs, valid_targets)
                total_loss += loss.item() * valid_targets.size(0)
                
                # Get top-k predictions efficiently (k=10)
                # Yuxi change this for multiple days
                metrics_dict, n_valid = topk_mrr_from_logits(
                    outputs, batch_y, padding_idx=padding_idx, ks=(1, 3, 5, 10)
                )
                total_samples += n_valid
                correct_counts[0] += metrics_dict['acc@1'] * n_valid
                correct_counts[2] += metrics_dict['acc@3'] * n_valid
                correct_counts[4] += metrics_dict['acc@5'] * n_valid
                correct_counts[9] += metrics_dict['acc@10'] * n_valid
                mrr_sum += metrics_dict['MRR'] * n_valid


                if (i + 1) % 500 == 0:  # Reduced frequency of progress updates
                    print(f"\tValidation batch: {i + 1}/{len(vali_loader)}")
                    torch.cuda.empty_cache()
        torch.cuda.empty_cache()
                
        # Calculate final metrics efficiently
        # accuracies = {
        #     f'acc@{k}': (correct_counts[k-1] / total_samples).item()
        #     for k in (1, 3, 5, 10)
        # }
        accuracies = {
            'acc@1': (correct_counts[0] / total_samples).item(),
            'acc@3': (correct_counts[2] / total_samples).item(),
            'acc@5': (correct_counts[4] / total_samples).item(),
            'acc@10': (correct_counts[9] / total_samples).item(),
            'MRR': (mrr_sum / total_samples)
        }
        # accuracies['MRR'] = mrr_sum / total_samples
        
        avg_loss = total_loss / total_samples
        accuracy = accuracies['acc@1']

        return avg_loss, accuracy, accuracies



    def train(self, setting):
        """
        Train the model to predict the next POI ID.
        Args:
            setting (str): Name for saving checkpoints and logs.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
            
        print(f'Best model will be saved to {path}') 

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        best_val_accuracy = 0.0

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler(init_scale=8)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            total_loss = torch.tensor(0.0, device=self.device)
            total_correct = torch.tensor(0, device=self.device)
            total_samples = torch.tensor(0, device=self.device)

            self.model.train()
            self.critic.train()
            epoch_time = time.time()
            # for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # epsilon decay (linear)
            epsilon_start = 0.15
            epsilon_end = 0.02

            if self.args.train_epochs > 1:
                progress = epoch / (self.args.train_epochs - 1)
            else:
                progress = 1.0

            self.current_epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

            print(f"[epsilon] epoch {epoch+1}, epsilon = {self.current_epsilon:.4f}")
            for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark, batch_future_full_7d) in enumerate(train_loader):

                # print(f"[debug] start batch {i}")
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                batch_y_f = batch_y_f.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_future_full_7d = batch_future_full_7d.float().to(self.device, non_blocking=True)  # [B,336,7] or empty
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                critic_loss = None

                if getattr(self.args, "train_rollout_days", 1) > 1:
                    outputs, batch_y_roll, diff_loss, critic_loss = self.rollout_train_teacher_forced(
                        train_data,
                        batch_x,
                        batch_future_full_7d,
                        rollout_days=self.args.train_rollout_days,
                        plan_mode=self.args.train_plan_mode
                    )
                    batch_y = batch_y_roll
                else:
                    # train_rollout_days == 1  (7->1 task)
                    if getattr(self.args, "train_plan_mode", "highfreq") == "horizon_tokens":
                        plans_all = self.model.plan_head(batch_x_mark)  # [B,3,H], batch_x_mark is [B,7,H]
                        K = int(torch.randint(low=1, high=4, size=(1,), device=batch_x.device).item())  # 1/2/3
                        plan_tokens = plans_all[:, :K, :].contiguous()  # [B,K,H]
                        outputs, diff_loss = self.model(batch_x, batch_x_mark, batch_y_f, None, plan_tokens)
                    else:
                        outputs, diff_loss = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)

                batch_y = batch_y.view(-1)
                
                try:
                    outputs = outputs.view(-1, outputs.size(-1))
                except RuntimeError as e:
                    outputs = outputs.reshape(-1, outputs.size(-1)).float()
                
                # loss = criterion(outputs, batch_y)
                ce_loss = criterion(outputs, batch_y)
                loss = ce_loss

                if diff_loss is not None:
                    loss = loss + self.args.lambda_distill * diff_loss

                if critic_loss is not None:
                    loss = loss + getattr(self.args, "critic_loss_weight", 1.0) * critic_loss
                    

                if (i + 1) % 500 == 0:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                        

                if self.args.use_amp:
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        print("NaN detected in loss before loss computation!")
                        continue
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    if self.args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    model_optim.step()

                if (i + 1) % 100 == 0:
                    torch.cuda.empty_cache()

                total_loss += loss.detach()
                _, predicted = torch.max(outputs, dim=1)
                if not self.args.label_missing:
                    total_correct += (predicted == batch_y).sum()
                    total_samples += batch_y.size(0)
                else:
                    valid_mask = batch_y != (self.num_classes - 1)
                    total_correct += ((predicted == batch_y) & valid_mask).sum().item()
                    total_samples += valid_mask.sum().item()

            
            # if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
             
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)    

            train_loss = (total_loss / total_samples).item()
            train_accuracy = (total_correct / total_samples).item()
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(f'Training time: {time.time() - epoch_time:.2f}s')
            
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': self.model.state_dict(),
            #     'optimizer_state_dict': model_optim.state_dict(),
            #     'scheduler_state_dict': scheduler.state_dict(),
            # }, os.path.join(path, 'current_checkpoint.pth'))

            wandb.log({
                        "iteration_loss": train_loss,
                        "iteration_accuracy": train_accuracy,
                        "iteration": i + 1,
                        "epoch": epoch + 1
                    })
            vali_loss, vali_accuracy, vali_metrics = self.vali(vali_data, vali_loader, criterion)
            torch.cuda.empty_cache()
            test_loss, test_accuracy, test_metrics = self.vali(test_data, test_loader, criterion, is_test=True)
            torch.cuda.empty_cache()

            if vali_accuracy > best_val_accuracy:
                best_val_accuracy = vali_accuracy
                if not self.args.use_multi_gpu or (self.args.use_multi_gpu and dist.get_rank() == 0):
                    best_model_path = os.path.join(path, 'best_checkpoint.pth')
                    
                    save_obj = {
                        "model_state_dict": self.model.state_dict(),
                        "critic_state_dict": self.critic.state_dict(),
                    }
                    torch.save(save_obj, best_model_path)
                    print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
                    full_test_acc, full_test_metrics = self.test(setting, test=0, save_analysis=True)
                    torch.cuda.empty_cache()
            else: 
                full_test_acc, full_test_metrics = self.test(setting, test=0, save_analysis=False)
                torch.cuda.empty_cache()

            log_dict = {
                "vali_loss": vali_loss,
                "vali_accuracy": vali_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "iteration": i + 1,
                "epoch": epoch + 1
            }

            if full_test_metrics is not None:
                log_dict.update({
                    "full_test_acc@1": full_test_metrics.get("acc@1", 0.0),
                    "full_test_acc@3": full_test_metrics.get("acc@3", 0.0),
                    "full_test_acc@5": full_test_metrics.get("acc@5", 0.0),
                    "full_test_acc@10": full_test_metrics.get("acc@10", 0.0),
                    "full_test_MRR": full_test_metrics.get("MRR", 0.0),
                    "full_test_BLEU": full_test_metrics.get("BLEU", 0.0),
                    "full_test_DTW": full_test_metrics.get("DTW", 0.0),
                })

            wandb.log(log_dict)
            if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                print("Epoch: {}, Steps: {} | Train Loss: {:.4f} Vali Loss: {:.4f} Test Loss: {:.4f} | | Train Acc: {:.4f} Vali Acc: {:.4f} Test Acc: {:.4f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss, train_accuracy, vali_accuracy, test_accuracy))
                print(f"Vali: Acc@1: {vali_metrics['acc@1']:.4f}, Acc@3: {vali_metrics['acc@3']:.4f}, Acc@5: {vali_metrics['acc@5']:.4f}, Acc@10: {vali_metrics['acc@10']:.4f}, MRR: {vali_metrics['MRR']:.4f}")
                print(f"Test: Acc@1: {test_metrics['acc@1']:.4f}, Acc@3: {test_metrics['acc@3']:.4f}, Acc@5: {test_metrics['acc@5']:.4f}, Acc@10: {test_metrics['acc@10']:.4f}, MRR: {test_metrics['MRR']:.4f}")
                if full_test_metrics is not None:
                    print(f"Full Test: Acc@1: {full_test_metrics['acc@1']:.4f}, "
                        f"Acc@3: {full_test_metrics['acc@3']:.4f}, "
                        f"Acc@5: {full_test_metrics['acc@5']:.4f}, "
                        f"Acc@10: {full_test_metrics['acc@10']:.4f}, "
                        f"MRR: {full_test_metrics['MRR']:.4f}, "
                        f"BLEU: {full_test_metrics['BLEU']:.4f}, "
                        f"DTW: {full_test_metrics['DTW']:.4f}")
                print("Epoch: {} cost time: {:.0f}s".format(epoch + 1, time.time() - epoch_time))  
            if self.args.enable_early_stopping:
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                        print("Early stopping")
                    break
            if self.args.cosine:
                scheduler.step()
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.use_multi_gpu:
                train_loader.sampler.set_epoch(epoch + 1)
            
            torch.cuda.empty_cache()
            print(f"Epoch time cost: {time.time() - epoch_time}s")

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            # If using multiple GPUs, save the model's state_dict
            if torch.distributed.get_rank() == 0:  # Save only on the main process
                save_obj = {
                "model_state_dict": self.model.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            }
            torch.save(save_obj, best_model_path)
            dist.barrier()
        else:
            # Save the model's state_dict directly
            # torch.save(self.model.state_dict(), best_model_path)
            save_obj = {
                "model_state_dict": self.model.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
            }
            torch.save(save_obj, best_model_path)

        print("Training completed.")
        return self.model

    def test(self, setting, test=0, save_analysis=True):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))

            if "model_state_dict" in load_item:
                model_sd = {k.replace('module.', ''): v for k, v in load_item["model_state_dict"].items()}
                self.model.load_state_dict(model_sd, strict=False)

                if "critic_state_dict" in load_item:
                    critic_sd = {k.replace('module.', ''): v for k, v in load_item["critic_state_dict"].items()}
                    self.critic.load_state_dict(critic_sd, strict=False)
            else:
                # backward compatibility with old checkpoints
                self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        correct_predictions = 0
        total_samples = 0
        ks = (1, 3, 5, 10)
        padding_idx = 40000 if self.args.data == 'yj' else 0
        acc_weighted = {f'acc@{k}': 0.0 for k in ks}
        mrr_weighted = 0.0
        total_valid = 0
        bleu_sum = 0.0
        dtw_sum = 0.0
        traj_count = 0
        step_records = []
        selection_records = []
        candidate_records = []

        self.model.eval()
        self.critic.eval()
        with torch.no_grad():
            # for i, (batch_x,batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark, batch_future_full_7d) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y_f = batch_y_f.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_future_full_7d = batch_future_full_7d.float().to(self.device, non_blocking=True)  # [B,336,7] or empty

                batch_y_seq = batch_y.clone()   # keep [B, T] for BLEU/DTW
                batch_y = batch_y.view(-1)
                if i == 0:
                    print("[DEBUG] batch_x day col unique:",
                        torch.unique(batch_x[:, :, 3].long()).detach().cpu().numpy()[:20])
                    print("[DEBUG] batch_x first sample days:",
                        batch_x[0, :, 3].long().detach().cpu().numpy()[:80])

                    print("[DEBUG] batch_future_full_7d day col unique:",
                        torch.unique(batch_future_full_7d[:, :, 3].long()).detach().cpu().numpy()[:30])
                    print("[DEBUG] future first sample days:",
                        batch_future_full_7d[0, :, 3].long().detach().cpu().numpy()[:120])

                    print("[DEBUG] batch_y_f first sample days:",
                        batch_y_f[0, :, 3].long().detach().cpu().numpy()[:60])

                if self.args.rollout_days > 1:
                    outputs, batch_y_roll, rollout_analysis  = self.rollout_predict_highfreq(
                        test_data,
                        batch_x,
                        batch_x_mark,
                        batch_y_mark,
                        batch_future_full_7d,
                        rollout_days=self.args.rollout_days,
                        plan_mode=getattr(self.args, "plan_mode", "horizon_tokens"),
                        K_eval=getattr(self.args, "plan_horizon", 3),
                        selector_mode=getattr(self.args, "critic_infer_mode", "oracle"),
                        return_analysis=True,
                    )
                    selection_records.extend(rollout_analysis["selection_records"])
                    candidate_records.extend(rollout_analysis["candidate_records"])
                    execution_records = rollout_analysis.get("execution_records", [])
                    execution_lookup = {}
                    for r in execution_records:
                        key = (r["uid"], r["start_day"], r["rollout_day_idx"])
                        execution_lookup[key] = r
                    batch_y = batch_y_roll.view(-1)
                    batch_y_seq = batch_y_roll.long()   # [B, rollout_days*48]
                else:
                    if getattr(self.args, "plan_mode", "horizon_tokens") == "horizon_tokens":
                        with torch.no_grad():
                            plans_all = self.model.plan_head(batch_x_mark)     # batch_x_mark就是[ B,7,H ]
                            plan_tokens = plans_all[:, :getattr(self.args, "plan_horizon", 3), :]
                        outputs, _ = self.model(batch_x, batch_x_mark, batch_y_f, None, plan_tokens)
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)

                # ---------- trajectory-level metrics: BLEU / DTW ----------
                pred_seq_batch = outputs.argmax(dim=-1).detach().cpu().numpy()   # [B, T]
                true_seq_batch = batch_y_seq.detach().cpu().numpy()              # [B, T]
                uid_np = batch_x[:, 0, 0].long().detach().cpu().numpy()
                start_day_np = batch_x[:, 0, 3].long().detach().cpu().numpy()

                for b in range(pred_seq_batch.shape[0]):
                    pred_seq = pred_seq_batch[b].tolist()
                    true_seq = true_seq_batch[b].tolist()

                    # remove padding before computing BLEU / DTW
                    pred_seq_valid = []
                    true_seq_valid = []
                    for p, t in zip(pred_seq, true_seq):
                        if t != padding_idx:
                            pred_seq_valid.append(p)
                            true_seq_valid.append(t)

                    if len(true_seq_valid) == 0:
                        continue

                    bleu_sum += compute_bleu(pred_seq_valid, true_seq_valid)
                    # dtw_sum += compute_dtw(pred_seq_valid, true_seq_valid)
                    dtw_sum += compute_rhythm_dtw(pred_seq_valid, true_seq_valid, padding_value=padding_idx)
                    traj_count += 1

                    for t_idx, (p, t) in enumerate(zip(pred_seq, true_seq)):
                        if t == padding_idx:
                            continue

                        rollout_day_idx = t_idx // 48
                        slot_in_day = t_idx % 48

                        if self.args.rollout_days > 1:
                            full_row = batch_future_full_7d[b, t_idx, :].detach().cpu()
                            tod = int(full_row[1].item())
                            dow = int(full_row[2].item())
                        else:
                            tod = int(batch_y_f[b, slot_in_day, 1].detach().cpu().item())
                            dow = int(batch_y_f[b, slot_in_day, 2].detach().cpu().item())

                        pred_row = int(p) // 200
                        pred_col = int(p) % 200
                        true_row = int(t) // 200
                        true_col = int(t) % 200

                        distance_meter = float(
                            np.sqrt((pred_row - true_row) ** 2 + (pred_col - true_col) ** 2) * 500
                        )

                        exec_key = (int(uid_np[b]), int(start_day_np[b]), int(rollout_day_idx))
                        exec_info = execution_lookup.get(exec_key, {})

                        decision_idx_val = exec_info.get("decision_idx", -1)
                        selection_current_day_val = exec_info.get("selection_current_day", -1)
                        selected_K_val = exec_info.get("selected_K", -1)
                        step_in_selected_plan_val = exec_info.get("step_in_selected_plan", -1)

                        step_records.append({
                            "uid": int(uid_np[b]),
                            "start_day": int(start_day_np[b]),
                            "rollout_day_idx": int(rollout_day_idx),
                            "absolute_day": int(start_day_np[b] + rollout_day_idx),
                            "slot_in_day": int(slot_in_day),
                            "decision_idx": int(decision_idx_val),
                            "selection_current_day": int(selection_current_day_val),
                            "selected_K": int(selected_K_val),
                            "step_in_selected_plan": int(step_in_selected_plan_val),
                            "tod": int(tod),
                            "dow": int(dow),
                            "is_weekend": int(dow in [5, 6]),
                            "true_id": int(t),
                            "pred_id": int(p),
                            "true_row": int(true_row),
                            "true_col": int(true_col),
                            "pred_row": int(pred_row),
                            "pred_col": int(pred_col),
                            "hit1": int(p == t),
                            "distance_meter": distance_meter,
                        })
                # ----------------------------------------------------------
                
                outputs = outputs.view(-1, outputs.size(-1)).float()
                batch_metrics, n_valid = topk_mrr_from_logits(outputs, batch_y, padding_idx=padding_idx, ks=ks)
                total_valid += n_valid
                for k in ks:
                    acc_weighted[f'acc@{k}'] += batch_metrics[f'acc@{k}'] * n_valid
                mrr_weighted += batch_metrics['MRR'] * n_valid

                # Get predicted labels
                _, predicted = torch.max(outputs, dim=1)
                if not self.args.label_missing:
                    correct_predictions += (predicted == batch_y).sum().item()
                    total_samples += batch_y.size(0)
                else:
                    valid_mask = batch_y != (self.num_classes - 1)
                    correct_predictions += ((predicted == batch_y) & valid_mask).sum().item()
                    total_samples += valid_mask.sum().item() # TODO: adjust to different dataset

                preds.append(predicted.detach().cpu())
                trues.append(batch_y.detach().cpu())

                # Logging
                if (i + 1) % 500 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (test_steps - i)
                    print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            # Aggregate predictions and ground truths
            preds = torch.cat(preds, dim=0).numpy()
            trues = torch.cat(trues, dim=0).numpy()

            valid_mask = trues != padding_idx
            preds = preds[valid_mask]
            trues = trues[valid_mask]

            # Calculate accuracy
            accuracy = correct_predictions / total_samples
            # print(f'Test Accuracy: {accuracy:.4f}')
            final_metrics = {k: acc_weighted[k] / max(total_valid, 1) for k in acc_weighted}
            final_metrics['MRR'] = mrr_weighted / max(total_valid, 1)
            final_metrics['BLEU'] = bleu_sum / max(traj_count, 1)
            final_metrics['DTW'] = dtw_sum / max(traj_count, 1)
            print("Test metrics:", final_metrics)
            # wandb.log(final_metrics)
            if save_analysis:
                analysis_dir = os.path.join(folder_path, "analysis")
                os.makedirs(analysis_dir, exist_ok=True)

                step_csv_path = os.path.join(analysis_dir, "step_metrics.csv")
                selection_csv_path = os.path.join(analysis_dir, "selection_metrics.csv")
                selection_records = sorted(
                    selection_records,
                    key=lambda r: (r["uid"], r["start_day"], r["decision_idx"])
                )

                candidate_records = sorted(
                    candidate_records,
                    key=lambda r: (r["uid"], r["start_day"], r["decision_idx"], r["candidate_K"])
                )

                step_records = sorted(
                    step_records,
                    key=lambda r: (r["uid"], r["start_day"], r["rollout_day_idx"], r["slot_in_day"])
                )

                if len(step_records) > 0:
                    with open(step_csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(step_records[0].keys()))
                        writer.writeheader()
                        writer.writerows(step_records)
                    print(f"Saved step-level analysis to: {step_csv_path}")

                if len(selection_records) > 0:
                    with open(selection_csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(selection_records[0].keys()))
                        writer.writeheader()
                        writer.writerows(selection_records)
                    print(f"Saved selection-level analysis to: {selection_csv_path}")
                
                candidate_csv_path = os.path.join(analysis_dir, "candidate_metrics.csv")

                if len(candidate_records) > 0:
                    with open(candidate_csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(candidate_records[0].keys()))
                        writer.writeheader()
                        writer.writerows(candidate_records)
                    print(f"Saved candidate-level analysis to: {candidate_csv_path}")

            # Calculate precision, recall, F1-score
            # from sklearn.metrics import classification_report
            # report = classification_report(trues, preds, target_names=None)
            #print("Classification Report:\n", report)
            # wandb.log({
            #     "test_accuracy": accuracy,
            #     "classification_report": report
            # })

            # Save the classification report to a file
            # with open("result_classification_report.txt", 'a') as f:
            #     f.write(setting + "\n")
            #     f.write(f'Test Accuracy: {accuracy:.4f}\n')
            #     f.write(report)
            #     f.write('\n')

        return accuracy, final_metrics
    
    def _model_ref(self):
        """DDP-safe access to the underlying model."""
        if hasattr(self.args, "use_multi_gpu") and self.args.use_multi_gpu:
            return self.model.module
        return self.model

    def _build_history_stats_per_sample(self, hist_x):
        """
        hist_x: [B, 336, 7]
        return: [B, 4]
        """
        hist_ids = hist_x[:, :, 6].long()
        padding_idx = 40000 if self.args.data == 'yj' else 0

        B, T = hist_ids.shape
        day_len = 48
        num_days = T // day_len

        rows = []

        for b in range(B):
            seq = hist_ids[b]
            valid_seq = seq[seq != padding_idx]

            if valid_seq.numel() == 0:
                rows.append([0.0, 0.0, 0.0, 0.0])
                continue

            hist_unique_ratio = valid_seq.unique().numel() / float(valid_seq.numel())

            if valid_seq.numel() > 1:
                hist_transition_rate = (valid_seq[1:] != valid_seq[:-1]).float().mean().item()
            else:
                hist_transition_rate = 0.0

            stays = []
            run_len = 1
            for i in range(1, valid_seq.numel()):
                if valid_seq[i].item() == valid_seq[i - 1].item():
                    run_len += 1
                else:
                    stays.append(run_len)
                    run_len = 1
            stays.append(run_len)
            hist_repeat_stay_mean_norm = float(np.mean(stays)) / day_len

            daily_uniques = []
            for d in range(num_days):
                day_seq = seq[d * day_len:(d + 1) * day_len]
                day_valid = day_seq[day_seq != padding_idx]
                if day_valid.numel() == 0:
                    daily_uniques.append(0.0)
                else:
                    daily_uniques.append(float(day_valid.unique().numel()) / day_len)

            hist_daily_unique_std_norm = float(np.std(daily_uniques))

            rows.append([
                hist_unique_ratio,
                hist_transition_rate,
                hist_repeat_stay_mean_norm,
                hist_daily_unique_std_norm,
            ])

        return torch.tensor(rows, device=hist_x.device, dtype=hist_x.dtype)
    
    def _build_plan_stats_per_sample(self, plan_tokens, plan_offset=0):
        """
        plan_tokens: [B, K, H]
        return:
            plan_vec: [B, H]
            plan_stats: [B, 4]
                K_norm
                plan_vec_norm
                plan_token_dispersion
                plan_token_step_dispersion
        """
        model_ref = self._model_ref()
        plan_beta = float(getattr(self.args, "plan_weight_beta", 1.0))

        plan_vec = model_ref.build_weighted_plan_vec(
            plan_tokens,
            plan_offset=plan_offset,
            beta=plan_beta
        )  # [B, H]

        B, K, H = plan_tokens.shape

        K_norm = torch.full(
            (B,),
            float(K) / max(1.0, float(getattr(self.args, "plan_horizon", 3))),
            device=plan_tokens.device,
            dtype=plan_tokens.dtype
        )

        plan_vec_norm = plan_vec.norm(dim=-1)  # [B]

        plan_token_dispersion = (
            (plan_tokens - plan_vec.unsqueeze(1))
            .norm(dim=-1)
            .mean(dim=1)
        )  # [B]

        if K > 1:
            plan_token_step_dispersion = (
                (plan_tokens[:, 1:, :] - plan_tokens[:, :-1, :])
                .norm(dim=-1)
                .mean(dim=1)
            )  # [B]
        else:
            plan_token_step_dispersion = torch.zeros(
                (B,),
                device=plan_tokens.device,
                dtype=plan_tokens.dtype
            )

        plan_stats = torch.stack([
            K_norm,
            plan_vec_norm,
            plan_token_dispersion,
            plan_token_step_dispersion,
        ], dim=1)  # [B, 4]

        return plan_vec, plan_stats

    def _build_history_stats(self, hist_x):
        """
        hist_x: [B, 336, 7]
        return 4 batch-level history features:
            hist_unique_ratio
            hist_transition_rate
            hist_repeat_stay_mean_norm
            hist_daily_unique_std_norm
        """
        hist_ids = hist_x[:, :, 6].long()   # [B, 336]
        padding_idx = 40000 if self.args.data == 'yj' else 0

        B, T = hist_ids.shape
        day_len = 48
        num_days = T // day_len

        unique_ratios = []
        transition_rates = []
        repeat_stay_means = []
        daily_unique_stds = []

        for b in range(B):
            seq = hist_ids[b]
            valid_seq = seq[seq != padding_idx]

            if valid_seq.numel() == 0:
                unique_ratios.append(0.0)
                transition_rates.append(0.0)
                repeat_stay_means.append(0.0)
                daily_unique_stds.append(0.0)
                continue

            # 1) unique ratio
            unique_ratio = valid_seq.unique().numel() / float(valid_seq.numel())
            unique_ratios.append(unique_ratio)

            # 2) transition rate
            if valid_seq.numel() > 1:
                transition_rate = (valid_seq[1:] != valid_seq[:-1]).float().mean().item()
            else:
                transition_rate = 0.0
            transition_rates.append(transition_rate)

            # 3) mean stay length
            stays = []
            run_len = 1
            for i in range(1, valid_seq.numel()):
                if valid_seq[i].item() == valid_seq[i - 1].item():
                    run_len += 1
                else:
                    stays.append(run_len)
                    run_len = 1
            stays.append(run_len)
            repeat_stay_mean = float(np.mean(stays)) / day_len   # normalize by 48
            repeat_stay_means.append(repeat_stay_mean)

            # 4) daily unique std
            daily_uniques = []
            for d in range(num_days):
                day_seq = seq[d * day_len:(d + 1) * day_len]
                day_valid = day_seq[day_seq != padding_idx]
                if day_valid.numel() == 0:
                    daily_uniques.append(0.0)
                else:
                    daily_uniques.append(float(day_valid.unique().numel()) / day_len)
            daily_unique_std = float(np.std(daily_uniques))
            daily_unique_stds.append(daily_unique_std)

        device = hist_x.device
        dtype = hist_x.dtype

        return (
            torch.tensor(unique_ratios, device=device, dtype=dtype).mean(),
            torch.tensor(transition_rates, device=device, dtype=dtype).mean(),
            torch.tensor(repeat_stay_means, device=device, dtype=dtype).mean(),
            torch.tensor(daily_unique_stds, device=device, dtype=dtype).mean(),
        )

    def _build_candidate_features(
        self,
        logits,        # [B, 48, C] or None
        diff_loss,     # scalar tensor or None
        plan_tokens,   # [B, K, H]
        plan_vec,      # [B, H]
        K,             # int
        offset=0,      # int
        hist_x=None,   # [B, 336, 7] or None
        hist_day_embeds=None,  # [B, 7, H] or None
    ):
        """
        Build critic features according to critic_feature_mode.
        Supported:
        - k_planvecnorm: 2 dims
        - plan_only: 4 dims
        - traj_only: 5 dims
        - hybrid: 8 dims
        """

        mode = getattr(self.args, "critic_feature_mode", "traj_only")

        device = plan_tokens.device
        dtype = plan_tokens.dtype

        # ---------- Common plan features ----------
        K_norm = torch.tensor(
            float(K) / max(1.0, float(getattr(self.args, "plan_horizon", 3))),
            device=device, dtype=dtype
        )

        plan_vec_norm = plan_vec.norm(dim=-1).mean()

        # average distance from each token to the summary vector
        plan_token_dispersion = (plan_tokens - plan_vec.unsqueeze(1)).norm(dim=-1).mean()

        # average distance between adjacent plan tokens
        if plan_tokens.size(1) > 1:
            plan_token_step_dispersion = (plan_tokens[:, 1:, :] - plan_tokens[:, :-1, :]).norm(dim=-1).mean()
        else:
            plan_token_step_dispersion = torch.tensor(0.0, device=device, dtype=dtype)

        if mode == "plan_only":
            feat = torch.stack([
                K_norm,
                plan_vec_norm,
                plan_token_dispersion,
                plan_token_step_dispersion,
            ], dim=0)   # [4]
            return feat

        elif mode == "k_planvecnorm":
            feat = torch.stack([
                K_norm,
                plan_vec_norm,
            ], dim=0)   # [2]
            return feat

        elif mode == "traj_only":
            if hist_x is None:
                raise ValueError("hist_x is required for traj_only critic features.")

            hist_unique_ratio, hist_transition_rate, hist_repeat_stay_mean_norm, hist_daily_unique_std_norm = \
                self._build_history_stats(hist_x)

            feat = torch.stack([
                K_norm,
                hist_unique_ratio,
                hist_transition_rate,
                hist_repeat_stay_mean_norm,
                hist_daily_unique_std_norm,
            ], dim=0)   # [5]
            return feat

        elif mode == "hybrid":
            if hist_x is None:
                raise ValueError("hist_x is required for hybrid critic features.")

            hist_unique_ratio, hist_transition_rate, hist_repeat_stay_mean_norm, hist_daily_unique_std_norm = \
                self._build_history_stats(hist_x)

            feat = torch.stack([
                K_norm,
                plan_vec_norm,
                plan_token_dispersion,
                plan_token_step_dispersion,
                hist_unique_ratio,
                hist_transition_rate,
                hist_repeat_stay_mean_norm,
                hist_daily_unique_std_norm,
            ], dim=0)   # [8]
            return feat

        else:
            raise ValueError(f"Unknown critic_feature_mode: {mode}")

    def _select_horizon_by_segment_loss(
        self,
        hist_x,                 # [B, 336, 7]
        day_x_prompt,           # [B, 7, H]
        batch_future_full_7d,   # [B, 7*48, 7]
        current_day,            # int, segment start day in rollout
        remaining_days,         # int
        day_len=48,
        return_best_segment_out=False,
        return_candidate_pack=False,
        return_analysis_pack=False,
        max_plan_horizon=3,
    ):
        """
        Segment-level oracle selector:
        enumerate K in {1, ..., min(max_plan_horizon, remaining_days)},
        run the segment-level actor on a fixed max_plan_horizon-day canvas,
        compute CE only over the first K*day_len valid slots, and choose the
        K with the lowest segment-level CE.

        Returns:
            best_K: int
            best_plan_tokens: [B, plan_horizon, H], with tokens after best_K zeroed
            best_segment_out: [B, canvas_len, C] if return_best_segment_out is True
        """
        model_ref = self._model_ref()
        device = hist_x.device

        max_K = min(int(max_plan_horizon), int(remaining_days))
        if max_K < 1:
            raise ValueError(f"remaining_days must be >= 1, got {remaining_days}")

        plan_dtype = model_ref.plan_head.plan_queries.dtype
        plans_all = model_ref.plan_head(day_x_prompt.to(dtype=plan_dtype))  # [B, plan_horizon, H]

        canvas_days = int(max_plan_horizon)
        canvas_len = canvas_days * int(day_len)
        seg_full = self._build_canvas_segment(
            batch_future_full_7d,
            current_day=current_day,
            day_len=day_len,
            canvas_days=canvas_days,
        )  # [B, canvas_len, 7]
        seg_y_f = seg_full[:, :, 0:4]

        candidate_loss_values = []
        candidate_plan_tokens = []
        candidate_segment_outs = [] if return_best_segment_out else None
        candidate_feats = [] if return_candidate_pack else None
        candidate_plan_stats = [] if return_analysis_pack else None
        candidate_critic_scores = [] if return_analysis_pack else None

        criterion = self._select_criterion()
        use_label_missing = getattr(self.args, "label_missing", False)

        with torch.no_grad():
            for K in range(1, max_K + 1):
                valid_len = int(K) * int(day_len)

                # Use the prefix for critic features (matches learned-critic inference),
                # but execute on a fixed planning canvas with tokens after K zeroed out.
                plan_tokens_prefix = plans_all[:, :K, :].contiguous()      # [B, K, H]
                plan_tokens_exec = plans_all.clone().contiguous()          # [B, plan_horizon, H]
                plan_tokens_exec = self._mask_plan_tokens(plan_tokens_exec, K)

                plan_vec_analysis, plan_stats_k = self._build_plan_stats_per_sample(
                    plan_tokens_prefix,
                    plan_offset=0
                )

                loss_mask = self._make_valid_mask(
                    batch_size=hist_x.size(0),
                    canvas_len=canvas_len,
                    valid_len=valid_len,
                    device=device,
                )

                out, diff_loss = self.model(
                    hist_x,
                    day_x_prompt,
                    seg_y_f,
                    None,
                    plan_tokens_exec,
                    plan_offset=0,
                    loss_mask=loss_mask,
                )  # [B, canvas_len, C]

                logits = out[:, :valid_len, :].reshape(-1, out.size(-1)).float()
                labels = seg_full[:, :valid_len, 6].reshape(-1).long()

                if use_label_missing:
                    valid_mask = labels != (self.num_classes - 1)
                    if bool(valid_mask.any()):
                        loss_k = criterion(logits[valid_mask], labels[valid_mask])
                    else:
                        loss_k = logits.new_tensor(1e9)
                else:
                    loss_k = criterion(logits, labels)

                candidate_loss_values.append(loss_k.detach())
                candidate_plan_tokens.append(plan_tokens_exec)
                if return_best_segment_out:
                    candidate_segment_outs.append(out)
                if return_candidate_pack:
                    feat_k = self._build_candidate_features(
                        logits=None,
                        diff_loss=None,
                        plan_tokens=plan_tokens_prefix,
                        plan_vec=plan_vec_analysis,
                        K=K,
                        offset=0,
                        hist_x=hist_x,
                        hist_day_embeds=None,
                    )
                    candidate_feats.append(feat_k)
                if return_analysis_pack:
                    candidate_plan_stats.append(plan_stats_k.detach())
                    candidate_critic_scores.append(None)

        candidate_loss_tensor = torch.stack(candidate_loss_values)
        best_idx = int(candidate_loss_tensor.argmin().item())
        best_K = best_idx + 1
        best_plan_tokens = candidate_plan_tokens[best_idx]

        analysis_pack = None
        if return_analysis_pack:
            analysis_pack = {
                "candidate_losses": torch.stack(candidate_loss_values).detach(),      # [max_K]
                "candidate_plan_stats": torch.stack(candidate_plan_stats, dim=0),     # [max_K, B, 4]
                "candidate_scores": None,
                "oracle_best_idx": best_idx,
            }

        if return_candidate_pack:
            candidate_feat_tensor = torch.stack(candidate_feats, dim=0)
            oracle_best_idx = best_idx

            if return_best_segment_out:
                best_segment_out = candidate_segment_outs[best_idx]
                if return_analysis_pack:
                    return best_K, best_plan_tokens, best_segment_out, candidate_feat_tensor, oracle_best_idx, analysis_pack
                return best_K, best_plan_tokens, best_segment_out, candidate_feat_tensor, oracle_best_idx

            if return_analysis_pack:
                return best_K, best_plan_tokens, candidate_feat_tensor, oracle_best_idx, analysis_pack
            return best_K, best_plan_tokens, candidate_feat_tensor, oracle_best_idx

        if return_best_segment_out:
            best_segment_out = candidate_segment_outs[best_idx]
            if return_analysis_pack:
                return best_K, best_plan_tokens, best_segment_out, analysis_pack
            return best_K, best_plan_tokens, best_segment_out

        if return_analysis_pack:
            return best_K, best_plan_tokens, analysis_pack

        return best_K, best_plan_tokens
    
    def _select_horizon_by_critic(
        self,
        hist_x,
        day_x_prompt,
        day_y_f,
        remaining_days,
        return_best_day1_out=False,
        max_plan_horizon=3,
        return_analysis_pack=False,
    ):
        model_ref = self._model_ref()

        max_K = min(int(max_plan_horizon), int(remaining_days))
        if max_K < 1:
            raise ValueError(f"remaining_days must be >= 1, got {remaining_days}")

        plan_dtype = model_ref.plan_head.plan_queries.dtype
        plans_all = model_ref.plan_head(day_x_prompt.to(dtype=plan_dtype))   # [B, plan_horizon, H]

        candidate_scores = []
        candidate_plan_tokens = []
        candidate_plan_stats = [] if return_analysis_pack else None

        for K in range(1, max_K + 1):
            plan_tokens = plans_all[:, :K, :].contiguous()

            plan_vec, plan_stats_k = self._build_plan_stats_per_sample(
                plan_tokens,
                plan_offset=0
            )

            feat_k = self._build_candidate_features(
                logits=None,
                diff_loss=None,
                plan_tokens=plan_tokens,
                plan_vec=plan_vec,
                K=K,
                offset=0,
                hist_x=hist_x,
                hist_day_embeds=None,
            )   # [4]

            score_k = self.critic(feat_k.unsqueeze(0)).squeeze(0)   # scalar

            candidate_scores.append(score_k)
            candidate_plan_tokens.append(plan_tokens)
            if return_analysis_pack:
                candidate_plan_stats.append(plan_stats_k.detach())

        candidate_score_tensor = torch.stack(candidate_scores, dim=0)   # [max_K]
        best_idx = int(candidate_score_tensor.argmax().item())
        best_K = best_idx + 1
        best_plan_tokens = candidate_plan_tokens[best_idx]
        analysis_pack = None
        if return_analysis_pack:
            analysis_pack = {
                "candidate_losses": None,
                "candidate_plan_stats": torch.stack(candidate_plan_stats, dim=0),   # [max_K, B, 4]
                "candidate_scores": candidate_score_tensor.detach(),                # [max_K]
                "oracle_best_idx": None,
            }

        if return_best_day1_out:
            raise NotImplementedError(
                "_select_horizon_by_critic does not return best_day1_out. "
                "Forward the selected K in rollout execution instead."
            )
            if return_analysis_pack:
                return best_K, best_plan_tokens, best_day1_out, analysis_pack
            return best_K, best_plan_tokens, best_day1_out

        if return_analysis_pack:
            return best_K, best_plan_tokens, analysis_pack

        return best_K, best_plan_tokens
    
    def _mask_plan_tokens(self, plan_tokens, selected_K):
        """Zero out unselected future plan tokens while keeping a fixed 3-day canvas."""
        if plan_tokens is None:
            return None
        masked = plan_tokens.clone()
        selected_K = int(selected_K)
        if selected_K < masked.size(1):
            masked[:, selected_K:, :] = 0
        return masked

    def _build_canvas_segment(self, batch_future_full_7d, current_day, day_len, canvas_days):
        """
        Slice a fixed-length future canvas and pad with zeros if near the end.
        Returns seg_full [B, canvas_days*day_len, F].
        """
        B, total_len, Fdim = batch_future_full_7d.shape
        canvas_len = int(canvas_days) * int(day_len)
        s = int(current_day) * int(day_len)
        e = min(s + canvas_len, total_len)
        seg_full = batch_future_full_7d[:, s:e, :]
        if seg_full.size(1) < canvas_len:
            pad_len = canvas_len - seg_full.size(1)
            pad = seg_full.new_zeros(B, pad_len, Fdim)
            # keep user id if available so embedding index remains valid
            if Fdim > 0:
                pad[:, :, 0] = batch_future_full_7d[:, 0:1, 0]
            seg_full = torch.cat([seg_full, pad], dim=1)
        return seg_full

    def _make_valid_mask(self, batch_size, canvas_len, valid_len, device):
        mask = torch.zeros(batch_size, canvas_len, device=device, dtype=torch.float32)
        mask[:, :int(valid_len)] = 1.0
        return mask

    def _pred_place_to_xy(self, pred_place):
        """Vectorized place-id to grid coordinate conversion matching id_to_xy(): x=pid//200+1, y=pid%200+1."""
        grid_size = int(round(float(getattr(self.args, "grid_size", 200))))
        max_cell = grid_size * grid_size
        pid = pred_place.long().clamp(0, max_cell - 1)
        x = (pid // grid_size + 1).to(dtype=torch.float32)
        y = (pid % grid_size + 1).to(dtype=torch.float32)
        return x, y

    def _build_pred_full_for_update(self, seg_full, logits, valid_len):
        """
        Build predicted future records for context update.
        Temporal/user fields come from seg_full; place and coordinates come from model predictions.
        """
        pred_full = seg_full[:, :int(valid_len), :].clone()
        pred_place_raw = logits[:, :int(valid_len), :].argmax(dim=-1)  # [B, valid_len]
        grid_size = int(round(float(getattr(self.args, "grid_size", 200))))
        pred_place = pred_place_raw.long().clamp(0, grid_size * grid_size - 1)
        if pred_full.size(-1) > 6:
            pred_full[:, :, 6] = pred_place.to(dtype=pred_full.dtype)
        if pred_full.size(-1) > 5:
            px, py = self._pred_place_to_xy(pred_place)
            pred_full[:, :, 4] = px.to(dtype=pred_full.dtype)
            pred_full[:, :, 5] = py.to(dtype=pred_full.dtype)
        return pred_full

    def rollout_predict_highfreq(
        self,
        data_obj,
        batch_x,               # [B, 336, 7]
        batch_x_mark,          # [B, 7, H]
        batch_y_mark,          # legacy arg, kept for compatibility
        batch_future_full_7d,  # [B, 336, 7]
        rollout_days=3,
        day_len=48,
        plan_mode="horizon_tokens",
        K_eval=None,           # kept for compatibility; not used in adaptive selector mode
        selector_mode="oracle",
        return_analysis=False,
    ):
        """
        Adaptive segment-wise rollout for eval/test.

        New behavior (plan_mode == "horizon_tokens"):
        - At each replanning step, enumerate feasible K in {1, ..., min(plan_horizon, remaining_days)}
        - Compare candidate segment-level losses
        - Choose best K*
        - Execute this K*-day segment
        - Replan after the segment ends

        Returns:
            logits_all: [B, rollout_days*48, C]
            labels_all: [B, rollout_days*48]
        """
        device = batch_x.device

        uid = batch_x[:, 0, 0].long()
        start_day = batch_x[:, 0, 3].long()

        hist_x = batch_x.clone()

        logits_list = []
        label_list = []
        analysis_records = {
            "selection_records": [],
            "candidate_records": [],
            "execution_records": []
        }

        current_day = 0
        max_plan_horizon = int(getattr(self.args, "plan_horizon", 3))
        decision_idx = 0

        while current_day < rollout_days:
            remaining_days = rollout_days - current_day

            # Current day block kept for compatibility; selector uses a segment-level canvas
            s = current_day * day_len
            e = (current_day + 1) * day_len

            day_full = batch_future_full_7d[:, s:e, :]     # [B,48,7]
            day_y_f = day_full[:, :, 0:4]                  # [B,48,4]
            day1_label = day_full[:, :, 6].long()          # [B,48]

            day_x_prompt = data_obj.get_x_prompt(uid, start_day + current_day, device=device)  # [B,7,H]

            # ---------------------------------------------------------
            # New adaptive selector
            # ---------------------------------------------------------
            if plan_mode == "horizon_tokens":
                if selector_mode == "learned":
                    if return_analysis:
                        best_K, best_plan_tokens, analysis_pack = self._select_horizon_by_critic(
                            hist_x=hist_x,
                            day_x_prompt=day_x_prompt,
                            day_y_f=day_y_f,
                            remaining_days=remaining_days,
                            max_plan_horizon=max_plan_horizon,
                            return_analysis_pack=True,
                        )
                    else:
                        best_K, best_plan_tokens = self._select_horizon_by_critic(
                            hist_x=hist_x,
                            day_x_prompt=day_x_prompt,
                            day_y_f=day_y_f,
                            remaining_days=remaining_days,
                            max_plan_horizon=max_plan_horizon,
                        )
                        analysis_pack = None
                else:
                    if return_analysis:
                        best_K, best_plan_tokens, analysis_pack = self._select_horizon_by_segment_loss(
                            hist_x=hist_x,
                            day_x_prompt=day_x_prompt,
                            batch_future_full_7d=batch_future_full_7d,
                            current_day=current_day,
                            remaining_days=remaining_days,
                            day_len=day_len,
                            max_plan_horizon=max_plan_horizon,
                            return_analysis_pack=True,
                        )
                    else:
                        best_K, best_plan_tokens = self._select_horizon_by_segment_loss(
                            hist_x=hist_x,
                            day_x_prompt=day_x_prompt,
                            batch_future_full_7d=batch_future_full_7d,
                            current_day=current_day,
                            remaining_days=remaining_days,
                            day_len=day_len,
                            max_plan_horizon=max_plan_horizon,
                        )
                        analysis_pack = None
            else:
                best_K = 1
                best_plan_tokens = None
                analysis_pack = None
            
            if return_analysis:
                uid_cpu = uid.detach().cpu().numpy()
                start_day_cpu = start_day.detach().cpu().numpy()

                hist_stats = self._build_history_stats_per_sample(hist_x).detach().cpu().numpy()

                selected_plan_stats = None
                if analysis_pack is not None and analysis_pack["candidate_plan_stats"] is not None:
                    selected_plan_stats = analysis_pack["candidate_plan_stats"][best_K - 1].detach().cpu().numpy()
                    # [B, 4] = K_norm, plan_vec_norm, plan_token_dispersion, plan_token_step_dispersion

                for b in range(hist_x.size(0)):
                    row = {
                        "uid": int(uid_cpu[b]),
                        "start_day": int(start_day_cpu[b]),
                        "decision_idx": int(decision_idx),
                        "current_day": int(current_day),
                        "absolute_day": int(start_day_cpu[b] + current_day),
                        "remaining_days": int(remaining_days),
                        "selected_K": int(best_K),
                        "selector_mode": str(selector_mode),
                        "critic_feature_mode": str(getattr(self.args, "critic_feature_mode", "none")),
                        "plan_mode": str(plan_mode),

                        "hist_unique_ratio": float(hist_stats[b, 0]),
                        "hist_transition_rate": float(hist_stats[b, 1]),
                        "hist_repeat_stay_mean_norm": float(hist_stats[b, 2]),
                        "hist_daily_unique_std_norm": float(hist_stats[b, 3]),
                    }

                    if selected_plan_stats is not None:
                        row.update({
                            "selected_K_norm": float(selected_plan_stats[b, 0]),
                            "selected_plan_vec_norm": float(selected_plan_stats[b, 1]),
                            "selected_plan_token_dispersion": float(selected_plan_stats[b, 2]),
                            "selected_plan_token_step_dispersion": float(selected_plan_stats[b, 3]),
                        })

                    analysis_records["selection_records"].append(row)
            
            if return_analysis and analysis_pack is not None and analysis_pack["candidate_plan_stats"] is not None:
                candidate_plan_stats = analysis_pack["candidate_plan_stats"].detach().cpu().numpy()
                # [max_K, B, 4]

                candidate_losses = analysis_pack["candidate_losses"]
                if candidate_losses is not None:
                    candidate_losses = candidate_losses.detach().cpu().numpy()

                candidate_scores = analysis_pack["candidate_scores"]
                if candidate_scores is not None:
                    candidate_scores = candidate_scores.detach().cpu().numpy()

                oracle_best_idx = analysis_pack.get("oracle_best_idx", None)

                for k_idx in range(candidate_plan_stats.shape[0]):
                    candidate_K = k_idx + 1

                    for b in range(hist_x.size(0)):
                        row = {
                            "uid": int(uid_cpu[b]),
                            "start_day": int(start_day_cpu[b]),
                            "current_day": int(current_day),
                            "absolute_day": int(start_day_cpu[b] + current_day),
                            "remaining_days": int(remaining_days),
                            "decision_idx": int(decision_idx),

                            "candidate_K": int(candidate_K),
                            "selected_K": int(best_K),
                            "is_selected": int(candidate_K == best_K),

                            "selector_mode": str(selector_mode),
                            "critic_feature_mode": str(getattr(self.args, "critic_feature_mode", "none")),
                            "plan_mode": str(plan_mode),

                            "K_norm": float(candidate_plan_stats[k_idx, b, 0]),
                            "plan_vec_norm": float(candidate_plan_stats[k_idx, b, 1]),
                            "plan_token_dispersion": float(candidate_plan_stats[k_idx, b, 2]),
                            "plan_token_step_dispersion": float(candidate_plan_stats[k_idx, b, 3]),

                            "hist_unique_ratio": float(hist_stats[b, 0]),
                            "hist_transition_rate": float(hist_stats[b, 1]),
                            "hist_repeat_stay_mean_norm": float(hist_stats[b, 2]),
                            "hist_daily_unique_std_norm": float(hist_stats[b, 3]),
                        }

                        if candidate_losses is not None:
                            row["oracle_segment_loss"] = float(candidate_losses[k_idx])
                            row["oracle_best_K"] = int(oracle_best_idx + 1) if oracle_best_idx is not None else -1
                            row["is_oracle_best"] = int(oracle_best_idx is not None and k_idx == oracle_best_idx)

                        if candidate_scores is not None:
                            row["critic_score"] = float(candidate_scores[k_idx])

                        analysis_records["candidate_records"].append(row)

            # ---------------------------------------------------------
            # Execute the selected segment in one actor call.
            # Fixed 3-day canvas; only the first best_K days are returned/used.
            # ---------------------------------------------------------
            canvas_days = max_plan_horizon
            canvas_len = canvas_days * day_len
            valid_len = int(best_K) * day_len

            if return_analysis:
                uid_cpu = uid.detach().cpu().numpy()
                start_day_cpu = start_day.detach().cpu().numpy()
                for offset in range(best_K):
                    day_idx = current_day + offset
                    for b in range(hist_x.size(0)):
                        analysis_records["execution_records"].append({
                            "uid": int(uid_cpu[b]),
                            "start_day": int(start_day_cpu[b]),
                            "decision_idx": int(decision_idx),
                            "selection_current_day": int(current_day),
                            "selected_K": int(best_K),
                            "step_in_selected_plan": int(offset),
                            "rollout_day_idx": int(day_idx),
                            "absolute_day": int(start_day_cpu[b] + day_idx),
                        })

            seg_full = self._build_canvas_segment(
                batch_future_full_7d,
                current_day=current_day,
                day_len=day_len,
                canvas_days=canvas_days,
            )  # [B, 144, 7] if plan_horizon=3
            seg_y_f = seg_full[:, :, 0:4]
            seg_label = seg_full[:, :valid_len, 6].long()
            loss_mask = self._make_valid_mask(seg_full.size(0), canvas_len, valid_len, device)

            with torch.no_grad():
                if plan_mode == "horizon_tokens":
                    plan_tokens_exec = self._mask_plan_tokens(best_plan_tokens, best_K)
                    out_canvas, _ = self.model(
                        hist_x,
                        day_x_prompt,
                        seg_y_f,
                        None,
                        plan_tokens_exec,
                        plan_offset=0,
                        loss_mask=loss_mask,
                    )  # [B,144,C]
                else:
                    day_y_prompt = data_obj.get_y_prompt(uid, start_day + current_day, device=device)
                    out_canvas, _ = self.model(
                        hist_x,
                        day_x_prompt,
                        seg_y_f,
                        None,
                        day_y_prompt,
                        loss_mask=loss_mask,
                    )

            out_valid = out_canvas[:, :valid_len, :]
            logits_list.append(out_valid)
            label_list.append(seg_label)

            # Prediction-feedback segment update: update context once after the segment.
            pred_full = self._build_pred_full_for_update(seg_full, out_canvas, valid_len)
            hist_x = torch.cat([hist_x[:, valid_len:, :], pred_full], dim=1)

            current_day += best_K
            decision_idx += 1

        logits_all = torch.cat(logits_list, dim=1)  # [B, rollout_days*48, C]
        labels_all = torch.cat(label_list, dim=1)   # [B, rollout_days*48]

        if return_analysis:
            return logits_all, labels_all, analysis_records
        
        return logits_all, labels_all
    
    def rollout_train_teacher_forced(
        self,
        data_obj,
        batch_x,               # [B, 336, 7]
        batch_future_full_7d,  # [B, 336, 7]
        rollout_days=3,
        day_len=48,
        plan_mode='highfreq',
    ):
        """
        Adaptive segment-wise training rollout.

        New behavior (plan_mode == "horizon_tokens"):
        - At each replanning step, enumerate feasible K
        - Compare candidate segment-level losses
        - Choose best K*
        - Execute the selected K*-day segment
        - Accumulate full rollout logits / labels / diff_loss

        Returns:
            logits_all: [B, rollout_days*48, C]
            labels_all: [B, rollout_days*48]
            diff_loss_avg: scalar tensor
        """

        device = batch_x.device
        uid = batch_x[:, 0, 0].long()
        start_day = batch_x[:, 0, 3].long()

        hist_x = batch_x.clone()

        logits_list = []
        label_list = []
        diff_loss_list = []
        critic_loss_list = []

        current_day = 0
        max_plan_horizon = int(getattr(self.args, "plan_horizon", 3))
        # decision_idx = 0

        while current_day < rollout_days:
            remaining_days = rollout_days - current_day

            # Current day block for selecting K by Day1 loss
            s = current_day * day_len
            e = (current_day + 1) * day_len

            day_full = batch_future_full_7d[:, s:e, :]     # [B,48,7]
            day_y_f = day_full[:, :, 0:4]                  # [B,48,4]
            day1_label = day_full[:, :, 6].long()          # [B,48]

            day_x_prompt = data_obj.get_x_prompt(uid, start_day + current_day, device=device)  # [B,7,H]

            # ---------------------------------------------------------
            # New adaptive selector for horizon_tokens
            # ---------------------------------------------------------
            if plan_mode == 'horizon_tokens':
                best_K, best_plan_tokens, candidate_feat_tensor, oracle_best_idx = self._select_horizon_by_segment_loss(
                    hist_x=hist_x,
                    day_x_prompt=day_x_prompt,
                    batch_future_full_7d=batch_future_full_7d,
                    current_day=current_day,
                    remaining_days=remaining_days,
                    day_len=day_len,
                    return_candidate_pack=True,
                    max_plan_horizon=max_plan_horizon,
                )
                # epsilon-greedy horizon exploration during training only
                epsilon = getattr(self, "current_epsilon", 0.0)

                if np.random.rand() < epsilon:
                    random_K = np.random.randint(1, min(max_plan_horizon, remaining_days) + 1)

                    model_ref = self._model_ref()
                    plan_dtype = model_ref.plan_head.plan_queries.dtype
                    plans_all = model_ref.plan_head(day_x_prompt.to(dtype=plan_dtype))

                    best_K = int(random_K)
                    best_plan_tokens = plans_all.clone().contiguous()
                    best_plan_tokens = self._mask_plan_tokens(best_plan_tokens, best_K)

                if getattr(self.args, "use_learned_critic", False):
                    critic_scores = self.critic(candidate_feat_tensor)   # [max_K]
                    critic_target = torch.tensor([oracle_best_idx], device=self.device, dtype=torch.long)
                    critic_loss = F.cross_entropy(critic_scores.unsqueeze(0), critic_target)
                    critic_loss_list.append(critic_loss)
            else:
                best_K = 1
                best_plan_tokens = None
            # print(f"[debug] selected best_K={best_K}, remaining_days={remaining_days}")
            # ---------------------------------------------------------
            # Execute the selected segment in one actor call.
            # We use a fixed 3-day canvas and mask losses beyond best_K days.
            # ---------------------------------------------------------
            canvas_days = max_plan_horizon
            canvas_len = canvas_days * day_len
            valid_len = int(best_K) * day_len

            seg_full = self._build_canvas_segment(
                batch_future_full_7d,
                current_day=current_day,
                day_len=day_len,
                canvas_days=canvas_days,
            )  # [B, 144, 7] if plan_horizon=3
            seg_y_f = seg_full[:, :, 0:4]
            seg_label = seg_full[:, :valid_len, 6].long()
            loss_mask = self._make_valid_mask(seg_full.size(0), canvas_len, valid_len, device)

            if plan_mode == 'horizon_tokens':
                plan_tokens_exec = self._mask_plan_tokens(best_plan_tokens, best_K)
                out_canvas, diff_loss = self.model(
                    hist_x,
                    day_x_prompt,
                    seg_y_f,
                    None,
                    plan_tokens_exec,
                    plan_offset=0,
                    loss_mask=loss_mask,
                )  # [B, 144, C]
            elif plan_mode in ['highfreq', 'lowfreq']:
                # Legacy modes remain one-segment but use the current-day prompt.
                day_y_prompt = data_obj.get_y_prompt(uid, start_day + current_day, device=device)
                out_canvas, diff_loss = self.model(
                    hist_x,
                    day_x_prompt,
                    seg_y_f,
                    None,
                    day_y_prompt,
                    loss_mask=loss_mask,
                )
            elif plan_mode == 'horizon_fc':
                y_cat = data_obj.get_y_prompt_horizon_concat(
                    uid,
                    start_day + current_day,
                    horizon=min(remaining_days, max_plan_horizon),
                    device=device,
                )
                model_ref = self._model_ref()
                y_prompt0 = model_ref.plan_agg(y_cat)
                out_canvas, diff_loss = self.model(
                    hist_x,
                    day_x_prompt,
                    seg_y_f,
                    None,
                    y_prompt0,
                    loss_mask=loss_mask,
                )
            else:
                raise ValueError(f"Unknown plan_mode: {plan_mode}")

            out_valid = out_canvas[:, :valid_len, :]
            logits_list.append(out_valid)
            label_list.append(seg_label)

            if diff_loss is not None:
                diff_loss_list.append(diff_loss)

            # Prediction-feedback segment update: update context once after the segment.
            pred_full = self._build_pred_full_for_update(seg_full, out_canvas, valid_len)
            hist_x = torch.cat([hist_x[:, valid_len:, :], pred_full], dim=1)

            current_day += best_K

        logits_all = torch.cat(logits_list, dim=1)  # [B, rollout_days*48, C]
        labels_all = torch.cat(label_list, dim=1)   # [B, rollout_days*48]

        if len(diff_loss_list) > 0:
            diff_loss_avg = torch.stack(diff_loss_list).mean()
        else:
            diff_loss_avg = None

        if len(critic_loss_list) > 0:
            critic_loss_avg = torch.stack(critic_loss_list).mean()
        else:
            critic_loss_avg = None

        return logits_all, labels_all, diff_loss_avg, critic_loss_avg