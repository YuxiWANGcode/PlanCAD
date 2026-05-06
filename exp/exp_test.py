# from data_provider.data_factory import data_provider
# from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual
# from utils.metrics import metric
# import torch
# import torch.nn as nn
# from torch import optim
# import os
# import time
# import warnings
# import wandb
# from tqdm import tqdm
# import numpy as np
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
# import numpy as np

# warnings.filterwarnings('ignore')
# torch.autograd.set_detect_anomaly(True)


# def check_for_nan(tensor, name):
#     if torch.isnan(tensor).any():
#         print(f"NaN found in {name}")

# class Exp_Test(Exp_Basic):
#     def __init__(self, args):
#         super(Exp_Test, self).__init__(args)
#         wandb.init(
#             project="hummob",  # Change project name as needed
#             config={                              # Log hyperparameters
#                 "learning_rate": self.args.learning_rate,
#                 "batch_size": self.args.batch_size,
#                 "epochs": self.args.train_epochs,
#                 "model": self.args.model,
#             },
#             mode="disabled"
#         )
#         train_data, train_loader = self._get_data(flag='train')
#         self.args.num_classes = train_data.get_num_class()
#         self.num_classes = train_data.get_num_class()
        
#         print(f"Dataset: {self.args.data}, City: {self.args.city}, Num Users: {self.args.num_users}, Num Classes: {self.args.num_classes}")

        
        
#     def _build_model(self):
#         train_data, train_loader = self._get_data(flag='train')
#         self.args.num_classes = train_data.get_num_class()
#         self.args.num_users = train_data.get_num_users()
        
#         model = self.model_dict[self.args.model].Model(self.args)
#         if self.args.path != '':
#             model.load_state_dict(torch.load(self.args.path),strict=False)
        
#         if self.args.use_multi_gpu:
#             self.device = torch.device('cuda:{}'.format(self.args.local_rank))
#             model = DDP(model.cuda(), device_ids=[self.args.local_rank])
#         else:
#             self.device = torch.device(f'cuda:{self.args.gpu}' if torch.cuda.is_available() else 'cpu')
#             model = model.to(self.device)
#         return model

#     def _get_data(self, flag, **kwargs):
#         data_set, data_loader = data_provider(self.args, flag, **kwargs)
#         return data_set, data_loader

#     def _select_optimizer(self):
#         p_list = []
#         for n, p in self.model.named_parameters():
#             if not p.requires_grad:
#                 continue
#             else:
#                 p_list.append(p)
#                 if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
#                     print(n, p.dtype, p.shape)
#         model_optim = optim.AdamW([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
#         if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
#             print('next learning rate is {}'.format(self.args.learning_rate))
#         return model_optim

#     def _select_criterion(self):
#         criterion = nn.CrossEntropyLoss()
#         if self.args.data == 'yj':
#             criterion = nn.CrossEntropyLoss(ignore_index=40000)
#         elif self.args.data == 'us':
#             criterion = nn.CrossEntropyLoss(ignore_index=0)
#         return criterion

    
#     def vali(self, vali_data, vali_loader, criterion, is_test=False):
#         total_loss = 0
#         total_samples = 0
#         correct_counts = torch.zeros(10, device=self.device)  # Track top-1 through top-10 in one tensor
#         mrr_sum = 0
        
#         self.model.eval()
        
#         # real_records = np.zeros((self.args.num_users, 48*7))
#         # predict_records = np.zeros((self.args.num_users, 48*7))
#         # full like 40000
#         real_records = np.full((self.args.num_users, 48*7), 40000) if self.args.data == 'yj' else np.full((self.args.num_users, 48*7), 0)
#         predict_records = np.full((self.args.num_users, 48*7), 40000) if self.args.data == 'yj' else np.full((self.args.num_users, 48*7), 0)
        
#         with torch.no_grad():
#             for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
#                 # Move data to device all at once
#                 batch_x = batch_x.float().to(self.device, non_blocking=True)
#                 batch_y_f = batch_y_f.float().to(self.device, non_blocking=True)
#                 batch_y = batch_y.long().to(self.device, non_blocking=True)
#                 batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
#                 batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

#                 # Forward pass
                
#                 with torch.cuda.amp.autocast():
#                     outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
                
#                 # Handle test vs validation sequence lengths
#                 if is_test:
#                     outputs = outputs[:, -self.args.token_len:, :]
#                     batch_y = batch_y[:, -self.args.token_len:]
                
#                 outputs_class = outputs.argmax(dim=-1)
#                 uid = batch_y_f[:, 0, 0].long() 
#                 day = batch_y_f[:, 0, 2].long()
                
#                 for j in range(len(uid)):
#                     start_col = day[j].item() * 48
#                     end_col = start_col + outputs_class.shape[1]
                    
#                     # Assign the real and predicted values to the respective arrays
#                     real_records[uid[j], start_col:end_col] = batch_y[j].cpu().numpy()
#                     predict_records[uid[j], start_col:end_col] = outputs_class[j].cpu().numpy()
                
#                 # Reshape for computation
#                 batch_y = batch_y.view(-1)
#                 outputs = outputs.view(-1, outputs.size(-1))
                

#                 # Filter valid samples (handle padding)
#                 padding_idx = 40000 if self.args.data == 'yj' else 0
#                 valid_mask = batch_y != padding_idx
#                 valid_outputs = outputs[valid_mask]
#                 valid_targets = batch_y[valid_mask]
                
#                 # Calculate loss efficiently
#                 loss = criterion(valid_outputs, valid_targets)
#                 total_loss += loss.item() * valid_targets.size(0)
                
#                 # Get top-k predictions efficiently (k=10)
#                 scores, topk_indices = valid_outputs.topk(10, dim=1, sorted=True)
#                 correct = topk_indices.eq(valid_targets.unsqueeze(1))
                
#                 # Update top-k accuracy counts (cumulative sum along k dimension)
#                 correct_at_k = correct.cumsum(dim=1)
#                 correct_counts.add_(correct_at_k.sum(dim=0))
                
#                 # Calculate MRR efficiently
#                 first_correct_pos = correct.float().mul(
#                     torch.arange(1, 11, device=self.device).float()
#                 )
#                 first_correct_pos[first_correct_pos == 0] = float('inf')
#                 mrr_sum += (1.0 / first_correct_pos.min(dim=1).values).sum().item()
                
#                 total_samples += valid_targets.size(0)

#                 if (i + 1) % 500 == 0:  # Reduced frequency of progress updates
#                     print(f"\tValidation batch: {i + 1}/{len(vali_loader)}")

#         # Calculate final metrics efficiently
#         accuracies = {
#             f'acc@{k}': (correct_counts[k-1] / total_samples).item()
#             for k in (1, 3, 5, 10)
#         }
#         print(f"Total samples: {total_samples}")
#         accuracies['MRR'] = mrr_sum / total_samples
#         print(f"Validation accuracy: {accuracies['acc@1']:.4f}, MRR: {accuracies['MRR']:.4f}")
#         avg_loss = total_loss / total_samples
#         accuracy = accuracies['acc@1']
        
#         print(real_records.shape, predict_records.shape)
        
#         valid_mask = real_records != 40000 if self.args.data == 'yj' else real_records != 0
#         # compare accuracy- real vs predicted are the same ones
#         correct_predictions = np.sum(real_records[valid_mask] == predict_records[valid_mask])
#         total_predictions = np.sum(valid_mask)
#         accuracy = correct_predictions / total_predictions
        
#         padding_idx = 40000 if self.args.data == 'yj' else 0
#         # Calculate BLEU score
#         bleu_score = self.calculate_bleu_score(real_records, predict_records)
#         print(f"BLEU score: {bleu_score:.4f}")
        
#         # Calculate DTW distance
#         dtw_distance = self.calculate_dtw_distance(real_records, predict_records, padding_value=padding_idx)
#         print(f"DTW distance: {dtw_distance:.0f}")
        
#         time_accuracy, day_accuracy = self.calculate_temporal_accuracy(real_records, predict_records, padding_value=padding_idx)
#         time_str = ", ".join([f"{x:.4f}" for x in time_accuracy])
#         day_str = ", ".join([f"{x:.4f}" for x in day_accuracy])
#         print("Time-of-day accuracy:", time_str)
#         print("Day accuracy:", day_str)

#         return avg_loss, accuracy, accuracies
    
#     def calculate_temporal_accuracy(self, real_records, predict_records, padding_value=40000):
#         """
#         Calculate accuracy by time of day and by day.
#         Returns two lists:
#         1. Time-of-day accuracy (48 numbers for each timeslot)
#         2. Day accuracy (7 numbers for each day)
        
#         Args:
#             real_records: numpy array of shape [num_users, 336]
#             predict_records: numpy array of shape [num_users, 336]
#             padding_value: value for invalid/padding entries
        
#         Returns:
#             tuple: (time_accuracy, day_accuracy)
#         """
#         # Initialize arrays for counting
#         valid_counts_time = np.zeros(48)
#         correct_counts_time = np.zeros(48)
#         valid_counts_day = np.zeros(7)
#         correct_counts_day = np.zeros(7)
        
#         # Get valid mask
#         valid_mask = real_records != padding_value
        
#         # For each time slot (0-47)
#         for time_slot in range(48):
#             # Get indices for this time slot across all days
#             time_indices = np.arange(time_slot, 336, 48)
            
#             # Create mask for this time slot
#             time_mask = valid_mask[:, time_indices]
            
#             # Count valid entries and correct predictions
#             valid_counts_time[time_slot] = np.sum(time_mask)
#             correct_counts_time[time_slot] = np.sum(
#                 (real_records[:, time_indices] == predict_records[:, time_indices]) & time_mask
#             )
        
#         # For each day (0-6)
#         for day in range(7):
#             # Get indices for this day
#             start_idx = day * 48
#             end_idx = start_idx + 48
            
#             # Create mask for this day
#             day_mask = valid_mask[:, start_idx:end_idx]
            
#             # Count valid entries and correct predictions
#             valid_counts_day[day] = np.sum(day_mask)
#             correct_counts_day[day] = np.sum(
#                 (real_records[:, start_idx:end_idx] == predict_records[:, start_idx:end_idx]) & day_mask
#             )
        
#         # Calculate accuracies
#         time_accuracy = np.where(valid_counts_time > 0,
#                             correct_counts_time / valid_counts_time,
#                             0)
        
#         day_accuracy = np.where(valid_counts_day > 0,
#                             correct_counts_day / valid_counts_day,
#                             0)
        
#         return time_accuracy, day_accuracy
    
#     def calculate_dtw_distance(self, real_records, predict_records, padding_value=40000):
#         """
#         Calculate DTW distance between real and predicted sequences using vectorized operations.
#         Each place ID is converted to row (//200) and col (%200) coordinates.
#         Each grid is 500m x 500m.
        
#         Args:
#             real_records: numpy array with ground truth
#             predict_records: numpy array with predictions
#             padding_value: value used for padding/invalid entries
        
#         Returns:
#             float: Average distance in meters
#         """
#         # Get valid mask and corresponding values
#         valid_mask = real_records != padding_value
#         real_valid = real_records[valid_mask]
#         pred_valid = predict_records[valid_mask]
        
#         # Convert to coordinates
#         real_rows = real_valid // 200
#         real_cols = real_valid % 200
#         pred_rows = pred_valid // 200
#         pred_cols = pred_valid % 200
        
#         # Calculate Euclidean distances in meters
#         distances = np.sqrt((real_rows - pred_rows)**2 + (real_cols - pred_cols)**2) * 500
#         # Return average distance
#         return np.mean(distances)

#     def calculate_bleu_score(self, real_records, predict_records, padding_value=40000, n_gram=3):
#         """
#         Calculate BLEU score treating all valid sequences as one continuous sequence.
        
#         Args:
#             real_records: numpy array of shape [num_users, seq_len] with ground truth
#             predict_records: numpy array of shape [num_users, seq_len] with predictions
#             padding_value: value used for padding/invalid entries
#             n_gram: size of n-grams to use (default 3)
        
#         Returns:
#             float: BLEU score for the entire sequence
#         """
#         def get_ngrams(sequence, n):
#             """Get n-grams from a sequence"""
#             return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
        
#         # Get valid mask
#         valid_mask = real_records != padding_value
        
#         # Get all valid sequences as one continuous sequence
#         real_seq = real_records[valid_mask]
#         pred_seq = predict_records[valid_mask]
        
#         # Check if we have enough data
#         if len(real_seq) < n_gram or len(pred_seq) < n_gram:
#             return 0.0
        
#         # Get n-grams for entire sequences
#         ref_ngrams = get_ngrams(real_seq, n_gram)
#         cand_ngrams = get_ngrams(pred_seq, n_gram)
        
#         if not cand_ngrams:
#             return 0.0
#         from collections import Counter
#         # Count n-grams
#         ref_counts = Counter(ref_ngrams)
#         cand_counts = Counter(cand_ngrams)
        
#         # Calculate matches
#         matches = sum(min(cand_counts[ngram], ref_counts[ngram]) 
#                     for ngram in cand_counts if ngram in ref_counts)
        
#         # Calculate precision
#         total_cand_ngrams = len(cand_ngrams)
#         if total_cand_ngrams == 0:
#             return 0.0
        
#         # Basic BLEU score (just precision, no length penalty)
#         bleu = matches / total_cand_ngrams
        
#         return bleu

#     def train(self, setting):
#         """
#         Train the model to predict the next POI ID.
#         Args:
#             setting (str): Name for saving checkpoints and logs.
#         """
#         test_data, test_loader = self._get_data(flag='test')

#         # print out number of parameters in self.model and how much requires grad
#         total_params = sum(p.numel() for p in self.model.parameters())
#         trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")
#         print(f"Portion of trainable parameters: {trainable_params / total_params:.4f}")
        
#         time_now = time.time()

#         criterion = self._select_criterion()
#         scaler = torch.cuda.amp.GradScaler(init_scale=1024)
#         test_loss, test_accuracy, test_metrics = self.vali(test_data, test_loader, criterion, is_test=True)
        
#         print(f"Time: {time.time() - time_now:.1f}s")
#         return self.model

#     def test(self, setting, test=0):
#         test_data, test_loader = self._get_data(flag='test')

#         print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
#         if test:
#             print('loading model')
#             setting = self.args.test_dir
#             best_model_path = self.args.test_file_name
#             print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
#             load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
#             self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

#         preds = []
#         trues = []
#         folder_path = './test_results/' + setting + '/'
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)

#         time_now = time.time()
#         test_steps = len(test_loader)
#         iter_count = 0
#         correct_predictions = 0
#         total_samples = 0

#         self.model.eval()
#         with torch.no_grad():
#             for i, (batch_x,batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
#                 iter_count += 1
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y_f = batch_y_f.float().to(self.device)
#                 batch_y = batch_y.long().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
#                 batch_y_mark = batch_y_mark.float().to(self.device)
                
#                 batch_y = batch_y.view(-1)

#                 if self.args.use_amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
#                         outputs = outputs.view(-1, outputs.size(-1)).float()
#                 else:
#                     outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
#                     outputs = outputs.view(-1, outputs.size(-1)).float()

#                 # Get predicted labels
#                 _, predicted = torch.max(outputs, dim=1)
#                 if not self.args.label_missing:
#                     correct_predictions += (predicted == batch_y).sum().item()
#                     total_samples += batch_y.size(0)
#                 else:
#                     valid_mask = batch_y != (self.num_classes - 1)
#                     correct_predictions += ((predicted == batch_y) & valid_mask).sum().item()
#                     total_samples += valid_mask.sum().item() # TODO: adjust to different dataset

#                 preds.append(predicted.detach().cpu())
#                 trues.append(batch_y.detach().cpu())

#                 # Logging
#                 if (i + 1) % 500 == 0:
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * (test_steps - i)
#                     print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()

#             # Aggregate predictions and ground truths
#             preds = torch.cat(preds, dim=0).numpy()
#             trues = torch.cat(trues, dim=0).numpy()

#             # Calculate accuracy
#             accuracy = correct_predictions / total_samples
#             print(f'Test Accuracy: {accuracy:.4f}')

#             # Calculate precision, recall, F1-score
#             from sklearn.metrics import classification_report
#             report = classification_report(trues, preds, target_names=None)
#             #print("Classification Report:\n", report)
#             wandb.log({
#                 "test_accuracy": accuracy,
#                 "classification_report": report
#             })

#             # Save the classification report to a file
#             with open("result_classification_report.txt", 'a') as f:
#                 f.write(setting + "\n")
#                 f.write(f'Test Accuracy: {accuracy:.4f}\n')
#                 f.write(report)
#                 f.write('\n')

#         return accuracy

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import wandb
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np

warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)


def check_for_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN found in {name}")

class Exp_Test(Exp_Basic):
    def __init__(self, args):
        super(Exp_Test, self).__init__(args)
        wandb.init(
            project="hummob",  # Change project name as needed
            config={                              # Log hyperparameters
                "learning_rate": self.args.learning_rate,
                "batch_size": self.args.batch_size,
                "epochs": self.args.train_epochs,
                "model": self.args.model,
            },
            mode="disabled"
        )
        train_data, train_loader = self._get_data(flag='train')
        self.args.num_classes = train_data.get_num_class()
        self.num_classes = train_data.get_num_class()
        
        print(f"Dataset: {self.args.data}, City: {self.args.city}, Num Users: {self.args.num_users}, Num Classes: {self.args.num_classes}")

    def get_model_size(self):
        """Calculate model size in MB"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        model_size_mb = (param_size + buffer_size) / 1024 / 1024
        return model_size_mb
        
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
            else:
                p_list.append(p)
                if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
                    print(n, p.dtype, p.shape)
        model_optim = optim.AdamW([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.use_multi_gpu and self.args.local_rank == 0) or not self.args.use_multi_gpu:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        if self.args.data == 'yj':
            criterion = nn.CrossEntropyLoss(ignore_index=40000)
        elif self.args.data == 'us':
            criterion = nn.CrossEntropyLoss(ignore_index=0)
        return criterion

    def measure_inference_performance(self, data_loader, num_samples=100):
        """Measure inference latency and peak memory usage"""
        self.model.eval()
        
        # Warm up GPU
        print("Warming up GPU...")
        with torch.no_grad():
            for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                if i >= 5:  # 5 warmup iterations
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y_f = batch_y_f.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    _ = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
        
        # Reset peak memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        
        # Measure inference time
        latencies = []
        batch_sizes = []
        memory_peaks = []
        
        print(f"Measuring inference performance on {num_samples} samples...")
        with torch.no_grad():
            for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
                if i >= num_samples:
                    break
                
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y_f = batch_y_f.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                
                # Synchronize before timing
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                
                start_time = time.time()
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
                
                # Synchronize after computation
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                
                end_time = time.time()
                
                batch_latency = (end_time - start_time) * 1000  # Convert to ms
                latencies.append(batch_latency)
                batch_sizes.append(batch_x.size(0))
                
                # Record peak memory
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024  # MB
                    memory_peaks.append(peak_memory)
        
        # Calculate statistics
        avg_batch_latency = np.mean(latencies)
        std_batch_latency = np.std(latencies)
        avg_sample_latency = avg_batch_latency / np.mean(batch_sizes)
        
        if torch.cuda.is_available():
            peak_memory_usage = max(memory_peaks)
            avg_memory_usage = np.mean(memory_peaks)
        else:
            peak_memory_usage = 0
            avg_memory_usage = 0
        
        return {
            'avg_batch_latency_ms': avg_batch_latency,
            'std_batch_latency_ms': std_batch_latency,
            'avg_sample_latency_ms': avg_sample_latency,
            'peak_memory_mb': peak_memory_usage,
            'avg_memory_mb': avg_memory_usage
        }
    
    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = 0
        total_samples = 0
        correct_counts = torch.zeros(10, device=self.device)  # Track top-1 through top-10 in one tensor
        mrr_sum = 0
        
        self.model.eval()
        
        # real_records = np.zeros((self.args.num_users, 48*7))
        # predict_records = np.zeros((self.args.num_users, 48*7))
        # full like 40000
        real_records = np.full((self.args.num_users, 48*7), 40000) if self.args.data == 'yj' else np.full((self.args.num_users, 48*7), 0)
        predict_records = np.full((self.args.num_users, 48*7), 40000) if self.args.data == 'yj' else np.full((self.args.num_users, 48*7), 0)
        
        with torch.no_grad():
            for i, (batch_x, batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # Move data to device all at once
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y_f = batch_y_f.float().to(self.device, non_blocking=True)
                batch_y = batch_y.long().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                # Forward pass
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
                
                # Handle test vs validation sequence lengths
                if is_test:
                    outputs = outputs[:, -self.args.token_len:, :]
                    batch_y = batch_y[:, -self.args.token_len:]
                
                outputs_class = outputs.argmax(dim=-1)
                uid = batch_y_f[:, 0, 0].long() 
                day = batch_y_f[:, 0, 2].long()
                
                for j in range(len(uid)):
                    start_col = day[j].item() * 48
                    end_col = start_col + outputs_class.shape[1]
                    
                    # Assign the real and predicted values to the respective arrays
                    real_records[uid[j], start_col:end_col] = batch_y[j].cpu().numpy()
                    predict_records[uid[j], start_col:end_col] = outputs_class[j].cpu().numpy()
                
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
                scores, topk_indices = valid_outputs.topk(10, dim=1, sorted=True)
                correct = topk_indices.eq(valid_targets.unsqueeze(1))
                
                # Update top-k accuracy counts (cumulative sum along k dimension)
                correct_at_k = correct.cumsum(dim=1)
                correct_counts.add_(correct_at_k.sum(dim=0))
                
                # Calculate MRR efficiently
                first_correct_pos = correct.float().mul(
                    torch.arange(1, 11, device=self.device).float()
                )
                first_correct_pos[first_correct_pos == 0] = float('inf')
                mrr_sum += (1.0 / first_correct_pos.min(dim=1).values).sum().item()
                
                total_samples += valid_targets.size(0)

                if (i + 1) % 500 == 0:  # Reduced frequency of progress updates
                    print(f"\tValidation batch: {i + 1}/{len(vali_loader)}")

        # Calculate final metrics efficiently
        accuracies = {
            f'acc@{k}': (correct_counts[k-1] / total_samples).item()
            for k in (1, 3, 5, 10)
        }
        print(f"Total samples: {total_samples}")
        accuracies['MRR'] = mrr_sum / total_samples
        print(f"Validation accuracy: {accuracies['acc@1']:.4f}, MRR: {accuracies['MRR']:.4f}")
        avg_loss = total_loss / total_samples
        accuracy = accuracies['acc@1']
        
        print(real_records.shape, predict_records.shape)
        
        valid_mask = real_records != 40000 if self.args.data == 'yj' else real_records != 0
        # compare accuracy- real vs predicted are the same ones
        correct_predictions = np.sum(real_records[valid_mask] == predict_records[valid_mask])
        total_predictions = np.sum(valid_mask)
        accuracy = correct_predictions / total_predictions
        
        padding_idx = 40000 if self.args.data == 'yj' else 0
        # Calculate BLEU score
        bleu_score = self.calculate_bleu_score(real_records, predict_records)
        print(f"BLEU score: {bleu_score:.4f}")
        
        # Calculate DTW distance
        dtw_distance = self.calculate_dtw_distance(real_records, predict_records, padding_value=padding_idx)
        print(f"DTW distance: {dtw_distance:.0f}")
        
        time_accuracy, day_accuracy = self.calculate_temporal_accuracy(real_records, predict_records, padding_value=padding_idx)
        time_str = ", ".join([f"{x:.4f}" for x in time_accuracy])
        day_str = ", ".join([f"{x:.4f}" for x in day_accuracy])
        print("Time-of-day accuracy:", time_str)
        print("Day accuracy:", day_str)

        return avg_loss, accuracy, accuracies
    
    def calculate_temporal_accuracy(self, real_records, predict_records, padding_value=40000):
        """
        Calculate accuracy by time of day and by day.
        Returns two lists:
        1. Time-of-day accuracy (48 numbers for each timeslot)
        2. Day accuracy (7 numbers for each day)
        
        Args:
            real_records: numpy array of shape [num_users, 336]
            predict_records: numpy array of shape [num_users, 336]
            padding_value: value for invalid/padding entries
        
        Returns:
            tuple: (time_accuracy, day_accuracy)
        """
        # Initialize arrays for counting
        valid_counts_time = np.zeros(48)
        correct_counts_time = np.zeros(48)
        valid_counts_day = np.zeros(7)
        correct_counts_day = np.zeros(7)
        
        # Get valid mask
        valid_mask = real_records != padding_value
        
        # For each time slot (0-47)
        for time_slot in range(48):
            # Get indices for this time slot across all days
            time_indices = np.arange(time_slot, 336, 48)
            
            # Create mask for this time slot
            time_mask = valid_mask[:, time_indices]
            
            # Count valid entries and correct predictions
            valid_counts_time[time_slot] = np.sum(time_mask)
            correct_counts_time[time_slot] = np.sum(
                (real_records[:, time_indices] == predict_records[:, time_indices]) & time_mask
            )
        
        # For each day (0-6)
        for day in range(7):
            # Get indices for this day
            start_idx = day * 48
            end_idx = start_idx + 48
            
            # Create mask for this day
            day_mask = valid_mask[:, start_idx:end_idx]
            
            # Count valid entries and correct predictions
            valid_counts_day[day] = np.sum(day_mask)
            correct_counts_day[day] = np.sum(
                (real_records[:, start_idx:end_idx] == predict_records[:, start_idx:end_idx]) & day_mask
            )
        
        # Calculate accuracies
        time_accuracy = np.where(valid_counts_time > 0,
                            correct_counts_time / valid_counts_time,
                            0)
        
        day_accuracy = np.where(valid_counts_day > 0,
                            correct_counts_day / valid_counts_day,
                            0)
        
        return time_accuracy, day_accuracy
    
    def calculate_dtw_distance(self, real_records, predict_records, padding_value=40000):
        """
        Calculate DTW distance between real and predicted sequences using vectorized operations.
        Each place ID is converted to row (//200) and col (%200) coordinates.
        Each grid is 500m x 500m.
        
        Args:
            real_records: numpy array with ground truth
            predict_records: numpy array with predictions
            padding_value: value used for padding/invalid entries
        
        Returns:
            float: Average distance in meters
        """
        # Get valid mask and corresponding values
        valid_mask = real_records != padding_value
        real_valid = real_records[valid_mask]
        pred_valid = predict_records[valid_mask]
        
        # Convert to coordinates
        real_rows = real_valid // 200
        real_cols = real_valid % 200
        pred_rows = pred_valid // 200
        pred_cols = pred_valid % 200
        
        # Calculate Euclidean distances in meters
        distances = np.sqrt((real_rows - pred_rows)**2 + (real_cols - pred_cols)**2) * 500
        # Return average distance
        return np.mean(distances)

    def calculate_bleu_score(self, real_records, predict_records, padding_value=40000, n_gram=3):
        """
        Calculate BLEU score treating all valid sequences as one continuous sequence.
        
        Args:
            real_records: numpy array of shape [num_users, seq_len] with ground truth
            predict_records: numpy array of shape [num_users, seq_len] with predictions
            padding_value: value used for padding/invalid entries
            n_gram: size of n-grams to use (default 3)
        
        Returns:
            float: BLEU score for the entire sequence
        """
        def get_ngrams(sequence, n):
            """Get n-grams from a sequence"""
            return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]
        
        # Get valid mask
        valid_mask = real_records != padding_value
        
        # Get all valid sequences as one continuous sequence
        real_seq = real_records[valid_mask]
        pred_seq = predict_records[valid_mask]
        
        # Check if we have enough data
        if len(real_seq) < n_gram or len(pred_seq) < n_gram:
            return 0.0
        
        # Get n-grams for entire sequences
        ref_ngrams = get_ngrams(real_seq, n_gram)
        cand_ngrams = get_ngrams(pred_seq, n_gram)
        
        if not cand_ngrams:
            return 0.0
        from collections import Counter
        # Count n-grams
        ref_counts = Counter(ref_ngrams)
        cand_counts = Counter(cand_ngrams)
        
        # Calculate matches
        matches = sum(min(cand_counts[ngram], ref_counts[ngram]) 
                    for ngram in cand_counts if ngram in ref_counts)
        
        # Calculate precision
        total_cand_ngrams = len(cand_ngrams)
        if total_cand_ngrams == 0:
            return 0.0
        
        # Basic BLEU score (just precision, no length penalty)
        bleu = matches / total_cand_ngrams
        
        return bleu

    def train(self, setting):
        """
        Train the model to predict the next POI ID.
        Args:
            setting (str): Name for saving checkpoints and logs.
        """
        test_data, test_loader = self._get_data(flag='test')

        # Print model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_size_mb = self.get_model_size()
        
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print(f"Portion of trainable parameters: {trainable_params / total_params:.4f}")
        print(f"Model size: {model_size_mb:.2f} MB")
        print("="*60 + "\n")
        
        # Measure inference performance
        print("\n" + "="*60)
        print("MEASURING INFERENCE PERFORMANCE")
        print("="*60)
        perf_metrics = self.measure_inference_performance(test_loader, num_samples=100)
        
        print(f"Average batch latency: {perf_metrics['avg_batch_latency_ms']:.2f} ± {perf_metrics['std_batch_latency_ms']:.2f} ms")
        print(f"Average per-sample latency: {perf_metrics['avg_sample_latency_ms']:.2f} ms")
        print(f"Peak GPU memory usage: {perf_metrics['peak_memory_mb']:.2f} MB")
        print(f"Average GPU memory usage: {perf_metrics['avg_memory_mb']:.2f} MB")
        print("="*60 + "\n")
        
        # Log to wandb
        wandb.log({
            "model_size_mb": model_size_mb,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "avg_batch_latency_ms": perf_metrics['avg_batch_latency_ms'],
            "avg_sample_latency_ms": perf_metrics['avg_sample_latency_ms'],
            "peak_memory_mb": perf_metrics['peak_memory_mb'],
        })
        
        time_now = time.time()

        criterion = self._select_criterion()
        scaler = torch.cuda.amp.GradScaler(init_scale=1024)
        test_loss, test_accuracy, test_metrics = self.vali(test_data, test_loader, criterion, is_test=True)
        
        print(f"Time: {time.time() - time_now:.1f}s")
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.test_label_len, self.args.token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            load_item = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in load_item.items()}, strict=False)

        # Print model size after loading
        model_size_mb = self.get_model_size()
        print(f"\nLoaded model size: {model_size_mb:.2f} MB")
        
        # Measure inference performance before testing
        print("\nMeasuring inference performance...")
        perf_metrics = self.measure_inference_performance(test_loader, num_samples=50)
        print(f"Inference latency: {perf_metrics['avg_sample_latency_ms']:.2f} ms/sample")
        print(f"Peak memory usage: {perf_metrics['peak_memory_mb']:.2f} MB")

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

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x,batch_y_f, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y_f = batch_y_f.float().to(self.device)
                batch_y = batch_y.long().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                batch_y = batch_y.view(-1)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
                        outputs = outputs.view(-1, outputs.size(-1)).float()
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_y_f, None, batch_y_mark)
                    outputs = outputs.view(-1, outputs.size(-1)).float()

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

            # Calculate accuracy
            accuracy = correct_predictions / total_samples
            print(f'Test Accuracy: {accuracy:.4f}')

            # Calculate precision, recall, F1-score
            from sklearn.metrics import classification_report
            report = classification_report(trues, preds, target_names=None)
            #print("Classification Report:\n", report)
            wandb.log({
                "test_accuracy": accuracy,
                "classification_report": report,
                "model_size_mb": model_size_mb,
                "inference_latency_ms": perf_metrics['avg_sample_latency_ms'],
                "peak_memory_mb": perf_metrics['peak_memory_mb']
            })

            # Save the classification report to a file
            with open("result_classification_report.txt", 'a') as f:
                f.write(setting + "\n")
                f.write(f'Test Accuracy: {accuracy:.4f}\n')
                f.write(f'Model Size: {model_size_mb:.2f} MB\n')
                f.write(f'Inference Latency: {perf_metrics["avg_sample_latency_ms"]:.2f} ms/sample\n')
                f.write(f'Peak Memory Usage: {perf_metrics["peak_memory_mb"]:.2f} MB\n')
                f.write(report)
                f.write('\n')

        return accuracy