import argparse
# --- auto_select_gpu.py (inline) ---
import os, subprocess

def pick_free_gpus(num=1, min_free_mb=8000):
    try:
        q = ["nvidia-smi",
             "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"]
        out = subprocess.check_output(q, stderr=subprocess.STDOUT).decode().strip().splitlines()
        pairs = []
        for line in out:
            idx_s, free_s = [x.strip() for x in line.split(",")]
            pairs.append((int(idx_s), int(free_s)))  # (gpu_index, freeMB)
        
        pairs.sort(key=lambda x: x[1], reverse=True)

        
        good = [i for i, mb in pairs if mb >= min_free_mb]
        chosen = (good or [i for i, _ in pairs])[:num]
        return chosen
    except Exception as e:
        
        print(f"[auto-gpu] warn: {e}. fallback to no selection.")
        return []

def set_cuda_visible_devices(gpus):
    if not gpus:
        return False
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    print(f"[auto-gpu] Using GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")
    return True


if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
    chosen = pick_free_gpus(num=1, min_free_mb=12000)  
    set_cuda_visible_devices(chosen)
# --- end auto_select_gpu.py ---

import random
import numpy as np
import torch
import torch.distributed as dist
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_classification_id import Exp_Classification
from exp.exp_test import Exp_Test

if __name__ == '__main__':
    fix_seed = 42
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Model')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='hm_classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, zero_shot_forecasting, in_context_forecasting, hm_classification]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='AutoTimes_Llama',
                        help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--city', type=str, default='C', help='city')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='data.csv', help='data file')
    parser.add_argument('--test_data_path', type=str, default='data.csv', help='test data file used in zero shot forecasting')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--drop_last',  action='store_true', default=False, help='drop last batch in data loader')
    parser.add_argument('--val_set_shuffle', action='store_false', default=True, help='shuffle validation set')
    parser.add_argument('--drop_short', action='store_true', default=False, help='drop too short sequences in dataset')
    parser.add_argument('--label_missing', action='store_true', default=False)


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=48, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=576, help='label length')
    parser.add_argument('--token_len', type=int, default=48, help='token length')
    parser.add_argument('--test_seq_len', type=int, default=672, help='test seq len')
    parser.add_argument('--test_label_len', type=int, default=576, help='test label len')
    parser.add_argument('--test_pred_len', type=int, default=48, help='test pred len')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--feature_decode_dim', type=int, default=5)
    parser.add_argument('--times_embeds_size', type=int, default=64)
    parser.add_argument('--place_embeds_size', type=int, default=128)
    parser.add_argument('--user_embeds_size', type=int, default=64)
    parser.add_argument('--latlon_emb_dim', type=int, default=16)
    parser.add_argument('--num_attn_layers', type=int, default=1)
    
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_ckp_dir', type=str, default='meta-llama/Llama-3.2-1B', help='llm checkpoints dir')
    parser.add_argument('--backbone', type=str, default='meta-llama/Llama-3.2-1B')
    parser.add_argument('--token', type=str, default='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')
    parser.add_argument('--mlp_hidden_dim', type=int, default=512, help='mlp hidden dim')
    parser.add_argument('--pooling_hidden_dim', type=int, default=32, help='mlp hidden dim')
    parser.add_argument('--mlp_hidden_layers', type=int, default=2, help='mlp hidden layers')
    parser.add_argument('--mlp_activation', type=str, default='tanh', help='mlp activation')
    parser.add_argument('--d_model', type=int, default=256, help='mlp hidden dim')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--hidden_size', type=int, default=256, help='hidden size')

    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')

    #diffusion
    parser.add_argument('--lambda_distill', type=float, default=1.0, help='weight for diffusion/teacher loss')
    parser.add_argument('--use_diffusion', type=bool, default=True, help='use diffusion or not')
    parser.add_argument('--rollout_days', type=int, default=3, help='diffusion steps')
    # parser.add_argument('--train_plan_mode', type=str, default='highfreq', choices=['highfreq', 'lowfreq', 'horizon_fc'], help='plan mode during training')
    parser.add_argument('--train_plan_mode', type=str, default='horizon_tokens', help='plan mode during training')
    parser.add_argument('--train_rollout_days', type=int, default=3, help='rollout days during training')
    parser.add_argument('--horizon_sanity_check', type=int, default=1, help='horizon sanity check during training')
    parser.add_argument('--horizon_check_every', type=int, default=200, help='horizon sanity check every N steps during training')
    parser.add_argument('--plan_mode', type=str, default='horizon_tokens')
    parser.add_argument('--plan_horizon', type=int, default=3)
    parser.add_argument('--plan_weight_beta', type=float, default=1.0,
                    help='beta for day-aware weighted plan summary')
    parser.add_argument('--use_learned_critic', action='store_true', default=True,
                    help='train a small learned critic alongside oracle selector')
    parser.add_argument('--critic_hidden_dim', type=int, default=32,
                        help='hidden dim for the small critic MLP')
    parser.add_argument('--critic_dropout', type=float, default=0.1,
                        help='dropout for the small critic MLP')
    parser.add_argument('--critic_loss_weight', type=float, default=1.0,
                        help='weight for critic CE loss')
    parser.add_argument('--critic_infer_mode', type=str, default='learned',
                        choices=['oracle', 'learned'],
                        help='which selector to use at validation/test time')
    parser.add_argument('--save_best_with_critic', action='store_true', default=True,
                        help='if true, validation uses learned critic for model selection')
    parser.add_argument('--critic_feature_mode', type=str, default='traj_only',
                    choices=['k_planvecnorm', 'plan_only', 'traj_only', 'hybrid'],
                    help='which critic feature set to use')


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--enable_early_stopping', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing lr', default=False)
    parser.add_argument('--tmax', type=int, default=10, help='tmax in cosine anealing lr')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--mix_embeds', action='store_true', help='mix embeds', default=False)
    parser.add_argument('--test_dir', type=str, default='./test', help='test dir')
    parser.add_argument('--test_file_name', type=str, default='checkpoint.pth', help='test file')
    parser.add_argument('--grad_clip', action='store_true', help='grad clip', default=False)
    parser.add_argument('--label_smoothing', type=float, default=0, help='label smoothing')
    # warmup
    
    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--visualize', action='store_true', help='visualize', default=False)
    parser.add_argument('--path', type=str, default='')
    
    
    args = parser.parse_args()
    print(args)

    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))
        rank = int(os.environ.get("RANK", "0")) 
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()
        args.local_rank = local_rank
        print(ip, port, hosts, rank, local_rank, gpus)
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts,
                                rank=rank)
        torch.cuda.set_device(local_rank)
    
    if args.task_name == 'hm_classification':
        Exp = Exp_Classification
    elif args.task_name == 'test':
        Exp = Exp_Test
    else:
        Exp = Exp_Long_Term_Forecast
    backbone_str = args.backbone.split('/')[-1]
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
                args.task_name,
                args.city,
                args.model_id,
                args.model,
                args.data,
                args.llm_ckp_dir[-2:], # Llama-3.2-1B -> 1B
                backbone_str, # change to str after '/'
                args.seq_len,
                args.label_len,
                args.token_len,
                args.learning_rate,
                args.batch_size,
                args.weight_decay,
                args.mlp_hidden_dim,
                args.mlp_hidden_layers,
                args.cosine,
                args.mix_embeds,
                args.des, ii)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_sl{}_ll{}_tl{}_lr{}_bt{}_wd{}_hd{}_hl{}_cos{}_mix{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.token_len,
            args.learning_rate,
            args.batch_size,
            args.weight_decay,
            args.mlp_hidden_dim,
            args.mlp_hidden_layers,
            args.cosine,
            args.mix_embeds,
            args.des, ii)
        exp = Exp(args)  # set experiments
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
