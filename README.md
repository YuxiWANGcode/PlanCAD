# PlanCAD Running Instructions

## 1. Data preprocessing

Please first run the preprocessing script:

```bash
python data_pre.py
```
This script calls pre_new.py to generate train/val/test .npz files for the YJ dataset.

Currently, the default city in data_pre.py may need to be changed manually. Please set:

city = "D"

before running, if we are using City D.
Regarding the location where you store your data: Please modify the `load_yj_df()` function within `pre_new.py`.
The preprocessing generates files under: ./dataset/yj/

## 2. Training + rollout evaluation

After preprocessing, run:
```bash
python3 runPlanCAD.py \
--task_name hm_classification --is_training 1 \
--model_id PlanCADTest \
--model PlanCAD \
--data yj \
--city D \
--root_path ./dataset/yj/ \
--seq_len 336 \
--label_len 288 \
--pred_len 48 \
--token_len 48 \
--test_seq_len 336 \
--test_label_len 288 \
--test_pred_len 48 \
--batch_size 16 \
--learning_rate 1e-4 \
--train_epochs 15 \
--dropout 0.2 \
--gpu 0 \
--train_rollout_days 3 \
--critic_feature_mode hybrid \
--rollout_days 3
```
Please modify the following fields if needed:

`--rollout_days`

For example:

`--rollout_days 3`

runs 7-to-3 evaluation, while:

`--rollout_days 7`

runs 7-to-7 evaluation.

## 3. Project Structure

The key file organization is as follows:

```text
.
├── checkpoints/
├── data_provider/
├── dataset/
│   └── yj/tmp
├── exp/
├── data_pre.py
└── run_PlanCAD.py
```
