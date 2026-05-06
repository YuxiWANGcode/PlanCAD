from data_provider import pre_new as preprocess_data
import os, subprocess, sys
root_path = "."        
city = "D"              
seq_len = 336           
label_len = 288         
pred_len = 48           
interval = 48           
preprocess_data.root_path = root_path

for flag in ["train", "val", "test"]:
    print(f"\n Generate {flag} data...")
    preprocess_data.preprocess_yj_data(
        root_path=root_path,
        city=city,
        scale=False,
        size=(seq_len, label_len, pred_len),
        interval=interval,
        flag=flag
    )



for f in os.listdir("dataset/yj"):
    if f.startswith(f"yj_{city}") and f.endswith(".npz"):
        print(f)


os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["HF_TOKEN"] = "your_token"  # Hugging Face token


print("\n=== Step 2: Generating full_sequence.npy ===")
preprocess_data.yj_full4prompt(root_path=root_path, city=city)


def file_exists(p): return os.path.exists(p)

sizes = (336,288,48)
model_tag = "Llama-3.2-1B"
city = "D"
root = "./dataset/yj"

x_path = f"{root}/yj_{city}_{sizes[0]}_{sizes[1]}_{sizes[2]}_{model_tag}_x.pt"
y_paths = {
  "train": f"{root}/yj_{city}_{sizes[0]}_{sizes[1]}_{sizes[2]}_train_{model_tag}_y.pt",
  "val":   f"{root}/yj_{city}_{sizes[0]}_{sizes[1]}_{sizes[2]}_val_{model_tag}_y.pt",
  "test":  f"{root}/yj_{city}_{sizes[0]}_{sizes[1]}_{sizes[2]}_test_{model_tag}_y.pt"
}


if not file_exists(x_path):
    cmd = [sys.executable, "preprocess.py", "--dataset","yj","--city",city,
           "--llm_ckp_dir","meta-llama/Llama-3.2-1B","--flag","train"]
    subprocess.run(cmd, check=True)  
else:
    print("[skip] x already exists")

print("\n Finished.")