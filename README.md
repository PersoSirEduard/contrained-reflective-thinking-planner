# Constrained Reflective Planning for Physically Grounded LLMs

This repo contains the code for the paper *"Constrained Reflective Planning for Physically Grounded LLMs."*

## Installation
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install "vllm>=0.8.0"
pip install "trl>=0.21.0"
pip install "transformers>=4.46" "peft>=0.13" "accelerate>=1.0" "datasets>=3.0" bitsandbytes
pip install alfworld textworld
pip install comet_ml pyyaml python-dotenv openai
```

## Dataset
- **ALFWorld**: https://github.com/alfworld/alfworld
- **CALVIN**: https://github.com/mees/calvin

## Base Model
Qwen3-8B (`Qwen/Qwen3-8B` on Hugging Face).

## Suggested reading order
1. `Training/SFT/train_sft_qwen3.py` — SFT stage (learns the DAG planning format).
2. `Training/RL/RL.py` — RL stage on ALFWorld.
3. `Evaluation/Alfworld_eval_curriculum.py` and `Evaluation/Calvin_eval_planning.py` — evaluation entry points.
4. `Dataset/Conver_Alfworld_traj_to_SFT.py` — how SFT data is produced from expert trajectories.

## Project Structure

### Dataset
- `Conver_Alfworld_traj_to_SFT.py` — converts expert trajectories into the DAG-formatted SFT dataset.

### Training
- `SFT/train_sft_qwen3.py` — LoRA SFT on Qwen3-8B using the DAG dataset; the final step merges the adapter for downstream RL.
- `RL/RL.py` — GRPO training on ALFWorld starting from the SFT-merged checkpoint.

### Evaluation
- `Alfworld_eval_curriculum.py` — evaluation on the ALFWorld unseen-valid split.
- `Calvin_eval_planning.py` — planning-only evaluation on CALVIN (reuses pieces of the official CALVIN repo).

### Notebook
- `Scene1.ipynb` —  draft code used during development.



## How to run

Before running, update the file-path constants at the top of each script 
```
# 1. Build the SFT dataset
python Dataset/Conver_Alfworld_traj_to_SFT.py

# 2. SFT on Qwen3-8B (outputs a merged checkpoint for RL)
python Training/SFT/train_sft_qwen3.py

# 3. GRPO RL on ALFWorld
python Training/RL/RL.py

# 4. Evaluation
python Evaluation/Alfworld_eval_curriculum.py
python Evaluation/Calvin_eval_planning.py
```


Some scripts set `HF_HOME` / `TMPDIR` to server-specific paths (e.g. `/workspace`); override these for specific machine 

### Due to storage limitation in Github, all data, model checkpoint and other large files can be found [here](https://drive.google.com/drive/folders/195SH-k7o2lI03X-6eYSoj2KuLq9NwS2u?usp=drive_link)

