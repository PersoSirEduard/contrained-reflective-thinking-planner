import os


import comet_ml
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

#config
Model_name = "Qwen/Qwen3-8B"
load_dataset_path = "YOUR FOLDER PATH/alfworld_dag_sft.json"  
output_dir = "./Qwen3-8B-alfworld-sft-dag"

max_length = 1536   # DAG format is longer than simple {Subgoal, Command}
learning_rate = 2e-4
num_train_epochs = 3
train_batch_size = 2
gradient_accumulation_steps = 8  # effective batch = 2×8 = 16
logging_steps = 10
save_steps = 100

#make output dir
os.makedirs(output_dir, exist_ok=True)

#tokenizer
tokenizer = AutoTokenizer.from_pretrained(Model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#load dataset
train_dataset = load_dataset("json", data_files=load_dataset_path, split="train")

#covert messge -> prompt
def convert_message_to_prompt(examples):
   messages = examples["messages"]
   if len(messages) < 3:
        raise ValueError(f"Each data point should contain at least 3 messages, but got {len(messages)}: {messages}")
   prompt_message = messages[:-1]
   assistant_message = messages[-1]
   if assistant_message["role"] != "assistant":
        raise ValueError(f"The last message should be from the assistant, but got {assistant_message['role']}: {messages}")
   assistant_content = assistant_message["content"].strip()

   prompt = tokenizer.apply_chat_template(
        prompt_message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
   return {"prompt": prompt, "completion": assistant_content + tokenizer.eos_token}

train_dataset = train_dataset.map(convert_message_to_prompt, remove_columns=train_dataset.column_names)


# device_map="auto" spreads model across devices (incl. CPU) and adds communication
# overhead. For single-GPU PEFT training, pin everything to cuda:0.
model = AutoModelForCausalLM.from_pretrained(
    Model_name,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
model.config.use_cache = False

#lora config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

#trainer config
training_args = SFTConfig(
    output_dir=output_dir,
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_length=max_length,
    logging_steps=logging_steps,
    save_steps=save_steps,
    max_steps=100,
    save_total_limit=2,
    bf16=True,
    gradient_checkpointing=True,
    packing=True,
    completion_only_loss=True,
    report_to=["comet_ml"],
    run_name="Qwen3-8B-alfworld-sft-dag"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    args=training_args
)

trainer.train()

# Save LoRA adapter
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save merged model (base + LoRA weights fused) for RL training
merged_output_dir = output_dir + "-merged"
print(f"Merging LoRA weights into base model → {merged_output_dir}")

merged_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(Model_name, torch_dtype=torch.bfloat16, device_map="cpu"),
    output_dir,
    torch_dtype=torch.bfloat16,
)
merged_model = merged_model.merge_and_unload()
merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_output_dir)
print(f"Merged model saved to {merged_output_dir}")
