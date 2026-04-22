

import os
import sys
import json
import re
import glob
import random
import argparse
import time
from copy import deepcopy
from datetime import datetime
from collections import defaultdict
from difflib import get_close_matches
from pathlib import Path

import numpy as np
#run time config, THIS IS SPECIAliy for our setting, you can change it as you need
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNTIME_CACHE_ROOT = PROJECT_ROOT / ".runtime_cache"


def _pick_data_root():
    for candidate in (Path("/dev/shm"), Path("/tmp")):
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    return RUNTIME_CACHE_ROOT


def _pick_exec_root():
    RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if os.access(RUNTIME_CACHE_ROOT, os.W_OK):
        return RUNTIME_CACHE_ROOT
    tmp = Path("/tmp")
    if tmp.is_dir() and os.access(tmp, os.W_OK):
        return tmp
    return RUNTIME_CACHE_ROOT


def _configure_runtime_cache_dirs():
    data_root = _pick_data_root()
    exec_root = _pick_exec_root()
    cache_layout = {
        "HF_HOME": RUNTIME_CACHE_ROOT / "hf",
        "HUGGINGFACE_HUB_CACHE": RUNTIME_CACHE_ROOT / "hf" / "hub",
        "TRANSFORMERS_CACHE": RUNTIME_CACHE_ROOT / "hf" / "transformers",
        "VLLM_CACHE_ROOT": data_root / "calvin_vllm_cache",
        # Anything that dlopen()s compiled .so files must avoid noexec /dev/shm
        "TMPDIR": exec_root / "calvin_tmp",
        "TRITON_CACHE_DIR": exec_root / "calvin_triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": exec_root / "calvin_torchinductor_cache",
    }
    for env_name, path in cache_layout.items():
        if not os.environ.get(env_name):
            path.mkdir(parents=True, exist_ok=True)
            os.environ[env_name] = str(path)


_configure_runtime_cache_dirs()

#CAVLIN symbolic task, we directly copy from multistep_sequences.py in CALVIN codebase, and modify a bit for our evaluation setting. You can change the task definitions as you need.

task_categories = {
    "rotate_red_block_right": 1, "rotate_red_block_left": 1,
    "rotate_blue_block_right": 1, "rotate_blue_block_left": 1,
    "rotate_pink_block_right": 1, "rotate_pink_block_left": 1,
    "push_red_block_right": 1, "push_red_block_left": 1,
    "push_blue_block_right": 1, "push_blue_block_left": 1,
    "push_pink_block_right": 1, "push_pink_block_left": 1,
    "move_slider_left": 2, "move_slider_right": 2,
    "open_drawer": 3, "close_drawer": 3,
    "lift_red_block_table": 4, "lift_red_block_slider": 5, "lift_red_block_drawer": 6,
    "lift_blue_block_table": 4, "lift_blue_block_slider": 5, "lift_blue_block_drawer": 6,
    "lift_pink_block_table": 4, "lift_pink_block_slider": 5, "lift_pink_block_drawer": 6,
    "place_in_slider": 7, "place_in_drawer": 7,
    "turn_on_lightbulb": 8, "turn_off_lightbulb": 8,
    "turn_on_led": 8, "turn_off_led": 8,
    "push_into_drawer": 9,
    "stack_block": 10, "unstack_block": 11,
}

tasks = {
    "rotate_red_block_right": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "rotate_red_block_left": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "rotate_blue_block_right": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "rotate_blue_block_left": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "rotate_pink_block_right": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "rotate_pink_block_left": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "push_red_block_right": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "push_red_block_left": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "table"}}],
    "push_blue_block_right": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "push_blue_block_left": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "table"}}],
    "push_pink_block_right": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "push_pink_block_left": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "table"}}],
    "move_slider_left": [{"condition": {"slider": "right", "grasped": 0}, "effect": {"slider": "left"}}],
    "move_slider_right": [{"condition": {"slider": "left", "grasped": 0}, "effect": {"slider": "right"}}],
    "open_drawer": [{"condition": {"drawer": "closed", "grasped": 0}, "effect": {"drawer": "open"}}],
    "close_drawer": [{"condition": {"drawer": "open", "grasped": 0}, "effect": {"drawer": "closed"}}],
    "lift_red_block_table": [{"condition": {"red_block": "table", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}}],
    "lift_red_block_slider": [
        {"condition": {"red_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}},
        {"condition": {"red_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}},
    ],
    "lift_red_block_drawer": [{"condition": {"red_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"red_block": "grasped", "grasped": 1}}],
    "lift_blue_block_table": [{"condition": {"blue_block": "table", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}}],
    "lift_blue_block_slider": [
        {"condition": {"blue_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}},
        {"condition": {"blue_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}},
    ],
    "lift_blue_block_drawer": [{"condition": {"blue_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"blue_block": "grasped", "grasped": 1}}],
    "lift_pink_block_table": [{"condition": {"pink_block": "table", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}}],
    "lift_pink_block_slider": [
        {"condition": {"pink_block": "slider_left", "slider": "right", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}},
        {"condition": {"pink_block": "slider_right", "slider": "left", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}},
    ],
    "lift_pink_block_drawer": [{"condition": {"pink_block": "drawer", "drawer": "open", "grasped": 0}, "effect": {"pink_block": "grasped", "grasped": 1}}],
    "place_in_slider": [
        {"condition": {"red_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"red_block": "slider_right", "grasped": 0}},
        {"condition": {"red_block": "grasped", "slider": "left", "grasped": 1}, "effect": {"red_block": "slider_left", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"blue_block": "slider_right", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "slider": "left", "grasped": 1}, "effect": {"blue_block": "slider_left", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "slider": "right", "grasped": 1}, "effect": {"pink_block": "slider_right", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "slider": "left", "grasped": 1}, "effect": {"pink_block": "slider_left", "grasped": 0}},
    ],
    "place_in_drawer": [
        {"condition": {"red_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"red_block": "drawer", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"blue_block": "drawer", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "drawer": "open", "grasped": 1}, "effect": {"pink_block": "drawer", "grasped": 0}},
    ],
    "stack_block": [
        {"condition": {"red_block": "grasped", "blue_block": "table", "grasped": 1}, "effect": {"red_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"red_block": "grasped", "pink_block": "table", "grasped": 1}, "effect": {"red_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "red_block": "table", "grasped": 1}, "effect": {"blue_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"blue_block": "grasped", "pink_block": "table", "grasped": 1}, "effect": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "red_block": "table", "grasped": 1}, "effect": {"pink_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}},
        {"condition": {"pink_block": "grasped", "blue_block": "table", "grasped": 1}, "effect": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}},
    ],
    "unstack_block": [
        {"condition": {"red_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}, "effect": {"red_block": "table", "blue_block": "table"}},
        {"condition": {"red_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}, "effect": {"red_block": "table", "pink_block": "table"}},
        {"condition": {"blue_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}, "effect": {"blue_block": "table", "red_block": "table"}},
        {"condition": {"blue_block": "stacked_top", "pink_block": "stacked_bottom", "grasped": 0}, "effect": {"blue_block": "table", "pink_block": "table"}},
        {"condition": {"pink_block": "stacked_top", "red_block": "stacked_bottom", "grasped": 0}, "effect": {"pink_block": "table", "red_block": "table"}},
        {"condition": {"pink_block": "stacked_top", "blue_block": "stacked_bottom", "grasped": 0}, "effect": {"pink_block": "table", "blue_block": "table"}},
    ],
    "turn_on_lightbulb": [{"condition": {"lightbulb": 0, "grasped": 0}, "effect": {"lightbulb": 1}}],
    "turn_off_lightbulb": [{"condition": {"lightbulb": 1, "grasped": 0}, "effect": {"lightbulb": 0}}],
    "turn_on_led": [{"condition": {"led": 0, "grasped": 0}, "effect": {"led": 1}}],
    "turn_off_led": [{"condition": {"led": 1, "grasped": 0}, "effect": {"led": 0}}],
    "push_into_drawer": [
        {"condition": {"red_block": "table", "blue_block": ["slider_right", "slider_left"], "pink_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"red_block": "drawer", "grasped": 0}},
        {"condition": {"blue_block": "table", "red_block": ["slider_right", "slider_left"], "pink_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"blue_block": "drawer", "grasped": 0}},
        {"condition": {"pink_block": "table", "blue_block": ["slider_right", "slider_left"], "red_block": ["slider_right", "slider_left"], "drawer": "open", "grasped": 0}, "effect": {"pink_block": "drawer", "grasped": 0}},
    ],
}

# Natural language descriptions for each task (validation set phrasing)
task_to_language = {
    "rotate_red_block_right": "take the red block and rotate it to the right",
    "rotate_red_block_left": "take the red block and rotate it to the left",
    "rotate_blue_block_right": "take the blue block and rotate it to the right",
    "rotate_blue_block_left": "take the blue block and rotate it to the left",
    "rotate_pink_block_right": "take the pink block and rotate it to the right",
    "rotate_pink_block_left": "take the pink block and rotate it to the left",
    "push_red_block_right": "go push the red block right",
    "push_red_block_left": "go push the red block left",
    "push_blue_block_right": "go push the blue block right",
    "push_blue_block_left": "go push the blue block left",
    "push_pink_block_right": "go push the pink block right",
    "push_pink_block_left": "go push the pink block left",
    "move_slider_left": "push the sliding door to the left side",
    "move_slider_right": "push the sliding door to the right side",
    "open_drawer": "pull the handle to open the drawer",
    "close_drawer": "push the handle to close the drawer",
    "lift_red_block_table": "grasp and lift the red block",
    "lift_red_block_slider": "lift the red block from the sliding cabinet",
    "lift_red_block_drawer": "take the red block from the drawer",
    "lift_blue_block_table": "grasp and lift the blue block",
    "lift_blue_block_slider": "lift the blue block from the sliding cabinet",
    "lift_blue_block_drawer": "take the blue block from the drawer",
    "lift_pink_block_table": "grasp and lift the pink block",
    "lift_pink_block_slider": "lift the pink block from the sliding cabinet",
    "lift_pink_block_drawer": "take the pink block from the drawer",
    "place_in_slider": "store the grasped block in the sliding cabinet",
    "place_in_drawer": "store the grasped block in the drawer",
    "push_into_drawer": "slide the block that it falls into the drawer",
    "stack_block": "stack the grasped block",
    "unstack_block": "remove the stacked block",
    "turn_on_lightbulb": "use the switch to turn on the light bulb",
    "turn_off_lightbulb": "use the switch to turn off the light bulb",
    "turn_on_led": "press the button to turn on the led light",
    "turn_off_led": "press the button to turn off the led light",
}

ALL_TASK_NAMES = sorted(tasks.keys())


#symbolic helper 

# Check if a task's condition is satisfied in the current state
def check_condition(state, condition):
    for k, v in condition.items():
        if isinstance(v, (str, int)):
            if state[k] != v:
                return False
        elif isinstance(v, list):
            if state[k] not in v:
                return False
    return True

# Apply a task's effect to the current state to get the next state
def update_state(state, effect):
    next_state = deepcopy(state)
    for k, v in effect.items():
        next_state[k] = v
    return next_state

# Validate if a task is executable in the current state and return the next state if it is
def valid_task(curr_state, task):
    for t in task:
        if check_condition(curr_state, t["condition"]):
            return update_state(curr_state, t["effect"])
    return None


def get_valid_tasks(state):
    """Return list of task names that are executable in this state."""
    valid = []
    for task_name, task_variants in tasks.items():
        for t in task_variants:
            if check_condition(state, t["condition"]):
                valid.append(task_name)
                break
    return valid


def state_to_text(state):
    """Convert symbolic state dict to natural language description."""
    lines = []
    for block in ["red_block", "blue_block", "pink_block"]:
        loc = state[block]
        block_name = block.replace("_", " ")
        if loc == "table":
            lines.append(f"The {block_name} is on the table.")
        elif loc == "grasped":
            lines.append(f"The robot is holding the {block_name}.")
        elif loc == "drawer":
            lines.append(f"The {block_name} is inside the drawer.")
        elif loc == "slider_left":
            lines.append(f"The {block_name} is on the left side of the sliding cabinet.")
        elif loc == "slider_right":
            lines.append(f"The {block_name} is on the right side of the sliding cabinet.")
        elif loc == "stacked_top":
            lines.append(f"The {block_name} is stacked on top of another block.")
        elif loc == "stacked_bottom":
            lines.append(f"The {block_name} is at the bottom of a stack.")
        else:
            lines.append(f"The {block_name} is at: {loc}.")

    slider = state["slider"]
    lines.append(f"The sliding door is on the {slider} side.")

    drawer = state["drawer"]
    lines.append(f"The drawer is {drawer}.")

    lines.append(f"The lightbulb is {'on' if state['lightbulb'] else 'off'}.")
    lines.append(f"The LED is {'on' if state['led'] else 'off'}.")

    if state["grasped"]:
        lines.append("The robot gripper is holding an object.")
    else:
        lines.append("The robot gripper is empty.")

    return "\n".join(lines)


# Sequence generation ( from CALVIN's multistep_sequences.py)

def generate_initial_states(num_states=50, seed=42):
    """Generate diverse valid initial states."""
    rng = random.Random(seed)
    block_locations = ["table", "slider_left", "slider_right"]
    states = []

    for _ in range(num_states * 10):
        state = {
            "red_block": rng.choice(block_locations),
            "blue_block": rng.choice(block_locations),
            "pink_block": rng.choice(block_locations),
            "slider": rng.choice(["left", "right"]),
            "drawer": rng.choice(["open", "closed"]),
            "lightbulb": rng.choice([0, 1]),
            "led": rng.choice([0, 1]),
            "grasped": 0,
        }
        # Validate: blocks on slider must be on the correct side
        for block in ["red_block", "blue_block", "pink_block"]:
            if state[block] == "slider_left" and state["slider"] != "right":
                break
            if state[block] == "slider_right" and state["slider"] != "left":
                break
        else:
            # No two blocks in the same non-table location
            locs = [state[b] for b in ["red_block", "blue_block", "pink_block"] if state[b] != "table"]
            if len(locs) == len(set(locs)):
                states.append(state)
                if len(states) >= num_states:
                    break

    return states


def generate_valid_sequence(state, seq_len=5, max_attempts=500, rng=None):
    """Generate a random valid 5-step task sequence from given state."""
    if rng is None:
        rng = random.Random()

    for _ in range(max_attempts):
        curr_state = deepcopy(state)
        sequence = []
        used_categories = set()

        for step in range(seq_len):
            valid = []
            for task_name, task_variants in tasks.items():
                cat = task_categories[task_name]
                if cat in used_categories:
                    continue
                for t in task_variants:
                    if check_condition(curr_state, t["condition"]):
                        next_state = update_state(curr_state, t["effect"])
                        valid.append((task_name, next_state, cat))
                        break

            if not valid:
                break

            task_name, next_state, cat = rng.choice(valid)
            sequence.append(task_name)
            used_categories.add(cat)
            curr_state = next_state

        if len(sequence) == seq_len:
            return sequence

    return None


def generate_sequences(num_sequences=200, seed=42):
    """Generate evaluation sequences: (initial_state, [task1, ..., task5])."""
    rng = random.Random(seed)
    init_states = generate_initial_states(num_states=100, seed=seed)
    sequences = []

    for i in range(num_sequences):
        state = rng.choice(init_states)
        seq = generate_valid_sequence(state, rng=rng)
        if seq:
            sequences.append((deepcopy(state), seq))

    print(f"Generated {len(sequences)} valid 5-step sequences.")
    return sequences


# Model backends (reused from Alfworld_eval

def clean_json_output(text):
    text = text.strip()
    think_match = re.search(r'<think>.*?</think>\s*', text, re.DOTALL)
    if think_match:
        text = text[think_match.end():].strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    if text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-len("```")].strip()
    return text


class ModelBackend:
    def generate(self, system_content, user_content, temperature=0.3):
        raise NotImplementedError

    def generate_batch(self, pairs, temperature=0.3):
        """Default fallback: sequential calls. Override for true batching."""
        return [self.generate(s, u, temperature=temperature) for s, u in pairs]


class OpenAIBackend(ModelBackend):
    def __init__(self, model_name="gpt-4o-mini"):
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        self.client = OpenAI(timeout=60)
        self.model_name = model_name
        self.name = model_name
        print(f"[{self.name}] OpenAI client ready.")

    def generate(self, system_content, user_content, temperature=0.3):
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=temperature,
                    timeout=60,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"  [{self.name}] API error (attempt {attempt+1}/3): {e}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


class LocalTransformersBackend(ModelBackend):
    def __init__(self, model_path, lora_path=None, name=None, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name or os.path.basename(model_path)
        print(f"[{self.name}] Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[{self.name}] Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[{self.name}] Base model: {total_params/1e9:.2f}B params")

        if lora_path:
            from peft import PeftModel
            print(f"[{self.name}] Loading LoRA from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.model.eval()
        self.device = device
        print(f"[{self.name}] Ready.")

    def generate(self, system_content, user_content, temperature=0.3):
        import torch
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=512,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0, top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def cleanup(self):
        import torch
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class VLLMBackend(ModelBackend):


    def __init__(self, model_path, lora_path=None, name=None,
                 tensor_parallel_size=1, gpu_memory_utilization=0.85):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        from transformers import AutoTokenizer

        self.name = name or os.path.basename(model_path)
        self.SamplingParams = SamplingParams
        self.lora_path = lora_path
        self.LoRARequest = LoRARequest if lora_path else None

        print(f"[{self.name}] Loading vLLM engine from {model_path}...")
        kwargs = dict(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        if lora_path:
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = 64
            print(f"[{self.name}] LoRA adapter: {lora_path}")

        self.llm = LLM(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"[{self.name}] vLLM engine ready.")

    def _apply_chat(self, system_content, user_content):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
            )

    def _lora_kwargs(self):
        if self.lora_path:
            return {"lora_request": self.LoRARequest("adapter", 1, self.lora_path)}
        return {}

    def generate(self, system_content, user_content, temperature=0.3):
        prompt_text = self._apply_chat(system_content, user_content)
        params = self.SamplingParams(
            max_tokens=512, temperature=max(temperature, 0.01), top_p=0.95,
        )
        outputs = self.llm.generate([prompt_text], params, **self._lora_kwargs())
        return outputs[0].outputs[0].text

    def generate_batch(self, pairs, temperature=0.3):
        prompts = [self._apply_chat(s, u) for s, u in pairs]
        params = self.SamplingParams(
            max_tokens=512, temperature=max(temperature, 0.01), top_p=0.95,
        )
        outputs = self.llm.generate(prompts, params, **self._lora_kwargs())
        return [o.outputs[0].text for o in outputs]

    def cleanup(self):
        del self.llm
        import torch
        torch.cuda.empty_cache()



def _resolve_lora_path(lora_dir):
    if os.path.isfile(os.path.join(lora_dir, "adapter_config.json")):
        return lora_dir
    ckpts = sorted(
        glob.glob(os.path.join(lora_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else 0,
    )
    for ckpt in reversed(ckpts):
        if os.path.isfile(os.path.join(ckpt, "adapter_config.json")):
            print(f"  [LoRA] Auto-detected adapter at {ckpt}")
            return ckpt
    return lora_dir


def _make_local_backend(model_path, lora_path=None, name=None):
    if lora_path:
        lora_path = _resolve_lora_path(lora_path)
    backend = getattr(_make_local_backend, "force", "auto")
    if backend == "vllm":
        return VLLMBackend(model_path, lora_path=lora_path, name=name)
    elif backend == "transformers":
        return LocalTransformersBackend(model_path, lora_path=lora_path, name=name)
    else:  # auto
        try:
            import vllm  # noqa: F401
            return VLLMBackend(model_path, lora_path=lora_path, name=name)
        except ImportError:
            return LocalTransformersBackend(model_path, lora_path=lora_path, name=name)


MODEL_REGISTRY = {
    "gpt4o_mini": lambda: OpenAIBackend("gpt-4o-mini"),
    "gpt4o": lambda: OpenAIBackend("gpt-4o"),
    "qwen3_base": lambda: _make_local_backend(
        "Qwen/Qwen3-8B",
        name="qwen3-8b-base",
    ),
    "qwen3_sft": lambda: _make_local_backend(
        "./Qwen3-8B-alfworld-sft-merged",
        name="qwen3-8b-sft",
    ),
    "qwen3_sft_dag": lambda: _make_local_backend(
        "./Qwen3-8B-alfworld-sft-dag-merged",
        name="qwen3-8b-sft-dag",
    ),
    "qwen3_grpo": lambda: _make_local_backend(
        "./Qwen3-8B-alfworld-sft-dag-merged",
        lora_path="./Qwen3-8B-alfworld-grpo",
        name="qwen3-8b-grpo",
    ),
}


# Evaluation Prompts


SYSTEM_PROMPT_PLANNING = """You are an embodied AI planning agent in a CALVIN-style symbolic tabletop environment.
Your goal is to choose valid actions by reasoning over symbolic state preconditions and effects.

CRITICAL DAG RULES:
1. ALLOWED STATE TAGS (Nodes): Each node must be a verifiable symbolic state.
    Use only these tags:
    - Block_Location(color, location) where location in {table, slider_left, slider_right, drawer, grasped, stacked_top, stacked_bottom}
    - Slider_Position(side) where side in {left, right}
    - Drawer_State(state) where state in {open, closed}
    - Lightbulb_State(state) where state in {on, off}
    - LED_State(state) where state in {on, off}
    - Gripper_State(state) where state in {empty, holding}

2. ALLOWED ACTION TAGS (Edges): Edges must use one abstract action tag from:
    - ROTATE(color, dir)
    - PUSH_BLOCK(color, dir)
    - MOVE_SLIDER(side)
    - OPEN_DRAWER
    - CLOSE_DRAWER
    - LIFT(color, source)
    - PLACE(destination)
    - PUSH_INTO_DRAWER
    - STACK_BLOCK
    - UNSTACK_BLOCK
    - TOGGLE_LIGHT(target, state)

3. STRICT PRUNING (NO HISTORY LOGGING):
    The DAG must represent your current valid plan, not an action history.
    If a branch becomes invalid due to preconditions, remove that branch.

4. PRECONDITION-STRICT REASONING:
    Never propose an action whose preconditions are not satisfied in the current state.
    Prefer shortest valid plans.

5. OUTPUT DISCIPLINE:
    You must output valid JSON only, no extra text before/after JSON.
"""
#Task1 : Next action prediction
TASK1_NEXT_ACTION = """Given the current state and a goal instruction, predict the SINGLE next action the robot should take.

Current State:
{state_text}

Goal: "{goal_instruction}"

Valid actions in this state: {valid_actions}

Respond in JSON format:
{{
    "Reflection": "brief explanation of why this action is needed",
    "DAG": {{"nodes": {{}}, "edges": []}},
    "Action_Chunks": {{"Subgoal": "next_step", "Command": "exact_action_name"}}
}}"""

# task2 : multi-step sequence generation
TASK2_SEQUENCE = """Given the current state, plan a sequence of actions to achieve ALL of the following goals IN ORDER.
Each goal must be completed before moving to the next.

Current State:
{state_text}

Goals (in order):
{goals_text}

Respond in JSON format:
{{
    "Reflection": "brief explanation of your plan",
    "DAG": {{"nodes": {{}}, "edges": []}},
    "Action_Chunks": [
        {{"Subgoal": "goal_1", "Command": "action_1"}},
        {{"Subgoal": "goal_2", "Command": "action_2"}}
    ]
}}"""

# Evaluation Tasks

#parsed model output helpers (with some robustness to common formatting issues)
def parse_model_response(response):

    clean = clean_json_output(response)
    try:
        return json.loads(clean), None
    except json.JSONDecodeError as e:
        # Try to find JSON in the response
        match = re.search(r'\{[^{}]*\}', clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group()), None
            except json.JSONDecodeError:
                pass
        # Try to find JSON array for plan
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group()), None
            except json.JSONDecodeError:
                pass
        return None, f"JSON parse error: {e}"


# Fuzzy matching for action names to handle minor formatting issues or synonyms in model output.
def fuzzy_match_action(predicted, valid_actions=None):
    predicted = predicted.strip().lower().replace(" ", "_").replace("-", "_")
    # Exact match
    if predicted in ALL_TASK_NAMES:
        return predicted
    # Close match
    matches = get_close_matches(predicted, ALL_TASK_NAMES, n=1, cutoff=0.7)
    if matches:
        return matches[0]
    # If valid_actions provided, try matching against those
    if valid_actions:
        matches = get_close_matches(predicted, valid_actions, n=1, cutoff=0.6)
        if matches:
            return matches[0]
    return predicted


# Extract a single action command from model output, with preference for ALFWorld-style "Action_Chunks" but backward compatibility with older formats.
def extract_action_from_output(llm_output, admissible_actions):
    clean = clean_json_output(llm_output)
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        return None, f"JSON parsing error: {e}"

    if not isinstance(parsed, dict):
        return None, "Output is not a dictionary."

    # Top-level compatibility fields
    for key in ("Command", "action", "Action"):
        if key in parsed:
            cmd = fuzzy_match_action(str(parsed[key]), admissible_actions)
            if cmd in admissible_actions:
                return cmd, None
            return None, f"Command '{cmd}' not admissible."

    # Preferred ALFWorld-style field
    ac = parsed.get("Action_Chunks", parsed.get("Action_Chunk"))
    if isinstance(ac, list) and ac:
        ac = ac[0]
    if isinstance(ac, dict) and "Command" in ac:
        cmd = fuzzy_match_action(str(ac["Command"]), admissible_actions)
        if cmd in admissible_actions:
            return cmd, None
        return None, f"Command '{cmd}' not admissible."

    return None, "No command found in output."

# Extract a list of action commands from model output, with preference for ALFWorld-style "Action_Chunks" but backward compatibility with older formats. Returns list of matched commands and optional error message.
def extract_action_list_from_output(llm_output, admissible_actions=None):
    clean = clean_json_output(llm_output)
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        return [], f"JSON parsing error: {e}"

    if not isinstance(parsed, dict):
        return [], "Output is not a dictionary."

    actions = []

    # Preferred ALFWorld-style list output
    ac = parsed.get("Action_Chunks", parsed.get("Action_Chunk"))
    if isinstance(ac, dict) and "Command" in ac:
        actions = [str(ac["Command"])]
    elif isinstance(ac, list):
        for item in ac:
            if isinstance(item, dict) and "Command" in item:
                actions.append(str(item["Command"]))

    # Backward compatibility with old format
    if not actions and isinstance(parsed.get("plan"), list):
        actions = [str(a) for a in parsed["plan"]]
    if not actions and isinstance(parsed.get("valid_actions"), list):
        actions = [str(a) for a in parsed["valid_actions"]]

    matched = []
    for a in actions:
        matched.append(fuzzy_match_action(a, admissible_actions))
    if admissible_actions is not None:
        matched = [a for a in matched if a in admissible_actions]

    return matched, None


def eval_task1_next_action(model, sequences, num_samples=200, verbose=False):
   # Task 1: Next-Action Prediction
    print("\n" + "=" * 70)
    print("TASK 1: Next-Action Prediction")
    print("=" * 70)

    # Phase 1: build all prompts deterministically.
    items = []
    for seq_idx, (init_state, task_seq) in enumerate(sequences[:num_samples]):
        curr_state = deepcopy(init_state)
        for step_idx, target_task in enumerate(task_seq):
            goal_text = task_to_language[target_task]
            state_text = state_to_text(curr_state)
            valid = get_valid_tasks(curr_state)
            prompt = TASK1_NEXT_ACTION.format(
                state_text=state_text,
                goal_instruction=goal_text,
                valid_actions=json.dumps(valid),
            )
            items.append({
                "seq_idx": seq_idx, "step": step_idx,
                "state": deepcopy(curr_state), "goal": goal_text,
                "target": target_task, "valid": valid, "prompt": prompt,
            })
            new_state = valid_task(curr_state, tasks[target_task])
            if new_state is None:
                break
            curr_state = new_state

    # Phase 2: batched inference.
    print(f"  Batched inference over {len(items)} prompts...")
    t0 = time.time()
    pairs = [(SYSTEM_PROMPT_PLANNING, it["prompt"]) for it in items]
    try:
        responses = model.generate_batch(pairs)
    except Exception as e:
        print(f"  [E] Batched generation failed: {e}")
        responses = [""] * len(items)
    print(f"  Inference took {time.time()-t0:.1f}s")

    # Phase 3: score.
    correct = 0
    total = 0
    results = []
    for it, response in zip(items, responses):
        predicted, _ = extract_action_from_output(response, it["valid"])
        is_correct = predicted == it["target"]
        if is_correct:
            correct += 1
        total += 1
        results.append({
            "seq_idx": it["seq_idx"], "step": it["step"],
            "state": it["state"], "goal": it["goal"],
            "target": it["target"], "predicted": predicted,
            "correct": is_correct, "raw_response": response[:500],
        })
        mark = "✓" if is_correct else "✗"
        if verbose or not is_correct:
            print(f"  [{mark}] Seq {it['seq_idx']} Step {it['step']}: target={it['target']} pred={predicted}")

    acc = 100 * correct / max(total, 1)
    print(f"\n  TASK 1 RESULT: {correct}/{total} = {acc:.1f}% accuracy")
    return {"task": "next_action", "correct": correct, "total": total, "accuracy": acc, "details": results}


def eval_task2_sequence_planning(model, sequences, num_samples=100, verbose=False):
   # Task 2: Multi-Step Sequence Planning
    print("\n" + "=" * 70)
    print("TASK 2: Multi-Step Sequence Planning")
    print("=" * 70)

    items = []
    for seq_idx, (init_state, task_seq) in enumerate(sequences[:num_samples]):
        state_text = state_to_text(init_state)
        goals_text = "\n".join(
            f"  {i+1}. {task_to_language[t]}" for i, t in enumerate(task_seq)
        )
        prompt = TASK2_SEQUENCE.format(state_text=state_text, goals_text=goals_text)
        items.append({"seq_idx": seq_idx, "task_seq": list(task_seq), "prompt": prompt})

    print(f"  Batched inference over {len(items)} prompts...")
    t0 = time.time()
    pairs = [(SYSTEM_PROMPT_PLANNING, it["prompt"]) for it in items]
    try:
        responses = model.generate_batch(pairs)
    except Exception as e:
        print(f"  [E] Batched generation failed: {e}")
        responses = [""] * len(items)
    print(f"  Inference took {time.time()-t0:.1f}s")

    exact_match = 0
    prefix_lengths = []
    step_correct = 0
    step_total = 0
    total = 0
    results = []

    for it, response in zip(items, responses):
        predicted_plan, _ = extract_action_list_from_output(response)
        task_seq = it["task_seq"]

        is_exact = predicted_plan == task_seq
        if is_exact:
            exact_match += 1
        total += 1

        prefix = 0
        for pred, gold in zip(predicted_plan, task_seq):
            if pred == gold:
                prefix += 1
            else:
                break
        step_correct += prefix
        step_total += len(task_seq)
        prefix_lengths.append(prefix)

        results.append({
            "seq_idx": it["seq_idx"],
            "target_seq": task_seq,
            "predicted_seq": predicted_plan,
            "exact_match": is_exact,
            "prefix_len": prefix,
            "raw_response": response[:500],
        })

        mark = "✓" if is_exact else f"prefix={prefix}/5"
        if verbose or not is_exact:
            print(f"  [{mark}] Seq {it['seq_idx']}: target={task_seq}")
            print(f"         pred  ={predicted_plan}")

    exact_acc = 100 * exact_match / max(total, 1)
    avg_prefix = np.mean(prefix_lengths) if prefix_lengths else 0
    step_acc = 100 * step_correct / max(step_total, 1)

    print(f"\n  TASK 2 RESULTS:")
    print(f"    Exact sequence match:  {exact_match}/{total} = {exact_acc:.1f}%")
    print(f"    Avg correct prefix:    {avg_prefix:.2f} / 5")
    print(f"    Step-level accuracy:   {step_correct}/{step_total} = {step_acc:.1f}%")

    return {
        "task": "sequence_planning",
        "exact_match": exact_match, "total": total,
        "exact_accuracy": exact_acc,
        "avg_prefix_length": float(avg_prefix),
        "step_accuracy": step_acc,
        "details": results,
    }


# Main

def main():
    parser = argparse.ArgumentParser(description="CALVIN Planning Evaluation")
    parser.add_argument("--model", type=str, default="qwen3_sft",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Single model to evaluate (backward compatible)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Multiple models to compare in one run")
    parser.add_argument("--num_sequences", type=int, default=200,
                        help="Number of 5-step sequences to generate")
    parser.add_argument("--tasks", nargs="+", default=["next_action", "sequence"],
                        choices=["next_action", "sequence"],
                        help="Which evaluation tasks to run")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Max samples per evaluation task")
    parser.add_argument("--backend", choices=["auto", "vllm", "transformers"],
                        default="auto",
                        help="Force backend for local models (default: auto = try vllm first)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set backend preference for local models (read by _make_local_backend)
    _make_local_backend.force = args.backend

    print(f"=" * 70)
    print(f"CALVIN Planning Evaluation")
    selected_models = args.models if args.models else [args.model]
    print(f"Models: {selected_models}")
    print(f"Backend: {args.backend}")
    print(f"Tasks: {args.tasks}")
    print(f"Samples per task: {args.num_samples}")
    print(f"=" * 70)

    # Generate sequences
    sequences = generate_sequences(num_sequences=args.num_sequences, seed=args.seed)

    # Run evaluations for each selected model
    all_results = {"models": selected_models, "timestamp": datetime.now().isoformat(), "results": {}}

    for model_key in selected_models:
        print(f"\n{'#' * 70}")
        print(f"Loading model: {model_key}")
        print(f"{'#' * 70}")
        model = MODEL_REGISTRY[model_key]()
        model_result = {}

        if "next_action" in args.tasks:
            r = eval_task1_next_action(model, sequences, num_samples=args.num_samples, verbose=args.verbose)
            model_result["next_action"] = {k: v for k, v in r.items() if k != "details"}

        if "sequence" in args.tasks:
            r = eval_task2_sequence_planning(model, sequences, num_samples=args.num_samples, verbose=args.verbose)
            model_result["sequence_planning"] = {k: v for k, v in r.items() if k != "details"}

        all_results["results"][model_key] = model_result

        if hasattr(model, "cleanup"):
            model.cleanup()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for model_key in selected_models:
        print(f"\nModel: {model_key}")
        model_result = all_results["results"].get(model_key, {})
        for task_name, task_result in model_result.items():
            print(f"  {task_name}:")
            for k, v in task_result.items():
                if k not in ("task", "details"):
                    print(f"    {k}: {v}")

    if len(selected_models) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        header = f"{'Model':24s} {'Task1 Acc':>10s} {'Task2 Exact':>12s} {'Task2 Prefix':>13s}"
        print(header)
        print("-" * len(header))
        for model_key in selected_models:
            mr = all_results["results"].get(model_key, {})
            t1 = mr.get("next_action", {}).get("accuracy", 0.0)
            t2e = mr.get("sequence_planning", {}).get("exact_accuracy", 0.0)
            t2p = mr.get("sequence_planning", {}).get("avg_prefix_length", 0.0)
            print(f"{model_key:24s} {t1:10.1f} {t2e:12.1f} {t2p:13.2f}")

    # Save results
    os.makedirs("eval_results", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "multi" if len(selected_models) > 1 else selected_models[0]
    outfile = f"eval_results/calvin_planning_{model_tag}_{ts}.json"
    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
