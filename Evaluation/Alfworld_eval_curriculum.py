

import os
import sys
import json
import re
import yaml
import time
import argparse
import glob
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from openai import OpenAI
from dotenv import load_dotenv
from peft import PeftModel
# Paths (auto-detect) we use it to avoid sever disk use too much space for caching models and data. Can also set env vars to override.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNTIME_CACHE_ROOT = PROJECT_ROOT / ".runtime_cache"


def _pick_temp_root():
    """Pick a temp root with writable space for vLLM metadata and temp files."""
    for candidate in (Path("/dev/shm"), Path("/tmp")):
        if candidate.is_dir() and os.access(candidate, os.W_OK):
            return candidate
    return RUNTIME_CACHE_ROOT


def _configure_runtime_cache_dirs():
    """Route model caches and temp files to locations with available space."""
    temp_root = _pick_temp_root()
    cache_layout = {
        "HF_HOME": RUNTIME_CACHE_ROOT / "hf",
        "HUGGINGFACE_HUB_CACHE": RUNTIME_CACHE_ROOT / "hf" / "hub",
        "TRANSFORMERS_CACHE": RUNTIME_CACHE_ROOT / "hf" / "transformers",
        "VLLM_CACHE_ROOT": temp_root / "alfworld_vllm_cache",
        "TMPDIR": temp_root / "alfworld_tmp",
    }
    for env_name, path in cache_layout.items():
        if not os.environ.get(env_name):
            path.mkdir(parents=True, exist_ok=True)
            os.environ[env_name] = str(path)


_configure_runtime_cache_dirs()

# Try common locations for ALFWorld data
_alfworld_data_candidates = [
    os.environ.get("ALFWORLD_DATA", ""),
    str(PROJECT_ROOT / "ALFworld"),
    "/workspace/alfworld_data",
]
ALFWORLD_DATA = next((p for p in _alfworld_data_candidates if p and os.path.isdir(p)), "")

_config_candidates = [
    os.environ.get("ALFWORLD_CONFIG", ""),
    str(PROJECT_ROOT / "ALFWorldOS" / "alfworld" / "configs" / "base_config.yaml"),
    str(PROJECT_ROOT / "base_config.yaml"),
    "/workspace/alfworld/configs/base_config.yaml",
]
ALFWORLD_CONFIG = next((p for p in _config_candidates if p and os.path.isfile(p)), "")

if ALFWORLD_DATA:
    os.environ["ALFWORLD_DATA"] = ALFWORLD_DATA

from alfworld.agents.environment import get_environment


# Utility functions for prompt construction, output parsing, and game collection.
def normalize_text(text):
    if text is None:
        return ""
    return " ".join(text.strip().lower().split())

# Extract global goal from the initial observation text.
def extract_global_goal(observation):
    if "Your task is to:" in observation:
        return observation.split("Your task is to:")[-1].strip()
    return "Complete the task described by the environment."

# Heuristic task type classification based on gamefile name and goal text.
def get_task_type(gamefile="", goal=""):
    name = os.path.basename(os.path.dirname(gamefile)) if gamefile else ""
    prefixes = {
        "pick_and_place": "pick_place",
        "look_at_obj_in_light": "examine",
        "pick_clean_then_place": "clean",
        "pick_heat_then_place": "heat",
        "pick_cool_then_place": "cool",
        "pick_two_obj": "pick_two",
    }
    for prefix, short in prefixes.items():
        if name.startswith(prefix):
            return short
    g = goal.lower()
    if "heat" in g or "warm" in g:
        return "heat"
    if "cool" in g or "chill" in g or "fridge" in g:
        return "cool"
    if "clean" in g or "rinse" in g or "wash" in g:
        return "clean"
    if "examine" in g or "light" in g or "look at" in g:
        return "examine"
    if "two" in g or "both" in g:
        return "pick_two"
    if "put" in g or "place" in g or "move" in g:
        return "pick_place"
    return "unknown"


def clean_json_output(text):
    text = text.strip()
    # Strip <think>...</think> blocks (handles truncated/incomplete ones)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If a stray <think> remains (no closing tag), drop everything up to first {
    if "<think>" in text:
        brace = text.find("{")
        if brace != -1:
            text = text[brace:]
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```\s*$", "", text).strip()
    # Truncate trailing text after the outermost closing brace
    depth = 0
    end_pos = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_pos = i
                break
    if end_pos is not None:
        text = text[: end_pos + 1]
    return text


def _regex_extract_command(text, admissible_commands):
    # Fallback regex extraction for cases where JSON parsing fails. Looks for a "Command" field and validates it.
    m = re.search(r'"Command"\s*:\s*"([^"]+)"', text)
    if not m:
        return None
    cmd = m.group(1).strip()
    if cmd in admissible_commands:
        return cmd
    return fuzzy_match_admissible(cmd, admissible_commands)


def extract_command(llm_output, admissible_commands):
    # Robustly extract the "Command" from the LLM output, handling potential formatting issues and validating against admissible commands.
    clean = clean_json_output(llm_output)
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        # JSON broken (truncated, trailing text, etc.) — try regex salvage
        cmd = _regex_extract_command(llm_output, admissible_commands)
        if cmd:
            return cmd, None
        return None, f"JSON parsing error: {e}"

    if not isinstance(parsed, dict):
        return None, "Output is not a dictionary."

    # Try top-level Command
    if "Command" in parsed:
        cmd = str(parsed["Command"]).strip()
        if cmd in admissible_commands:
            return cmd, None
        matched = fuzzy_match_admissible(cmd, admissible_commands)
        if matched:
            return matched, None
        return None, f"Command '{cmd}' not admissible."

    # Try Action_Chunks
    ac = parsed.get("Action_Chunks", parsed.get("Action_Chunk"))
    if isinstance(ac, list) and ac:
        ac = ac[0]
    if isinstance(ac, dict) and "Command" in ac:
        cmd = str(ac["Command"]).strip()
        if cmd in admissible_commands:
            return cmd, None
        matched = fuzzy_match_admissible(cmd, admissible_commands)
        if matched:
            return matched, None
        return None, f"Command '{cmd}' not admissible."

    return None, "No 'Command' found in output."


def fuzzy_match_admissible(plan_cmd, admissible_commands):
    # Fuzzy match the planned command to admissible commands, allowing for minor variations in phrasing or object references.
    plan_cmd_lower = plan_cmd.strip().lower()
    for ac in admissible_commands:
        if ac.strip().lower() == plan_cmd_lower:
            return ac
    parts = plan_cmd_lower.split()
    if not parts:
        return None
    verb = parts[0]
    if verb == "go" and len(parts) > 1 and parts[1] == "to":
        verb = "go to"
        rest = " ".join(parts[2:])
    else:
        rest = " ".join(parts[1:])
    key_obj = re.split(r"\s+(?:from|to|with|in|on)\s+", rest)[0].strip()
    key_obj_base = re.sub(r"\s*\d+$", "", key_obj).strip()
    best_match = None
    best_score = 0
    for ac in admissible_commands:
        ac_lower = ac.strip().lower()
        if verb == "go to":
            if not ac_lower.startswith("go to"):
                continue
        elif not (ac_lower.startswith(verb + " ") or ac_lower == verb):
            continue
        score = 0
        if key_obj and key_obj in ac_lower:
            score = 3
        elif key_obj_base and key_obj_base in ac_lower:
            score = 2
        elif verb == ac_lower.split()[0]:
            score = 1
        if score > best_score:
            best_score = score
            best_match = ac
    return best_match if best_score >= 1 else None



# Prompts(COPY FROM RL.py)
SYSTEM_PROMPT = (
    "You are an embodied AI agent in the ALFWorld text-based environment. "
    "Your goal is to achieve the given task optimally. "
    "You must reason using a Directed Acyclic Graph (DAG) format to plan and track your progress.\n\n"
    "CRITICAL DAG RULES:\n"
    "1. ALLOWED STATE TAGS (Nodes): Every node MUST use one of these tags to represent a verifiable world state:\n"
    "   - Agent_At(receptacle_id)\n"
    "   - Agent_Holding(object_id)\n"
    "   - Object_Location(object_id, receptacle_id)\n"
    "   - Receptacle_Opened(receptacle_id)\n"
    "   - Object_Property(object_id, property_type) [e.g., Clean, Hot, Cool, Examined, Sliced, On]\n"
    "2. ALLOWED ACTION TAGS (Edges): Edges represent transitions and must use these abstract tags:\n"
    "   - GOTO(receptacle_id), TAKE(object_id, receptacle_id), PUT(object_id, receptacle_id), "
    "OPEN(receptacle_id), CLOSE(receptacle_id), HEAT(object_id, receptacle_id), "
    "COOL(object_id, receptacle_id), CLEAN(object_id, receptacle_id), TOGGLE(object_id), "
    "EXAMINE(object_id, receptacle_id)\n"
    "3. STRICT PRUNING (NO HISTORY LOGGING): The DAG is your CURRENT valid plan, NOT a history log. "
    "If a path fails (e.g., you searched a cabinet and it was empty), you MUST COMPLETELY DELETE "
    "that node and its edges in your next response. Keep the DAG compact.\n"
    "4. NO FALSE DEPENDENCIES: Do not chain independent search actions. Parallel search locations "
    "must point to the EXACT SAME target state.\n"
    "5. DYNAMIC EXECUTION: Find the first 'pending' node whose prerequisite nodes are 'completed'. "
    "This is your Target_Goal. In Action_Chunks, output the EXACT admissible command string to "
    "progress along that edge."
)

IN_CONTEXT_INSTRUCTION = (
    "CRITICAL INSTRUCTION: You MUST output your internal reasoning and next action STRICTLY "
    "in the following JSON format. Do NOT output any natural language conversational text before "
    "or after the JSON block. Your entire response must be parseable by a JSON parser."
)

IN_CONTEXT_EXAMPLE = """{
    "Reflection": "The task is to put a hot apple in the garbage can. I need to find the apple first.",
    "DAG": {
        "nodes": {
            "S0": ["Agent_At(kitchen)", "completed"],
            "S1": ["Object_Location(apple, known)", "pending"],
            "S2": ["Agent_Holding(apple)", "pending"],
            "S3": ["Receptacle_Opened(microwave 1)", "pending"],
            "S4": ["Object_Property(apple, Hot)", "pending"],
            "S5": ["Object_Location(apple, garbagecan 1)", "pending"]
        },
        "edges": [
            {"from": "S0", "to": "S1", "action": "GOTO(countertop 1)"},
            {"from": "S0", "to": "S1", "action": "GOTO(fridge 1)"},
            {"from": "S1", "to": "S2", "action": "TAKE(apple, current_loc)"},
            {"from": "S2", "to": "S3", "action": "OPEN(microwave 1)"},
            {"from": "S3", "to": "S4", "action": "HEAT(apple, microwave 1)"},
            {"from": "S4", "to": "S5", "action": "PUT(apple, garbagecan 1)"}
        ],
        "Target_Goal": "S1",
        "Checker": "I need to achieve S1 first. I will try countertop 1."
    },
    "Action_Chunks": {
        "Subgoal": "S1",
        "Command": "go to countertop 1"
    }
}"""


# Build the user-side prompt content for evaluation, incorporating the current observation, admissible commands, action history, previous DAG, and any previous error messages.
def build_eval_prompt(observation, admissible_cmds, action_history,
                      previous_dag, previous_error, global_goal):
    """Build user-side prompt content for evaluation."""
    prompt = f"Task: {global_goal}\n\n"
    prompt += f"Current observation: {observation}\n\n"
    prompt += "Available actions:\n"
    for cmd in admissible_cmds:
        prompt += f"- {cmd}\n"
    if previous_error:
        prompt += f"\nPrevious error: {previous_error}\n"
    if action_history:
        prompt += "\nActions already taken:\n"
        for i, act in enumerate(action_history[-5:]):
            prompt += f"{i + 1}. {act}\n"
    if previous_dag:
        compact = json.dumps(previous_dag, separators=(",", ":"))
        prompt += f"\nPrevious DAG:\n{compact}\n"
        prompt += (
            "Update the DAG based on the NEW observation above: "
            "mark achieved goals as 'completed', prune failed paths, "
            "then pick the next action.\n"
        )
    else:
        prompt += (
            "\nOutput your reasoning DAG for the remaining plan, "
            "then output the NEXT action in 'Action_Chunks'.\n"
        )
    return prompt


# Model backends for evaluation — OpenAI API, local transformers, and vLLM. Each backend implements a consistent interface for generating responses based on system and user prompts.
class ModelBackend:
    """Base class for model inference."""
    name = "base"

    def generate(self, system_content, user_content, temperature=0.3):
        raise NotImplementedError


# OpenAI API backend — simple wrapper around the OpenAI Python client.
class OpenAIBackend(ModelBackend):
    def __init__(self, model_name="gpt-4o-mini"):

        load_dotenv()
        self.client = OpenAI(timeout=60)
        self.model_name = model_name
        self.name = model_name

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
                print(f"  [{self.name}] API error (attempt {attempt + 1}/3): {e}")
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


class LocalTransformersBackend(ModelBackend):

    
    def __init__(self, model_path, lora_path=None, name=None, device="cuda"):


        self.name = name or os.path.basename(model_path)
        # Load base model and tokenizer
        print(f"[{self.name}] Loading tokenizer from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[{self.name}] Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"[{self.name}] Base model loaded. Params: {total_params / 1e9:.2f}B")

        if lora_path:

            print(f"[{self.name}] Loading LoRA from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[{self.name}] LoRA loaded. Trainable params: {trainable / 1e6:.2f}M")

        self.model.eval()
        self.tokenizer.padding_side = "left"
        self.device = device
        print(f"[{self.name}] Ready.")

    def generate(self, system_content, user_content, temperature=0.3):
        import torch
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                top_p=0.95,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def cleanup(self):
        import torch
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


class VLLMBackend(ModelBackend):
    """vLLM backend — much faster than raw transformers for sequential generation.
    """
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
        # vLLM's LLM class handles both base model and LoRA loading, with efficient tensor parallelism and GPU memory management. We just need to pass the right config.
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

    def generate(self, system_content, user_content, temperature=0.3):
        #the prompt construction is the same as LocalTransformersBackend, but generation is much faster due to vLLM's optimizations. LoRA integration is also seamless via vLLM's request API.
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        params = self.SamplingParams(
            max_tokens=1024,
            temperature=max(temperature, 0.01),
            top_p=0.95,
        )
        kwargs = {}
        if self.lora_path:
            kwargs["lora_request"] = self.LoRARequest(
                "adapter", 1, self.lora_path
            )
        outputs = self.llm.generate([prompt_text], params, **kwargs)
        return outputs[0].outputs[0].text

    def cleanup(self):
        del self.llm
        import torch
        torch.cuda.empty_cache()


#the function to resolve the correct LoRA checkpoint path, handling both direct adapter dirs and parent dirs containing multiple checkpoints. This allows flexible organization of LoRA files without hardcoding specific checkpoint names.
def _resolve_lora_path(lora_dir):
    
    if os.path.isfile(os.path.join(lora_dir, "adapter_config.json")):
        return lora_dir
    # Look for checkpoint-* 
    ckpts = sorted(glob.glob(os.path.join(lora_dir, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else 0)
    for ckpt in reversed(ckpts):
        if os.path.isfile(os.path.join(ckpt, "adapter_config.json")):
            print(f"  [LoRA] Auto-detected adapter at {ckpt}")
            return ckpt
    # Fallback: return as-is, 
    return lora_dir

#in case we can not use VLLM, we fall back to a standard transformers-based backend. This function centralizes the logic for choosing the backend based on availability and environment variables, ensuring consistent model loading and inference interfaces across different setups.
def _make_local_backend(model_path, lora_path=None, name=None):
    """Try vLLM first, fall back to transformers. Respects _FORCE_BACKEND."""
    if lora_path:
        lora_path = _resolve_lora_path(lora_path)
    backend = getattr(_make_local_backend, "force", "auto")
    if backend == "vllm":
        return VLLMBackend(model_path, lora_path=lora_path, name=name)
    elif backend == "transformers":
        return LocalTransformersBackend(model_path, lora_path=lora_path, name=name)
    else:  # auto
        try:
            import vllm  
            return VLLMBackend(model_path, lora_path=lora_path, name=name)
        except ImportError:
            return LocalTransformersBackend(model_path, lora_path=lora_path, name=name)


# the model type we used for evaluation, mapping from a simple name to a backend constructor. This allows us to easily specify which model to evaluate (e.g., via command-line args) without changing the core evaluation logic, and supports both API-based and local models with optional LoRA fine-tuning.
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



#game collection 
def _make_expert_env(config, split):
    # Create a special env instance for collecting expert trajectories, which we will reset for each game. We set train_eval="train" to get the full expert behavior, but we read the initial state using the appropriate split (train/valid_seen/valid_unseen) to get the correct game distribution.
    train_eval_map = {
        "valid_unseen": "eval_out_of_distribution",
        "valid_seen": "eval_in_distribution",
        "train": "train",
    }
    env_obj = get_environment(config["env"]["type"])(
        config, train_eval=train_eval_map[split]
    )
    saved = env_obj.train_eval
    env_obj.train_eval = "train"
    env = env_obj.init_env(batch_size=1)
    env_obj.train_eval = saved
    return env


def _run_expert_on_game(env, max_steps=150):
    """Play the *current* game (just reset) with the handcoded expert.

    Returns (obs, info, expert_cmds):  initial observation/info + action list.
    """
    obs, info = env.reset()
    first_obs = obs[0]
    first_info = info
    expert_cmds = []
    for _ in range(max_steps):
        plan = info.get("extra.expert_plan", [[]])
        actions = plan[0] if isinstance(plan[0], list) else plan
        if not actions:
            break
        action = actions[0]
        obs, scores, dones, info = env.step([action])
        if action != "look":
            expert_cmds.append(action)
        if dones[0]:
            break
    return first_obs, first_info, expert_cmds


def collect_games(config, split, num_games):
    """Collect game metadata + expert trajectories using a single expert env.

    Returns list of dicts:  {game_idx, gamefile, goal, task_type, expert_cmds, expert_length}
    """
    expert_env = _make_expert_env(config, split)

    games = []
    for i in range(num_games):
        current_obs, info, expert_cmds = _run_expert_on_game(expert_env)
        goal = extract_global_goal(current_obs)

        gamefile = ""
        if "extra.gamefile" in info:
            gf = info["extra.gamefile"]
            gamefile = gf[0] if isinstance(gf, list) else gf

        task_type = get_task_type(gamefile, goal)

        games.append({
            "game_idx": i,
            "gamefile": gamefile,
            "goal": goal,
            "initial_obs": current_obs,
            "expert_cmds": expert_cmds,
            "expert_length": len(expert_cmds),
            "task_type": task_type,
        })
        print(
            f"  Game {i + 1:3d}/{num_games}: [{task_type:12s}] "
            f"expert_len={len(expert_cmds):2d} | {goal[:70]}"
        )

    return games


# create a fresh ALFWorld environment for evaluation, ensuring it's properly initialized with batch_size=1 and the correct train_eval setting based on the split. This allows us to reset the environment for each game evaluation while maintaining consistency with the game distribution (train/valid_seen/valid_unseen).
def make_env(config, split):
    """Create a fresh batch_size=1 ALFWorld env for one evaluation pass."""
    train_eval_map = {
        "valid_unseen": "eval_out_of_distribution",
        "valid_seen": "eval_in_distribution",
        "train": "train",
    }
    env = get_environment(config["env"]["type"])(
        config, train_eval=train_eval_map[split]
    )
    env = env.init_env(batch_size=1)
    return env


# single game evaluation function, which handles replaying the expert prefix (if specified), constructing the system and user prompts, and iteratively generating and executing model actions while updating the DAG context. It also implements retry logic for handling invalid model outputs and tracks detailed metrics about the evaluation process.
def evaluate_single_game(
    env,
    model,
    game_info,
    remaining,
    max_model_steps=30,
    max_retries=3,
    verbose=False,
):
    """Evaluate a model on one game, replaying expert prefix up to `remaining` steps from goal.

    Args:
        env:             ALFWorld env (batch_size=1), already positioned (caller did env.reset())
        model:           ModelBackend instance
        game_info:       dict from collect_games()
        remaining:       int (steps from goal) or "full" (no prefix replay)
        max_model_steps: max steps the model can take after prefix
        max_retries:     retries per step for invalid LLM output
        verbose:         print per-step details

    Returns:
        dict with {won, model_steps, prefix_steps, model_actions, ...}
    """
    # env.reset() was already called by the caller — read current state
    obs, info = env.reset()
    current_obs = obs[0]
    current_cmds = info["admissible_commands"][0]
    goal = extract_global_goal(current_obs)

    expert_cmds = game_info["expert_cmds"]
    n_expert = len(expert_cmds)

    #Determine prefix length
    if remaining == "full" or remaining < 0:
        k = 0  # model plays from the very beginning
        actual_remaining = n_expert if n_expert > 0 else max_model_steps
    else:
        k = max(0, n_expert - remaining)
        actual_remaining = min(remaining, n_expert) if n_expert > 0 else remaining

    #  Replay expert prefix
    prefix_actions = []
    for step_idx in range(k):
        if step_idx >= len(expert_cmds):
            break
        action = expert_cmds[step_idx]
        obs_list, scores, dones, infos = env.step([action])
        prefix_actions.append(action)
        current_obs = obs_list[0]
        current_cmds = infos["admissible_commands"][0]

        if dones[0]:
            won = infos.get("won", [False])[0]
            return {
                "won": won,
                "model_steps": 0,
                "prefix_steps": len(prefix_actions),
                "total_steps": len(prefix_actions),
                "model_actions": [],
                "prefix_actions": prefix_actions,
                "remaining_target": remaining,
                "actual_remaining": actual_remaining,
                "goal": goal,
                "task_type": game_info["task_type"],
                "expert_length": n_expert,
            }

    if verbose:
        print(f"    Replayed {len(prefix_actions)}/{n_expert} expert steps. "
              f"Model takes over ({actual_remaining} remaining).")

    #  Build system prompt 
    system_content = (
        SYSTEM_PROMPT + "\n\n" + IN_CONTEXT_INSTRUCTION + "\n\n"
        + "Example:\n" + IN_CONTEXT_EXAMPLE + "\n\n"
        + f"Your global goal is: {goal}\n"
    )

    #  Model plays
    model_actions = []
    previous_dag = None
    won = False

    # The model gets up to max_model_steps
    if remaining != "full" and remaining >= 0 and actual_remaining > 0:
        step_budget = min(max_model_steps, actual_remaining * 2)
    else:
        step_budget = max_model_steps

    for step in range(step_budget):
        previous_error = None
        action_to_execute = None

        for retry in range(max_retries):
            user_content = build_eval_prompt(
                current_obs,
                current_cmds,
                model_actions,
                previous_dag,
                previous_error,
                goal,
            )
            try:
                response = model.generate(system_content, user_content)
            except Exception as e:
                previous_error = f"Model error: {e}"
                continue

            cmd, err = extract_command(response, current_cmds)
            if cmd:
                action_to_execute = cmd
                # Extract DAG for context continuity
                clean = clean_json_output(response)
                try:
                    parsed = json.loads(clean)
                    if "DAG" in parsed:
                        previous_dag = parsed["DAG"]
                except Exception:
                    pass
                break
            else:
                previous_error = err

        if action_to_execute is None:
            if verbose:
                print(f"    Step {step}: FAILED after {max_retries} retries. "
                      f"Last error: {previous_error}")
            break

        if verbose:
            print(f"    Step {step}: {action_to_execute}")

        # Execute
        obs_list, scores, dones, infos = env.step([action_to_execute])
        model_actions.append(action_to_execute)
        current_obs = obs_list[0]
        current_cmds = infos["admissible_commands"][0]

        if infos.get("won", [False])[0]:
            won = True
            break
        if dones[0]:
            break

    return {
        "won": won,
        "model_steps": len(model_actions),
        "prefix_steps": len(prefix_actions),
        "total_steps": len(prefix_actions) + len(model_actions),
        "model_actions": model_actions,
        "prefix_actions": prefix_actions,
        "remaining_target": remaining,
        "actual_remaining": actual_remaining,
        "goal": goal,
        "task_type": game_info["task_type"],
        "expert_length": n_expert,
    }


# Evaluation Loop

def _eval_worker(args_tuple):
    """Worker for parallel API-model evaluation (one game)."""
    config, split, model, game, remaining, max_model_steps, max_retries, verbose, game_idx = args_tuple
    env = make_env(config, split)
    # Advance env to the correct game by resetting game_idx times
    for _ in range(game_idx):
        env.reset()
    result = evaluate_single_game(
        env, model, game,
        remaining=remaining,
        max_model_steps=max_model_steps,
        max_retries=max_retries,
        verbose=verbose,
    )
    return game_idx, result


def run_evaluation(config, split, games, model, remaining_levels,
                   max_model_steps=30, max_retries=3, verbose=False,
                   parallel_workers=1):
    """Run evaluation across multiple games and remaining levels, with optional parallelization for API-based models. Collects results by level and prints detailed progress and summary statistics.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _level_key(r):
        return f"remaining_{r}" if r != "full" else "full"

    def _is_skip(remaining, n_expert):
        return remaining != "full" and remaining >= 0 and n_expert == 0 and remaining > 0

    #  Parallel path (API models only) 
    if parallel_workers > 1 and isinstance(model, OpenAIBackend):
        results_by_level = {_level_key(r): [None] * len(games) for r in remaining_levels}
        wins_by_level = {_level_key(r): 0 for r in remaining_levels}
        applicable_by_level = {_level_key(r): 0 for r in remaining_levels}

        tasks = []
        for remaining in remaining_levels:
            for gi, game in enumerate(games):
                tasks.append((
                    config, split, model, game, remaining,
                    max_model_steps, max_retries, verbose, gi
                ))

        done_count = 0
        #multi threading with ThreadPoolExecutor to evaluate multiple games in parallel, while keeping track of which game and remaining level each future corresponds to. As results come in, we update the per-level statistics and print progress.
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_meta = {}
            for t in tasks:
                remaining_val = t[4]
                gi = t[8]
                f = executor.submit(_eval_worker, t)
                future_meta[f] = (remaining_val, gi)
            for future in as_completed(future_meta):
                remaining_val, gi = future_meta[future]
                _, result = future.result()
                lk = _level_key(remaining_val)
                game = games[gi]
                skip = _is_skip(remaining_val, game["expert_length"])
                result["skipped"] = skip
                if not skip:
                    applicable_by_level[lk] += 1
                    if result["won"]:
                        wins_by_level[lk] += 1
                results_by_level[lk][gi] = result
                done_count += 1
                status = "WIN" if result["won"] else ("SKIP" if skip else "FAIL")
                w = wins_by_level[lk]
                a = max(applicable_by_level[lk], 1)
                print(
                    f"  [{done_count}/{len(tasks)}] Game {gi + 1} r={remaining_val} "
                    f"[{game['task_type']:12s}] {status:4s} | "
                    f"model_steps={result['model_steps']:2d} "
                    f"| SR={w}/{a} ({w / a * 100:.0f}%)"
                )
        for remaining in remaining_levels:
            lk = _level_key(remaining)
            w = wins_by_level[lk]
            a = max(applicable_by_level[lk], 1)
            print(f"  [{model.name}] Remaining={remaining}: {w}/{a} ({w / a * 100:.1f}%)")
        return results_by_level

    # Sequential path: games-first, levels-second 
    # Create one env per level, all starting at game 0.
    level_envs = {r: make_env(config, split) for r in remaining_levels}
    results_by_level = {_level_key(r): [] for r in remaining_levels}
    wins_by_level = {r: 0 for r in remaining_levels}
    applicable_by_level = {r: 0 for r in remaining_levels}

    for gi, game in enumerate(games):
        level_line_parts = []
        for remaining in remaining_levels:
            env = level_envs[remaining]
            skip = _is_skip(remaining, game["expert_length"])

            result = evaluate_single_game(
                env, model, game,
                remaining=remaining,
                max_model_steps=max_model_steps,
                max_retries=max_retries,
                verbose=verbose,
            )
            result["skipped"] = skip
            if not skip:
                applicable_by_level[remaining] += 1
                if result["won"]:
                    wins_by_level[remaining] += 1

            results_by_level[_level_key(remaining)].append(result)
            status = "W" if result["won"] else ("S" if skip else "F")
            w = wins_by_level[remaining]
            a = max(applicable_by_level[remaining], 1)
            level_line_parts.append(
                f"r={remaining}:{status}({result['model_steps']}s) {w}/{a}"
            )

        print(
            f"  Game {gi + 1:3d}/{len(games)} [{game['task_type']:12s}] | "
            + "  ".join(level_line_parts)
        )

    # Final per-level summary
    for remaining in remaining_levels:
        w = wins_by_level[remaining]
        a = max(applicable_by_level[remaining], 1)
        print(
            f"  [{model.name}] Remaining={remaining}: "
            f"{w}/{a} ({w / a * 100:.1f}%)"
        )

    return results_by_level


# Reporting
def compute_stats(episodes):
    """Compute summary stats from a list of episode result dicts."""
    valid = [e for e in episodes if not e.get("skipped", False)]
    if not valid:
        return {"sr": 0, "wins": 0, "total": 0, "avg_model_steps": 0}
    wins = sum(1 for e in valid if e["won"])
    return {
        "sr": wins / len(valid),
        "wins": wins,
        "total": len(valid),
        "avg_model_steps": np.mean([e["model_steps"] for e in valid]),
        "avg_total_steps": np.mean([e["total_steps"] for e in valid]),
    }


def compute_per_type_stats(episodes):
    """Break down stats by task type."""
    by_type = defaultdict(list)
    for e in episodes:
        if not e.get("skipped", False):
            by_type[e["task_type"]].append(e)
    result = {}
    for t, eps in sorted(by_type.items()):
        wins = sum(1 for e in eps if e["won"])
        result[t] = {
            "sr": wins / len(eps) if eps else 0,
            "wins": wins,
            "total": len(eps),
        }
    return result


def print_report(all_results, remaining_levels):
    """Print a comprehensive report with comparison tables."""
    model_names = list(all_results.keys())

    # ── Per-model, per-level summary ──
    for model_name in model_names:
        print(f"\n{'=' * 70}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 70}")
        for remaining in remaining_levels:
            level_key = f"remaining_{remaining}" if remaining != "full" else "full"
            episodes = all_results[model_name].get(level_key, [])
            stats = compute_stats(episodes)
            per_type = compute_per_type_stats(episodes)
            print(
                f"\n  Remaining = {remaining:>4s}  |  "
                f"SR: {stats['wins']}/{stats['total']} "
                f"({stats['sr'] * 100:.1f}%)  |  "
                f"Avg model steps: {stats.get('avg_model_steps', 0):.1f}"
            )
            print(f"  {'─' * 55}")
            for t, ts in per_type.items():
                print(
                    f"    {t:15s}: {ts['wins']:3d}/{ts['total']:3d} "
                    f"({ts['sr'] * 100:5.1f}%)"
                )

    # ── Cross-model comparison table ──
    if len(model_names) > 1 or len(remaining_levels) > 1:
        print(f"\n{'═' * 70}")
        print("  COMPARISON TABLE")
        print(f"{'═' * 70}")

        # Header
        header = f"  {'Remaining':>10s}"
        for mn in model_names:
            header += f"  {mn:>20s}"
        print(header)
        print(f"  {'─' * (10 + 22 * len(model_names))}")

        for remaining in remaining_levels:
            level_key = f"remaining_{remaining}" if remaining != "full" else "full"
            row = f"  {str(remaining):>10s}"
            for mn in model_names:
                episodes = all_results[mn].get(level_key, [])
                stats = compute_stats(episodes)
                cell = f"{stats['sr'] * 100:.1f}% ({stats['wins']}/{stats['total']})"
                row += f"  {cell:>20s}"
            print(row)
        print(f"{'═' * 70}")



def parse_remaining_arg(remaining_list):
    """Parse --remaining argument: ints and 'full'."""
    levels = []
    for r in remaining_list:
        if r.lower() == "full":
            levels.append("full")
        else:
            levels.append(int(r))
    return levels


def main():
    parser = argparse.ArgumentParser(
        description="ALFWorld Curriculum Evaluation — multi-model, same games, configurable remaining steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
       
    )
    parser.add_argument(
        "--models", nargs="+", default=["gpt4o_mini"],
        help=f"Model keys to evaluate. Available: {list(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--custom_model", type=str, default=None,
        help="Path to a custom HuggingFace model (added to model list)",
    )
    parser.add_argument(
        "--custom_lora", type=str, default=None,
        help="Path to LoRA adapter for the custom model",
    )
    parser.add_argument(
        "--custom_name", type=str, default="custom_model",
        help="Display name for the custom model",
    )
    parser.add_argument(
        "--split", default="valid_unseen",
        choices=["valid_seen", "valid_unseen", "train"],
        help="Evaluation split (default: valid_unseen)",
    )
    parser.add_argument(
        "--max_games", type=int, default=30,
        help="Number of games to evaluate (default: 30)",
    )
    parser.add_argument(
        "--remaining", nargs="+", default=["full"],
        help="Remaining steps from goal. Use ints and/or 'full'. "
             "E.g.: --remaining 1 3 5 full  (default: full)",
    )
    parser.add_argument(
        "--max_steps", type=int, default=30,
        help="Max steps the model can take per episode (default: 30)",
    )
    parser.add_argument(
        "--max_retries", type=int, default=3,
        help="Max retries per step for invalid output (default: 3)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-step actions",
    )
    parser.add_argument(
        "--output_dir", default="./eval_results",
        help="Directory to save results JSON (default: ./eval_results)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Generation temperature (default: 0.3)",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Parallel workers for API model evaluation (default: 1, recommended: 4-8 for gpt4o)",
    )
    parser.add_argument(
        "--backend", choices=["auto", "vllm", "transformers"], default="auto",
        help="Force backend for local models (default: auto = try vllm first)",
    )
    args = parser.parse_args()

    remaining_levels = parse_remaining_arg(args.remaining)

    # Set backend preference for local models
    _make_local_backend.force = args.backend

    # Load config 
    if not ALFWORLD_CONFIG:
        print("ERROR: ALFWorld config not found. Set ALFWORLD_CONFIG env var.")
        sys.exit(1)
    with open(ALFWORLD_CONFIG, "r") as f:
        config = yaml.safe_load(f)
    config["env"]["type"] = "AlfredTWEnv"

    # Collect games
    print(f"\n{'#' * 60}")
    print(f"# Collecting {args.max_games} games from {args.split}")
    print(f"{'#' * 60}")
    games = collect_games(config, args.split, args.max_games)

    # Expert length distribution
    lengths = [g["expert_length"] for g in games]
    type_dist = defaultdict(int)
    for g in games:
        type_dist[g["task_type"]] += 1
    print(f"\nGame stats:")
    print(f"  Expert length: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.0f}")
    print(f"  Task types: {dict(type_dist)}")
    print(f"  Remaining levels to test: {remaining_levels}")

    # ── Build model list ──
    model_keys = list(args.models)
    if args.custom_model:
        cname = args.custom_name
        MODEL_REGISTRY[cname] = lambda: LocalTransformersBackend(
            args.custom_model,
            lora_path=args.custom_lora,
            name=cname,
        )
        model_keys.append(cname)

    # Validate model keys
    for mk in model_keys:
        if mk not in MODEL_REGISTRY:
            print(f"ERROR: Unknown model key '{mk}'. "
                  f"Available: {list(MODEL_REGISTRY.keys())}")
            sys.exit(1)

    # ── Evaluate each model ──
    all_results = {}
    for model_key in model_keys:
        print(f"\n{'#' * 60}")
        print(f"# Evaluating: {model_key}")
        print(f"{'#' * 60}")

        model = MODEL_REGISTRY[model_key]()

        model_results = run_evaluation(
            config=config,
            split=args.split,
            games=games,
            model=model,
            remaining_levels=remaining_levels,
            max_model_steps=args.max_steps,
            max_retries=args.max_retries,
            verbose=args.verbose,
            parallel_workers=args.parallel,
        )
        all_results[model.name] = model_results

        # Free GPU memory
        if isinstance(model, (LocalTransformersBackend, VLLMBackend)):
            model.cleanup()

    # Report
    remaining_strs = [str(r) for r in remaining_levels]
    print_report(all_results, remaining_strs)

    # Save results to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remaining_tag = "_".join(str(r) for r in remaining_levels)
    out_path = os.path.join(
        args.output_dir,
        f"eval_curriculum_{args.split}_r{remaining_tag}_{timestamp}.json",
    )

    # Serialize
    serializable = {
        "config": {
            "split": args.split,
            "remaining_levels": [str(r) for r in remaining_levels],
            "max_games": args.max_games,
            "max_steps": args.max_steps,
            "temperature": args.temperature,
            "timestamp": timestamp,
        },
        "games": [
            {
                "game_idx": g["game_idx"],
                "gamefile": g["gamefile"],
                "goal": g["goal"],
                "task_type": g["task_type"],
                "expert_length": g["expert_length"],
            }
            for g in games
        ],
        "results": {},
    }
    for model_name, results_by_level in all_results.items():
        serializable["results"][model_name] = {}
        for level_key, episodes in results_by_level.items():
            stats = compute_stats(episodes)
            per_type = compute_per_type_stats(episodes)
            serializable["results"][model_name][level_key] = {
                "overall": stats,
                "per_type": per_type,
                "episodes": [
                    {
                        "game_idx": i,
                        "won": e["won"],
                        "model_steps": e["model_steps"],
                        "prefix_steps": e["prefix_steps"],
                        "total_steps": e["total_steps"],
                        "model_actions": e["model_actions"],
                        "remaining_target": e["remaining_target"],
                        "actual_remaining": e["actual_remaining"],
                        "task_type": e["task_type"],
                        "skipped": e.get("skipped", False),
                    }
                    for i, e in enumerate(episodes)
                ],
            }

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
