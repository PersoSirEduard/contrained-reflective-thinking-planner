import os
from pathlib import Path

# Notice, the setting only used for our speific sever, because we try to save some space 
os.environ["HF_HOME"] = "/workspace/hf_cache"      
os.environ["TMPDIR"] = "/workspace/alfworld_tmp"   



Path("/workspace/hf_cache").mkdir(parents=True, exist_ok=True)
Path("/workspace/alfworld_tmp").mkdir(parents=True, exist_ok=True)

import glob
import shutil





import comet_ml
import json 
#laod expert trajectories
def load_expert_trajectories(file_path):
    expert_traj_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Each line in the file is a JSON object, so we parse it and update our expert_traj_data dictionary
            obj = json.loads(line)
            expert_traj_data.update(obj)
    return expert_traj_data
loaded_expert_traj_data = load_expert_trajectories("Alfworld_expert_trajectories.json")
import re 
#convert expert command to target state for reward calculation
def expert_command_to_target_state(command):
    cmd = command.lower().strip()

    #handle look 
    if cmd == "look" or cmd == "inventory":
        return "Observation_update"
    #handle go to
    elif cmd.startswith("go to"):
        destination = cmd.replace("go to", "").strip()
        return f"Agent_At({destination})"
    #handle take
    elif cmd.startswith("take"):
        #"take creditcard 1 from sidetable 1"
        obj = re.match(r"take (.*) from (.*)", cmd)
        if obj:
            item = obj.group(1).strip()
            return f"Agent_Holding({item})"
    #handle put Move
    elif cmd.startswith("put") or cmd.startswith("move"):
        #"put obj in/on dst" or "move obj to dst"
        match = re.match(r"(?:put|move) (.*) (?:in|on|to) (.*)", cmd)
        if match:
            obj = match.group(1).strip()
            dst = match.group(2).strip()
            return f"Object_Location({obj}, {dst})"
    #handle open
    elif cmd.startswith("open"):
        obj = cmd.replace("open", "").strip()
        return f"Receptacle_Opened({obj})"
    #handle close
    elif cmd.startswith("close"):
        obj = cmd.replace("close", "").strip()
        return f"Receptacle_Closed({obj})"
    #handle heat cool clean 
    elif cmd.startswith("heat"):
        match = re.match(r"heat (.*) with (.*)", cmd)
        if match:
            obj = match.group(1).strip()

            return f"Object_Property({obj}, Hot)"
    elif cmd.startswith("cool"):
        match = re.match(r"cool (.*) with (.*)", cmd)
        if match:
            obj = match.group(1).strip()
            return f"Object_Property({obj}, Cool)"
    elif cmd.startswith("clean"):
        match = re.match(r"clean (.*) with (.*)", cmd)
        if match:
            obj = match.group(1).strip()
            return f"Object_Property({obj}, Clean)"
    #toggle 
    elif cmd.startswith("use") or cmd.startswith("toggle"):
        m = re.match(r"^(?:use|toggle) (.+)$", cmd)
        if m:
            obj = m.group(1).strip()
            return f"Object_Property({obj}, Toggled)"
    # SliceObject -> "Target item is sliced"
    elif cmd.startswith("slice"):
        match = re.match(r"slice (.*) with (.*)", cmd)
        if match:
            obj = match.group(1).strip()
            return f"Object_Property({obj}, Sliced)"
    #default, unknown command, return None
    return "Unknown_State"


    
import numpy as np
from alfworld.agents.environment import get_environment
import alfworld.agents.modules.generic as generic
import alfworld
import os
import yaml
import glob
import shutil


_curriculum_state = {
    "level": 4,              # Resume from level 4 (THIS IS OUR SETTING, PLEASE TRAIN Start from 2if you train from scratch)
                            
    "recent_wins": [],       # rolling window of 0/1 at current level
    "total_games": 0,
    "promotions": 2,          # 2 promotions already happened (2→3→4) OUR FINAL SETTING 
}
_CURR_WINDOW = int(os.environ.get("GRPO_CURRICULUM_WINDOW", "15")) # the window size for computing promotion stats — we use a relatively small window to make the curriculum more responsive to recent performance shifts, which is important in the early stages of training when the model is rapidly improving.
_CURR_THRESHOLD = float(os.environ.get("GRPO_CURRICULUM_THRESHOLD", "0.50"))# The threshold for promotion is 50% win rate in the recent window
_CURR_BACKOFF = float(os.environ.get("GRPO_CURRICULUM_BACKOFF", "0.10"))# avoid overfitting to frontier level by replaying easier levels with 10% probability

_CURR_MAX_LEVEL = int(os.environ.get("GRPO_CURRICULUM_MAX_LEVEL", "10"))# We set the max level to 10, which is higher than the typical trajectory length in Alfworld (usually less than 10 steps)

# the function to sample the curriculum level based on the current curriculum state and the total number of expert steps in the current trajectory. It implements a back-off mechanism to replay easier levels with some probability, and ensures that the sampled level does not exceed the trajectory length.
def sample_curriculum_remaining(n_expert_total: int):

    if n_expert_total <= 1:
        return 1, 1
    cur_level = _curriculum_state["level"]
    # Back-off: with small probability, replay an easier level for retention.
    # Uniform random from [1, cur_level-1] so ALL earlier skills get refreshed,
    # not just the immediately preceding one.
    if cur_level > 1 and np.random.random() < _CURR_BACKOFF:
        level_used = np.random.randint(1, cur_level)  # [1, cur_level)
    else:
        level_used = cur_level
    # Clamp to the trajectory length.
    remaining = min(level_used, n_expert_total)
    # level_played must reflect ACTUAL difficulty (= remaining), not the
    # sampled level.  Otherwise short tasks (n_expert_total < cur_level)
    # get counted as frontier-level wins → premature promotion.
    return remaining, remaining

# the function to update the curriculum state after each game, based on whether the model won or lost, and the level of difficulty played. It tracks the recent wins in a rolling window, and promotes the curriculum level if the win rate meets the threshold. It also handles the case where the model is already at the max level, and logs relevant information about the curriculum progress.
def update_curriculum(won: bool, level_played: int):

    _curriculum_state["total_games"] += 1
    if level_played != _curriculum_state["level"]:
        return _curriculum_state["level"]   # back-off game, don't gate on it
    wins = _curriculum_state["recent_wins"]
    wins.append(1 if won else 0)
    if len(wins) > _CURR_WINDOW:
        wins.pop(0)
    if _curriculum_state["level"] >= _CURR_MAX_LEVEL:
        if len(wins) >= _CURR_WINDOW and not _curriculum_state.get("_cap_logged"):
            rate = sum(wins) / len(wins)
            print(f"[curriculum] AT CAP level {_curriculum_state['level']} "
                  f"(max={_CURR_MAX_LEVEL}), win_rate={rate:.2f} — training at ceiling")
            _curriculum_state["_cap_logged"] = True
        return _curriculum_state["level"]
    if len(wins) >= _CURR_WINDOW and (sum(wins) / len(wins)) >= _CURR_THRESHOLD:
        _curriculum_state["level"] += 1
        _curriculum_state["promotions"] += 1
        _curriculum_state["recent_wins"] = []
        if _curriculum_state["level"] >= _CURR_MAX_LEVEL:
            print(f"[curriculum] PROMOTED to MAX level {_curriculum_state['level']} "
                  f"(cap={_CURR_MAX_LEVEL}) — training at ceiling from now on")
        else:
            print(f"[curriculum] PROMOTED to level {_curriculum_state['level']} "
                  f"(window win rate met threshold {_CURR_THRESHOLD:.2f})")
    return _curriculum_state["level"]




def get_curriculum_stats():
    wins = _curriculum_state["recent_wins"]
    rate = (sum(wins) / len(wins)) if wins else 0.0
    return {
        "level": _curriculum_state["level"],
        "window_win_rate": rate,
        "window_size": len(wins),
        "total_games": _curriculum_state["total_games"],
        "promotions": _curriculum_state["promotions"],
    }

#this functio is for us to use to avoid us too much space on our sever 
def cleanup_textworld_tmp():
    # 1) Remove dirs that contain libdownward.so (successful textworld starts)
    for p in glob.glob('/tmp/tmp*/libdownward.so'):
        try:
            shutil.rmtree(os.path.dirname(p))
        except OSError:
            pass
    # 2) Remove dirs that contain fast_downward artifacts (partial/failed copies)
    for p in glob.glob('/tmp/tmp*/fast_downward*'):
        try:
            shutil.rmtree(os.path.dirname(p))
        except OSError:
            pass
    # 3) Remove empty /tmp/tmp* dirs (orphans from failed textworld.start)
    for p in glob.glob('/tmp/tmp*'):
        if os.path.isdir(p):
            try:
                if not os.listdir(p):
                    os.rmdir(p)
            except OSError:
                pass

# the alfworld setup
os.environ['ALFWORLD_DATA'] = "/workspace/alfworld_data"
config_path = "/workspace/alfworld/configs/base_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
#Overwrite the environment type in the configuration
config['env']['type'] = 'AlfredTWEnv'
#initialize the environment
env = get_environment(config['env']['type'])(config, train_eval='train')
env = env.init_env(batch_size=1)

obs, info = env.reset()
import json

#the system prompt 
system_prompt = (
    "You are an embodied AI agent in the ALFWorld text-based environment. Your goal is to achieve the given task optimally. "
    "You must reason using a Directed Acyclic Graph (DAG) format to plan and track your progress.\n\n"
    "CRITICAL DAG RULES:\n"
    "1. ALLOWED STATE TAGS (Nodes): Every node MUST use one of these tags to represent a verifiable world state:\n"
    "   - Agent_At(receptacle_id)\n"
    "   - Agent_Holding(object_id)\n"
    "   - Object_Location(object_id, receptacle_id)\n"
    "   - Receptacle_Opened(receptacle_id)\n"
    "   - Object_Property(object_id, property_type) [e.g., Clean, Hot, Cool, Examined, Sliced, On]\n"    
    "2. ALLOWED ACTION TAGS (Edges): Edges represent transitions and must use these abstract tags:\n"
    "   - GOTO(receptacle_id), TAKE(object_id, receptacle_id), PUT(object_id, receptacle_id), OPEN(receptacle_id), CLOSE(receptacle_id), HEAT(object_id, receptacle_id), COOL(object_id, receptacle_id), CLEAN(object_id, receptacle_id), TOGGLE(object_id), EXAMINE(object_id, receptacle_id)\n"
    "3. STRICT PRUNING (NO HISTORY LOGGING): The DAG is your CURRENT valid plan, NOT a history log. If a path fails (e.g., you searched a cabinet and it was empty), you MUST COMPLETELY DELETE that node and its edges in your next response. Keep the DAG compact.\n"
    "4. NO FALSE DEPENDENCIES: Do not chain independent search actions. Parallel search locations must point to the EXACT SAME target state (e.g., two GOTO edges both pointing to 'Object_Location(target, known)').\n"
    "5. DYNAMIC EXECUTION: Find the first 'pending' node whose prerequisite nodes are 'completed'. This is your Target_Goal. In Action_Chunks, output the EXACT admissible command string to progress along that edge."
    


)

#in context learning format 
in_conext_prompt_instruction = "CRITICAL INSTRUCTION: You MUST output your internal reasoning and next action STRICTLY in the following JSON format. Do NOT output any natural language conversational text before or after the JSON block. Your entire response must be parseable by a JSON parser."
#in context learning format example COT
in_context_example = """{
    "Reflection": "The task is to put a hot apple in the garbage can. I need to find the apple first. It might be on the countertop or in the fridge. I will plan parallel search paths.",
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
        "Checker": "S2, S3, S4, and S5 are future states. I need to achieve S1. I have parallel edges to countertop 1 and fridge 1. I will check countertop 1 first. If it's not there, I will prune the countertop edge in the next step and try the fridge."
    },
    "Action_Chunks": {
        "Subgoal": "S1",
        "Command": "go to countertop 1"
    }
}"""
#in context learning format example DAG 

#def create_full_prompt(system_prompt,in_conext_prompt_instruction,in_context_example,dynamic_prompt):
#    full_prompt = system_prompt + "\n\n" + in_conext_prompt_instruction + "\n\n" + "Here is an example of the JSON format for the reasoning DAG and action chunk:\n" + in_context_example + "\n\n" + "Now, based on the current observation and admissible commands, please output your internal reasoning and next action in the specified JSON format.\n\n" + dynamic_prompt
#    return full_prompt

import json 
class DAGManager:
    def __init__(self):
        self.current_DAG = None
    def extract_and_update_DAG(self,llm_output):
        #parse the LLM output to extract the DAG and action chunk
        clean_llm_output = llm_output.strip()
        # Strip Qwen3 thinking chain: <think>...</think>
        think_match = re.search(r'<think>.*?</think>\s*', clean_llm_output, re.DOTALL)
        if think_match:
            clean_llm_output = clean_llm_output[think_match.end():].strip()
        #clean format 
        if clean_llm_output.startswith("```json"):
            clean_llm_output = clean_llm_output[len("```json"):].strip()
        if clean_llm_output.startswith("```"):
            clean_llm_output = clean_llm_output[len("```"):].strip()
        if clean_llm_output.endswith("```"):
            clean_llm_output = clean_llm_output[:-len("```")].strip()
        #parse the cleaned output as JSON
        try:
            parased_data = json.loads(clean_llm_output)
            #check if the parsed data contains the DAG an
            if "DAG" not in parased_data:
                return False, "Missing 'DAG' in the output."
            
            #DAG data 
            dag_data = parased_data["DAG"]

            #check if the DAG data contains the necessary fields
            if "nodes" not in dag_data or "edges" not in dag_data:
                return False, "DAG data must contain 'nodes' and 'edges'."
            
            #ok successfully extract the DAG and action chunk
            self.current_DAG = dag_data
            return True, "DAG extracted and updated successfully."
        except json.JSONDecodeError as e:
            return False, f"JSON parsing error: {str(e)}"
    def get_current_DAG(self):
        return self.current_DAG
    

    #this validator is used to check whether the format of LLM output and whether the action follow the admissible commands
class ProgrammaticValidator:
    def __init__(self):
        pass
    def clean_command(self, llm_output, ):
        """
        Parameters:
        llm_output: the raw output from LLM
        Returns:
        cleaned_command: the cleaned command string extracted from the LLM output
        """
        #clean the llm output
        clean_llm_output = llm_output.strip()
        # Strip Qwen3 thinking chain: <think>...</think>
        think_match = re.search(r'<think>.*?</think>\s*', clean_llm_output, re.DOTALL)
        if think_match:
            clean_llm_output = clean_llm_output[think_match.end():].strip()
        elif clean_llm_output.startswith('<think>'):
            # Incomplete think block — try to find end or strip prefix
            end_idx = clean_llm_output.find('</think>')
            if end_idx != -1:
                clean_llm_output = clean_llm_output[end_idx + len('</think>'):].strip()
        #clean the llm output to ensure it is a valid JSON format
        if clean_llm_output.startswith("```json"):
            clean_llm_output = clean_llm_output[len("```json"):].strip()
        if clean_llm_output.startswith("```"):
            clean_llm_output = clean_llm_output[len("```"):].strip()
        if clean_llm_output.endswith("```"):
            clean_llm_output = clean_llm_output[:-len("```")].strip()
        return clean_llm_output
    def valiate_detailed(self,llm_output,admissible_commands):
        #detail of validation process
        detail = {
            "valid_json": False,
            "has_required_keys": False,
            "is_command_admissible": False,
            "command": None,
            "parsed_output": None,
            "error_message": "",
            "Subgoal": None
        }
        #clean the llm output
        clean_llm_output = self.clean_command(llm_output)
        #try to parse the JSON output
        try:
            parsed_output = json.loads(clean_llm_output)
            detail["valid_json"] = True
            detail["parsed_output"] = parsed_output
        except json.JSONDecodeError as e:
            detail["error_message"] = f"JSON parsing error: {str(e)}"
            return  detail
        
        # need to be dict 
        if not isinstance(parsed_output, dict):
            detail["error_message"] = "Output is not a dictionary."
            return  detail
        
        #check format schema {"Subgoal", "Command"}
        if "Subgoal"  in parsed_output and "Command"  in parsed_output:
            #we got all the required keys
            detail["has_required_keys"] = True

            #extract the command
            command = str(parsed_output["Command"]).strip()
            detail["command"] = command

            #check if the command is in admissible commands
            if command in admissible_commands:
                detail["is_command_admissible"] = True
                return detail
            else:
                detail["error_message"] = f"Command '{command}' is not in admissible commands."
                return  detail

            
        #get action chunk from reasoning DAG if it exist
        action_chunk = parsed_output.get('Action_Chunks')
        #fix potential json fromat issue 
        if isinstance(action_chunk, list):
           if len(action_chunk)>0:
                action_chunk = action_chunk[0]
           else:
               action_chunk = None

        if isinstance(action_chunk,dict):
            if  "Subgoal" in action_chunk and "Command" in action_chunk:
                detail["has_required_keys"] = True

                subgoal = str(action_chunk["Subgoal"]).strip()
                command = str(action_chunk["Command"]).strip()

                detail["subgoal"] = subgoal
                detail["command"] = command

                if command in admissible_commands:
                    detail["is_command_admissible"] = True
                else:
                    detail["error_message"] = f"Command '{command}' is not in admissible commands."
        return  detail
                
    def validate(self, llm_output, admissible_commands):
        detail = self.valiate_detailed(llm_output, admissible_commands)
        if detail["is_command_admissible"]:
            return True, detail["command"]
            
        return False, detail["error_message"]


import json 
# we need to ensure it follow the same format 
def normalize_text(text):
    if text is None:
        return None
    return " ".join(text.strip().lower().split())

#extract original observation to get the global goal
def extract_global_goal(observation):
    #if it has ""Your task is to:" in the observation, we can extract the global goal from there
    if "Your task is to:" in observation:
        return observation.split("Your task is to:")[-1].strip()
    #if not, we just return a default prompt 
    return "Complete the task described by the environment."


#compare the state from DAG with expert state to get the reward 
def compute_dag_reward(llm_dag, expert_state):
    #if no llm dag or nodes not exist, return zero reward
    if not llm_dag or "nodes" not in llm_dag:
        return 0.0
    #extract LLM dag states
    llm_planned_states = [node_info[0] for node_info in llm_dag["nodes"].values()]

    #we need convert ['Object_Location', 'apple 1', 'fridge 1'] to ('object_location', 'apple', 'fridge')
    def convert_state_format(state_str):
        if isinstance(state_str, str):
            m = re.match(r'\s*([A-Za-z_]+)\s*\((.*)\)\s*', state_str)
            if not m:
                return ()
        #get the tag
            tag = m.group(1).strip().lower()
                #the args, we need covert apple1 to apple 
            args = []
            for arg in m.group(2).split(','):
                clean_arg = re.sub(r'\d+', '', str(arg)).strip().lower().replace(' ', '')
                if clean_arg:
                    args.append(clean_arg)

        #covert to tuple 
            return tuple([tag] + args)
        return ()
    # convert expert state and llm state to set
    llm_state = set(
        convert_state_format(state) for state in llm_planned_states
        if state
    )
    expert_state_converted = set(
        convert_state_format(state) for state in expert_state
        if state
    )
    #remove empty tuple if exist
    llm_state.discard(())
    expert_state_converted.discard(())

    if not expert_state_converted:
        return 0.0
    
    def state_soft_match(llm_s, exp_s):
        #match the tag
        if llm_s[0] != exp_s[0]:
            return 0.0
        #match the args
        llm_args = llm_s[1:]
        exp_args = exp_s[1:]
        #exact match
        if llm_args == exp_args:
            return 1.0
        #Object location 
        if llm_s[0] == "object_location":
            if len(llm_args) >= 2 and len(exp_args) >= 2:
                if llm_args[0] == exp_args[0]: #same object
                    if "known" in llm_args or "known" in exp_args:
                        return 0.5#both are known but different location, still good
        #save predicate+partial perdict
        overlap =len(set(llm_args).intersection(set(exp_args)))
        if overlap == 0:
            return 0.0
        return overlap / max(len(exp_args), 1)
    matched_scores = []
    for exp_s in expert_state_converted:
        best_score = 0.0
        for llm_s in llm_state:
            score = state_soft_match(llm_s, exp_s)
            best_score = max(best_score, score)
        matched_scores.append(best_score)
    #compute the average matched score as coverage
    coverage = sum(matched_scores) / len(expert_state_converted)



    return coverage




def find_task_id_by_goal(loaded_expert_traj_data, global_goal):
    goal = global_goal.strip().lower()
    # Primary: exact goal match in Initial_Observation
    for task_id, traj in loaded_expert_traj_data.items():
        init_obs = traj[0].get("Initial_Observation", "").lower()
        if goal in init_obs:
            return task_id
    # Fallback: extract entity keywords from goal and match against task_id string
    # Handles pick_two tasks where goal text differs from Initial_Observation
    import re
    goal_words = set(re.findall(r'[a-z]+(?:lamp|pan|pot|cup|mug|plate|bowl|vase|knife|fork|spoon|pen|pencil|book|pillow|cloth|towel|soap|candle|key|box|bottle|apple|tomato|potato|egg|lettuce|bread|cell|remote|cd|watch|alarm|credit|statue|newspaper)', goal))
    if not goal_words:
        # Fallback: use nouns longer than 3 chars, skip common words
        stop_words = {"your", "task", "find", "some", "then", "place", "them", "with", "from", "into", "that", "this", "have", "will", "clean", "heat", "cool", "pick"}
        goal_words = {w for w in re.findall(r'[a-z]{4,}', goal) if w not in stop_words}
    if goal_words:
        best_task_id = None
        best_score = 0
        for task_id in loaded_expert_traj_data:
            tid_lower = task_id.lower().replace("-", " ").replace("_", " ")
            score = sum(1 for w in goal_words if w in tid_lower)
            if score > best_score:
                best_score = score
                best_task_id = task_id
        if best_score >= 1:
            return best_task_id
    return None

#interact with env 
def env_interact(env, command):
    obs,scores,dones,infos = env.step([command])
    #what are next obs 
    nex_obs = obs[0]
    score = scores[0]
    done = dones[0]
    next_admissible_commands = infos["admissible_commands"][0]
    won = infos.get("won", [False])[0]
    return nex_obs, score, done, next_admissible_commands, won, infos































































# the actual training loop and model setup






from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions


# Guided decoding 
# we use guided decoding to ensure the model output follow the required format, which is more sample efficient for learning the reasoning DAG and action chunk format. We implement it as a generation override in generate_rollout_completions, which is compatible with different versions of vLLM that may have different parameter classes for structured/guided decoding. If the import fails (e.g., due to an older vLLM version), we fall back to post-hoc validation only, which is less sample efficient but still provides a safety net against format errors.
if os.environ.get("GRPO_DISABLE_GUIDED", "").strip() != "1":
    try:
        from vllm.sampling_params import StructuredOutputsParams as _GUIDED_PARAM_CLS  
        _GUIDED_KEY = "structured_outputs"
    except Exception:
        try:
            from vllm.sampling_params import GuidedDecodingParams as _GUIDED_PARAM_CLS  
            _GUIDED_KEY = "guided_decoding"
        except Exception:
            _GUIDED_PARAM_CLS = None
            _GUIDED_KEY = None
            print("[guided-decoding] vLLM structured/guided params not importable — "
                  "falling back to post-hoc validator only.")
if _GUIDED_PARAM_CLS is not None:
    print(f"[guided-decoding] ENABLED via {_GUIDED_PARAM_CLS.__name__} (key={_GUIDED_KEY})")
else:
    print("[guided-decoding] DISABLED (GRPO_DISABLE_GUIDED=1 or import failed)")


def build_action_chunk_schema(admissible_cmds):
    """JSON schema for structured output — ensures valid JSON format.

    Command is a free string (NOT constrained to admissible enum) so that
    different generations can explore different commands → non-zero GRPO
    reward variance.  Post-hoc fuzzy_match_admissible handles correction.
    """
    cmd_schema = {"type": "string"}   # NO enum — key fix for zero-variance
    return {
        "type": "object",
        "properties": {
            "Reflection": {"type": "string"},
            "DAG": {"type": "object"},
            "Action_Chunks": {
                "type": "object",
                "properties": {
                    "Subgoal": {"type": "string"},
                    "Command": cmd_schema,
                },
                "required": ["Command"],
            },
        },
        "required": ["Action_Chunks"],
    }


def build_guided_overrides(admissible_cmds):
    """Return generation_overrides dict for generate_rollout_completions.

    Falls back to {} when guided decoding is unavailable — caller code is
    unchanged either way; post-hoc ProgrammaticValidator is still the safety
    net.
    """
    if _GUIDED_PARAM_CLS is None or not admissible_cmds:
        return {}
    schema = build_action_chunk_schema(admissible_cmds)
    try:
        param = _GUIDED_PARAM_CLS(json=schema)
    except TypeError:
        # older vLLM versions expect a JSON string
        try:
            param = _GUIDED_PARAM_CLS(json=json.dumps(schema))
        except Exception as e:
            print(f"[guided-decoding] could not build params for {len(admissible_cmds)} cmds: {e}")
            return {}
    except Exception as e:
        print(f"[guided-decoding] unexpected error building params: {e}")
        return {}
    return {_GUIDED_KEY: param}

# extract plan command from LLM completion, with cleaning and parsing. Returns (commands, full_parsed_json) where commands is a list of extracted command strings (or None if parsing failed), and full_parsed_json is the entire parsed output for potential additional insights (or None if parsing failed). The extraction logic looks for "Command" fields in "Action_Chunks" or "Plan" sections, and also supplements with commands derived from DAG edges if needed.
def extract_plan_commands(completion_text):
    # clean the completion text to isolate the JSON block, handling Qwen3 think chains and code block formatting
    clean = completion_text.strip()
    think_match = re.search(r'<think>.*?</think>\s*', clean, re.DOTALL)
    if think_match:
        clean = clean[think_match.end():].strip()
    elif clean.startswith('<think>'):
        return None, None  # incomplete thinking
    if clean.startswith("```json"): clean = clean[len("```json"):].strip()
    if clean.startswith("```"): clean = clean[len("```"):].strip()
    if clean.endswith("```"): clean = clean[:-len("```")].strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    commands = []
    # 1) Extract commands from "Plan" if it exists
    plan = parsed.get("Plan", [])
    if isinstance(plan, list):
        for step in plan:
            if isinstance(step, dict) and "Command" in step:
                commands.append(str(step["Command"]).strip())
    # 2) Extract commands from "Action_Chunks" if it exists (handles both singular and plural forms)
    if not commands:
        ac = parsed.get("Action_Chunks", parsed.get("Action_Chunk", None))
        if isinstance(ac, dict) and "Command" in ac:
            commands.append(str(ac["Command"]).strip())
        elif isinstance(ac, list):
            for chunk in ac:
                if isinstance(chunk, dict) and "Command" in chunk:
                    commands.append(str(chunk["Command"]).strip())
    # 3) Extract commands from DAG edges as fallback/supplement
    dag = parsed.get("DAG", {})
    if isinstance(dag, dict):
        edges = dag.get("edges", [])
        dag_cmds = []
        action_to_cmd = {
            "GOTO": "go to", "TAKE": "take", "PUT": "move",
            "OPEN": "open", "CLOSE": "close", "HEAT": "heat",
            "COOL": "cool", "CLEAN": "clean", "TOGGLE": "use",
            "EXAMINE": "examine", "SLICE": "slice",
        }
        for edge in edges:
            if isinstance(edge, dict) and "action" in edge:
                action_str = str(edge["action"]).strip()
                # Parse e.g. "GOTO(shelf 1)" -> "go to shelf 1"
                m = re.match(r'(\w+)\((.+)\)', action_str)
                if m:
                    tag = m.group(1).upper()
                    args = [a.strip() for a in m.group(2).split(',')]
                    prefix = action_to_cmd.get(tag)
                    if prefix and args:
                        if tag == "TAKE" and len(args) >= 2:
                            dag_cmds.append(f"take {args[0]} from {args[1]}")
                        elif tag == "PUT" and len(args) >= 2:
                            dag_cmds.append(f"move {args[0]} to {args[1]}")
                        elif tag in ("HEAT", "COOL", "CLEAN", "SLICE") and len(args) >= 2:
                            dag_cmds.append(f"{prefix} {args[0]} with {args[1]}")
                        else:
                            dag_cmds.append(f"{prefix} {args[0]}")
        # If we only got 0-1 commands from Plan/Action_Chunks, supplement with DAG edges
        if len(commands) <= 1 and dag_cmds:
            # Keep any existing command as first, then add DAG-derived ones
            seen = set(normalize_text(c) for c in commands)
            for dc in dag_cmds:
                if normalize_text(dc) not in seen:
                    commands.append(dc)
                    seen.add(normalize_text(dc))
    return commands if commands else None, parsed

# it's very likely that the LLM's generated command won't exactly match the admissible commands due to formatting differences, synonyms, or minor errors. This function implements a fuzzy matching heuristic to find the best matching admissible command for a given plan command. It first tries exact matches (ignoring case and whitespace), then it tries to match based on verb and key object similarity, allowing for some flexibility in phrasing while still guiding towards the intended action. If a good match is found (score >= 1), it returns the matched admissible command; otherwise, it returns None.
def fuzzy_match_admissible(plan_cmd, admissible_commands):
    """Find the best matching admissible command for an abstract plan command.
    Returns the matched admissible command or None."""
    plan_cmd_lower = plan_cmd.strip().lower()
    # Exact match first
    for ac in admissible_commands:
        if ac.strip().lower() == plan_cmd_lower:
            return ac
    # Extract verb and key object from plan command
    # e.g. "take apple from current_loc" -> verb="take", key_obj="apple"
    # e.g. "go to shelf 1" -> verb="go to", key_obj="shelf 1"
    # e.g. "move apple to garbagecan 1" -> verb="move", key_obj="apple"
    parts = plan_cmd_lower.split()
    if not parts:
        return None
    verb = parts[0]
    # Handle "go to" as a special case since it's a common multi-word verb
    if verb == "go" and len(parts) > 1 and parts[1] == "to":
        verb = "go to"
        rest = " ".join(parts[2:])
    else:
        rest = " ".join(parts[1:])
    # Strip "from ...", "to ...", "with ...", "in ...", "on ..." to get key object
    key_obj = re.split(r'\s+(?:from|to|with|in|on)\s+', rest)[0].strip()
    # Remove trailing digits to handle "apple" matching "apple 1"
    key_obj_base = re.sub(r'\s*\d+$', '', key_obj).strip()
    best_match = None
    best_score = 0
    for ac in admissible_commands:
        ac_lower = ac.strip().lower()
        # Must have same verb
        if verb == "go to":
            if not ac_lower.startswith("go to"):
                continue
        elif not ac_lower.startswith(verb + " "):
            if not ac_lower == verb:
                continue
        # Score: how well the key object matches
        score = 0
        if key_obj and key_obj in ac_lower:
            score = 3  # exact key object match (e.g. "shelf 1" in "go to shelf 1")
        elif key_obj_base and key_obj_base in ac_lower:
            score = 2  # base object match (e.g. "apple" in "take apple 1 from countertop 1")
        elif verb == ac_lower.split()[0]:
            score = 1  # just verb match
        if score > best_score:
            best_score = score
            best_match = ac
    return best_match if best_score >= 1 else None



# ══════════════════════════════════════════════════════════════════════════════
# Potential-based reward system — PDDL fact diffing.
# Instead of parsing feedback text, we read ground-truth PDDL facts directly
# from the TextWorld engine via game_state["facts"].  Each milestone predicate
# has a potential value.  Reward = Φ_new − Φ_old_max (one-shot per milestone).
# ══════════════════════════════════════════════════════════════════════════════


def _fact_entity_base(arg):
    """Strip trailing _N from a PDDL entity name: 'plate_2' → 'plate'.""" 
    """because the same object can appear multiple times with different suffixes, but we want them to all contribute to the same milestones."""
    return re.sub(r"_\d+$", "", arg)


def extract_goal_entities(global_goal):
    """Parse ALFWorld global goal text to extract target entity base names.

    Returns a set of lowercase base names, e.g. {'plate', 'countertop'}.
    Only PDDL facts whose entity arguments match these names will contribute
    to potential — this prevents the model from grabbing irrelevant objects
    to farm milestone rewards.

    Covers all 6 ALFWorld task families:
      clean/heat/cool some X and put it in Y  →  {X, Y}
      put a/some [clean/hot/cool] X in/on Y   →  {X, Y}
      find two X and put them in/on Y          →  {X, Y}
      put two X in/on Y                        →  {X, Y}
      look at X under the Y                    →  {X, Y}
      examine the X with the Y                 →  {X, Y}

    Falls back to empty set (= no filtering) if parsing fails.
    """
    goal = global_goal.lower().strip().rstrip(".")

    # "heat/clean/cool some X and put it in/on Y"
    m = re.match(
        r"(?:heat|clean|cool|slice)\s+(?:a|some|the)\s+(\w+)"
        r"\s+and\s+put\s+it\s+(?:in|on)\s+(.+)", goal)
    if m:
        return {m.group(1).strip(), m.group(2).strip()}

    # "find two X and put them in/on Y"
    m = re.match(
        r"find\s+two\s+(\w+)\s+and\s+put\s+them\s+(?:in|on)\s+(.+)", goal)
    if m:
        return {m.group(1).strip(), m.group(2).strip()}

    # "put a/some/two [cool/hot/clean/sliced] X in/on Y"
    m = re.match(
        r"put\s+(?:a|an|some|two|the)\s+"
        r"(?:cool\s+|hot\s+|clean\s+|sliced\s+)?"
        r"(\w+)\s+(?:in|on)\s+(.+)", goal)
    if m:
        return {m.group(1).strip(), m.group(2).strip()}

    # "look at X under the Y"
    m = re.match(
        r"look\s+at\s+(?:a\s+|an\s+|some\s+|the\s+)?"
        r"(\w+)\s+under\s+(?:the|a)\s+(.+)", goal)
    if m:
        return {m.group(1).strip(), m.group(2).strip()}

    # "examine the X with the Y"
    m = re.match(
        r"examine\s+(?:a\s+|an\s+|the\s+)?(\w+)"
        r"\s+with\s+(?:the|a)\s+(.+)", goal)
    if m:
        return {m.group(1).strip(), m.group(2).strip()}

    # Fallback: no filtering
    return set()


# PDDL predicates → potential values.  Only milestone predicates get > 0.
# Navigation (atLocation), container ops (opened), bookkeeping (checked) = 0.
FACT_POTENTIAL = {
    "holds":          0.5,    # agent picked up an object 
    "isclean":        1.0,    # object cleaned 
    "ishot":          1.0,    # object heated 
    "iscool":         1.0,    # object cooled 
    "issliced":       1.0,    # object sliced 
    "istoggled":      1.0,    # object toggled 
    "inreceptacle":   1.5,    # object placed in target receptacle 
}


def extract_facts_set(game_state):
    """Extract PDDL facts from a TextWorld game_state as a set of strings.

    Each fact becomes e.g. 'holds agent apple_0', 'ishot apple_0',
    'inreceptacle apple_0 garbagecan_0'.  Returns empty set if facts
    are unavailable.
    """
    raw_facts = game_state.get("facts") if game_state is not None else None
    if not raw_facts:
        return set()
    facts = set()
    for fact in raw_facts:
        try:
            name = fact.name.lower().strip()
            args = " ".join(n.strip() for n in fact.names)
            facts.add(f"{name} {args}")
        except Exception:
            continue
    return facts


def compute_potential_from_facts(facts_set, goal_entities=None):
    """Compute the potential  from a set of PDDL fact strings.

    Returns the max potential value among all milestone facts present.
    This means  tracks the highest milestone achieved so far.

    If *goal_entities* is provided (non-empty set of base names), only
    facts whose entity arguments match the goal are counted:
      - holds:          the held object must be a goal entity
      - property facts:  the object must be a goal entity
      - inreceptacle:   BOTH the object AND the receptacle must be
                        goal entities (prevents rewarding intermediate
                        placements like putting plate in microwave)
    """
    phi = 0.0
    for fact_str in facts_set:
        parts = fact_str.split()
        if not parts:
            continue
        predicate = parts[0]
        p = FACT_POTENTIAL.get(predicate, 0.0)
        if p <= 0.0:
            continue

        # ── Entity filtering ──
        if goal_entities:
            # entity args = everything except 'agent' (always the player)
            entity_args = [a for a in parts[1:] if a != "agent"]
            base_names = {_fact_entity_base(a) for a in entity_args}
            if predicate == "inreceptacle":
                # Both object AND receptacle must be goal-relevant
                if not base_names.issubset(goal_entities):
                    continue
            else:
                # At least one arg must be goal-relevant
                if not (base_names & goal_entities):
                    continue

        if p > phi:
            phi = p
    return phi


def score_step_with_potential(completion_text, admissible_cmds,
                              game_state, tw_env,
                              recent_actions=None,
                              phi_max=0.0,
                              goal_entities=None,
                              expert_step_new_facts=None):
    """Score a completion using potential-based reward + expert state matching.

    Reward budget:
        r_format_fail  -1.0   invalid JSON (hard termination signal)
        r_inadmissible -0.5   command not in admissible set
        pen_noop       -0.1   look / inventory (observation-only)
        pen_repeat     -0.2   repeating a recent action (spinning)
        r_state_match  +0.3…+1.0  model's state change overlaps expert's
        r_potential    +0.5…+1.0  potential transition (milestone achieved)
        r_win          +5.0   task completed (dominates everything)
        pen_length     -var   token bloat penalty

    r_state_match compares PDDL fact diffs: the expert plays from the same
    position in a separate game, and we record what facts change at each step.
    The model's action is rewarded if it produces the same state changes.
    This provides gradient signal for ALL steps (including navigation), not
    just milestone steps.

    Returns (reward, first_command, is_valid, reward_components, phi_after).
    """
    rc = {
        "r_format_fail": 0.0,
        "r_inadmissible": 0.0,
        "pen_noop": 0.0,
        "pen_repeat": 0.0,
        "r_state_match": 0.0,
        "r_potential": 0.0,
        "r_win": 0.0,
        "pen_length": 0.0,
        "total": 0.0,
    }

    # Extract first command 
    commands, parsed = extract_plan_commands(completion_text)
    if parsed is None:
        rc["r_format_fail"] = -1.0
        rc["total"] = -1.0
        return -1.0, None, False, rc, phi_max

    if not commands:
        validator = ProgrammaticValidator()
        detail = validator.valiate_detailed(completion_text, admissible_cmds)
        if detail["has_required_keys"] and detail["command"]:
            commands = [detail["command"]]
        else:
            rc["r_format_fail"] = -1.0
            rc["total"] = -1.0
            return -1.0, None, False, rc, phi_max

    first_cmd = commands[0]
    reward = 0.0

    #  Admissibility 
    if first_cmd not in admissible_cmds:
        matched = fuzzy_match_admissible(first_cmd, admissible_cmds)
        if matched:
            first_cmd = matched
        else:
            rc["r_inadmissible"] = -0.5
            rc["total"] = -0.5
            return -0.5, first_cmd, False, rc, phi_max

    #  Behavioral penalties 
    # If the expert's action at this step also produces no state change
    # (empty fact diff, e.g. expert does "look"), skip the noop penalty —
    # the model is doing what the expert would do.  Only applies to
    # look/inventory, NOT to help or other invalid commands.
    expert_is_noop = (expert_step_new_facts is not None
                      and isinstance(expert_step_new_facts, set)
                      and len(expert_step_new_facts) == 0)
    cmd_lower = normalize_text(first_cmd)
    if cmd_lower in ("look", "inventory"):
        if not expert_is_noop:
            reward -= 0.1
            rc["pen_noop"] = -0.1

    if recent_actions:
        recent_norm = [normalize_text(a) for a in recent_actions[-3:]]
        if cmd_lower in recent_norm:
            reward -= 0.2
            rc["pen_repeat"] = -0.2

    #  Execute in env → potential reward via PDDL fact DIFF 
    # We diff facts BEFORE and AFTER tw_env.step() so only NEW milestone
    # facts from THIS command contribute to potential — pre-existing facts
    # (from expert prefix replay or game initial state) are excluded.
    phi_after = phi_max
    if game_state is not None and tw_env is not None:
        try:
            adm = getattr(game_state, 'admissible_commands', [])
            exec_cmd = first_cmd if first_cmd in adm else fuzzy_match_admissible(first_cmd, adm)
            if exec_cmd:
                facts_before = extract_facts_set(game_state)
                game_state_after, env_r, done = tw_env.step(exec_cmd)
                won = getattr(game_state_after, 'won', False)

                if won:
                    reward += 1.5
                    rc["r_win"] = 1.5
                else:
                    # Only NEW facts from this step count toward potential
                    facts_after = extract_facts_set(game_state_after)
                    new_facts = facts_after - facts_before
                    step_phi = compute_potential_from_facts(new_facts, goal_entities)
                    if step_phi > phi_max:
                        potential_reward = step_phi - phi_max
                        reward += potential_reward
                        rc["r_potential"] = potential_reward
                        phi_after = step_phi

                    #  Expert state matching reward 
                    # Compare model's PDDL fact diff with expert's expected diff.
                    # Reward proportional to overlap — works for ALL steps including
                    # navigation (e.g., matching 'at' predicates).
                    if expert_step_new_facts is not None:
                        n_expert = len(expert_step_new_facts)
                        if n_expert == 0 and len(new_facts) == 0:
                            # Both expert and model produced no state change
                            # (e.g., both did "look") → perfect match
                            r_sm = 1.0
                            reward += r_sm
                            rc["r_state_match"] = r_sm
                        elif n_expert > 0:
                            overlap = new_facts & expert_step_new_facts
                            match_ratio = len(overlap) / n_expert
                            # Scale: 0.3 base + 0.7 * ratio → range [0.3, 1.0]
                            if match_ratio > 0:
                                r_sm = 0.3 + 0.7 * match_ratio
                                reward += r_sm
                                rc["r_state_match"] = r_sm
        except Exception as e:
            print(f"    [step exec error] {e}")

    #Token length penalty
    n_tokens = len(completion_text.split())
    if n_tokens > 500:
        len_pen = 0.001 * (n_tokens - 500)
        reward -= len_pen
        rc["pen_length"] = -len_pen

    rc["total"] = reward
    return reward, first_cmd, True, rc, phi_after


def alfworld_rollout_func(prompts, trainer):
    """Per-step N-way GRPO for long-horizon ALFWorld tasks.

    At EACH step of a multi-step trajectory:
      1. Build ONE prompt from current observation (shared by all N generations)
      2. Generate N completions from the SAME prompt → proper GRPO group
      3. Score each completion independently
      4. Pick one action (softmax-weighted by reward) to advance the shared env
      5. Observe new state → next step sees the CONSEQUENCE of the chosen action

    Each step produces one GRPO group (N items with identical prompt).
    GRPOTrainer computes advantage WITHIN each group → valid baseline.

    We must return EXACTLY len(prompts) items (= generation_batch_size).
    generation_batch_size = num_gen * max_horizon, so we need max_horizon groups.
    If a game ends early, we start a new game to fill remaining groups.

    NOTE: `env`, `system_prompt`, `in_conext_prompt_instruction`,
    `in_context_example`, `loaded_expert_traj_data` are all module-level
    globals used by this function.
    """
    tokenizer = trainer.processing_class
    num_gen = trainer.args.num_generations   # N = 8
    total_groups_needed = len(prompts) // num_gen  # = max_horizon (from config)

    batch_prompt_ids = []
    batch_completion_ids = []
    batch_logprobs = []
    batch_precomputed_reward = []
    batch_reward_components = []
    groups_collected = 0

    while groups_collected < total_groups_needed:
        # Start a new game ──
        obs, info = env.reset()
        initial_obs = obs[0]
        admissible_cmds = info["admissible_commands"][0]
        global_goal = extract_global_goal(initial_obs)

        gamefile = None
        if "extra.gamefile" in info:
            gf = info["extra.gamefile"]
            gamefile = gf[0] if isinstance(gf, list) else gf

        #  Entity filter for potential rewards 
        goal_entities = extract_goal_entities(global_goal)
        print(f"  [goal entities] {goal_entities or '{} (no filter)'}")

        #  Expert trajectory 
        match_task_id = find_task_id_by_goal(loaded_expert_traj_data, global_goal)
        expert_cmds = []
        if match_task_id:
            traj = loaded_expert_traj_data[match_task_id]
            if len(traj) > 0 and isinstance(traj[0], list):
                traj = traj[0]
            expert_cmds = [s["Action"] for s in traj if "Action" in s]
        else:
            print(f"  [WARN] no expert match for goal: {global_goal[:80]!r} "
                  f"(gamefile={gamefile}) — falling back to n_expert_total=0 branch")

        #  Strict level-gated curriculum 
        # Model must always be placed `remaining` steps from the goal, where
        # remaining == current curriculum level (with small back-off chance).
        # Promotion only on ≥_CURR_THRESHOLD win rate over _CURR_WINDOW games.
        n_expert_total = len(expert_cmds)
        if n_expert_total <= 1:
            k = 0
            remaining = n_expert_total
            level_played = 1
        else:
            remaining, level_played = sample_curriculum_remaining(n_expert_total)
            k = n_expert_total - remaining

        #  Clamp remaining to available groups 
        # generation_batch_size / num_gen = total_groups_needed.  If previous
        # games consumed some groups, we may not have enough for a full game
        # at the requested level.  Clamp remaining DOWN so the model is placed
        # closer to the goal and can actually win within the available steps.
        groups_left = total_groups_needed - groups_collected
        if n_expert_total > 1 and remaining > groups_left:
            remaining = max(groups_left, 1)
            k = n_expert_total - remaining
            level_played = remaining   # track actual level for curriculum

        if n_expert_total > 1:
            stats = get_curriculum_stats()
            print(f"  [curriculum] level={stats['level']}, "
                  f"window={stats['window_size']}/{_CURR_WINDOW}, "
                  f"window_win_rate={stats['window_win_rate']:.3f}, "
                  f"level_played={level_played}, remaining={remaining}/{n_expert_total}, k={k}")

        # Replay expert prefix to reach state at step k
        current_obs = initial_obs
        current_cmds = admissible_cmds
        action_history = []
        for step_idx in range(k):
            if step_idx < len(expert_cmds):
                action = expert_cmds[step_idx]
                next_obs, _, done, next_cmds, won, _ = env_interact(env, action)
                action_history.append(action)
                current_obs = next_obs
                current_cmds = next_cmds
                if done:
                    break

        # Use actual replay length (might be < k if game ended during replay)
        actual_k = len(action_history)
        expert_prefix = expert_cmds[:actual_k]
        expert_remaining = expert_cmds[actual_k:]
        expert_remaining_states = [expert_command_to_target_state(cmd) for cmd in expert_remaining]
        expert_remaining_states = [s for s in expert_remaining_states if s and s != "Unknown_State"]

        # How many steps to run in THIS game = remaining (already clamped above).
        if n_expert_total <= 1:
            game_budget = 1
        else:
            game_budget = max(remaining, 1)
        steps_for_this_game = min(total_groups_needed - groups_collected, game_budget)

        print(f"\n===== Game (groups {groups_collected+1}-{groups_collected+steps_for_this_game}"
              f"/{total_groups_needed}) =====")
        print(f"Goal: {global_goal[:120]}")
        print(f"  gamefile: {gamefile}")
        print(f"  k={k}/{len(expert_cmds)}, planned steps={steps_for_this_game}")
        print(f"  Expert remaining: {expert_remaining[:5]}")

        #  Create ONE textworld env for env-grounded scoring 
        tw_env = None
        if gamefile:
            try:
                import textworld
                from alfworld.agents.environment.alfred_tw_env import AlfredDemangler
                cleanup_textworld_tmp()
                request_infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True)
                demangler = AlfredDemangler(shuffle=False)
                tw_env = textworld.start(gamefile, request_infos=request_infos, wrappers=[demangler])
            except Exception as e:
                print(f"  [tw env create error] {e}")
                tw_env = None

        #  Pre-compute expert state trajectory (separate game) 
        # Play expert_remaining in a fresh tw_env, record the PDDL fact DIFF
        # at each step.  These become the "target states" for the model —
        # the model gets rewarded for reaching the SAME states (not the same
        # actions), so it can discover alternative action sequences.
        expert_step_new_facts = []   # list of sets, one per expert remaining step
        if tw_env is not None and expert_remaining:
            try:
                # Reset tw_env and replay expert prefix to reach step k
                exp_gs = tw_env.reset()
                for pre_cmd in expert_prefix:
                    exp_gs, _, d = tw_env.step(pre_cmd)
                    if d:
                        break
                # Now execute expert_remaining, recording fact diffs
                for exp_cmd in expert_remaining:
                    adm = getattr(exp_gs, 'admissible_commands', [])
                    matched = fuzzy_match_admissible(exp_cmd, adm)
                    if not matched:
                        break
                    facts_before = extract_facts_set(exp_gs)
                    exp_gs, _, d = tw_env.step(matched)
                    facts_after = extract_facts_set(exp_gs)
                    new_facts = facts_after - facts_before
                    expert_step_new_facts.append(new_facts)
                    if d:
                        break
                print(f"  [expert trajectory] recorded {len(expert_step_new_facts)} state diffs: "
                      f"{[len(f) for f in expert_step_new_facts]}")
            except Exception as e:
                print(f"  [expert trajectory error] {e}")
                expert_step_new_facts = []

        executed_actions = list(action_history)
        cur_obs = current_obs
        cur_cmds = current_cmds
        cur_dag_context = None
        episode_done = False
        game_won = False

        # Potential-based reward tracking: Φ_max = highest milestone achieved
        phi_max = 0.0

        for step in range(steps_for_this_game):
            if episode_done:
                # Game ended early — break and start a new game for remaining groups
                break

        
            # Pre-computed expert_step_new_facts assumes the expert's own prior
            # actions were taken.  If the model deviates from the expert's path, the expert's expected state changes may no longer be accurate.
            if step > 0 and tw_env is not None and step < len(expert_remaining):
                try:
                    recomp_gs = tw_env.reset()
                    for pre_cmd in expert_prefix:
                        recomp_gs, _, d = tw_env.step(pre_cmd)
                        if d:
                            break
                    for ea in executed_actions[len(expert_prefix):]:
                        adm = getattr(recomp_gs, 'admissible_commands', [])
                        matched = fuzzy_match_admissible(ea, adm)
                        if matched:
                            recomp_gs, _, d = tw_env.step(matched)
                            if d:
                                break
                    # Execute remaining expert actions from model's actual state
                    recomputed_facts = []
                    for exp_cmd in expert_remaining[step:]:
                        adm = getattr(recomp_gs, 'admissible_commands', [])
                        matched = fuzzy_match_admissible(exp_cmd, adm)
                        if not matched:
                            break
                        facts_before = extract_facts_set(recomp_gs)
                        recomp_gs, _, d = tw_env.step(matched)
                        facts_after = extract_facts_set(recomp_gs)
                        recomputed_facts.append(facts_after - facts_before)
                        if d:
                            break
                    # Overwrite stale targets with recomputed ones
                    for i, facts in enumerate(recomputed_facts):
                        idx = step + i
                        if idx < len(expert_step_new_facts):
                            expert_step_new_facts[idx] = facts
                        else:
                            expert_step_new_facts.append(facts)
                    print(f"  [re-plan] step {step}: recomputed {len(recomputed_facts)} "
                          f"expert targets from model state: "
                          f"{[len(f) for f in recomputed_facts]}")
                except Exception as e:
                    print(f"  [re-plan error] step {step}: {e}")

            remaining_states = (expert_remaining_states[step:]
                                if step < len(expert_remaining_states) else [])

            # Build ONE prompt for this step (shared by all N gens)
            user_content = f"Task: {global_goal}\n\n"
            user_content += f"Current observation: {cur_obs}\n\n"
            user_content += "Available actions:\n"
            for cmd in cur_cmds:
                user_content += f"- {cmd}\n"
            if executed_actions:
                user_content += "\nActions already taken:\n"
                for j, act in enumerate(executed_actions[-5:]):
                    user_content += f"{j+1}. {act}\n"
            if cur_dag_context:
                compact_dag = json.dumps(cur_dag_context, separators=(',', ':'))
                user_content += f"\nYour previous reasoning DAG:\n{compact_dag}\n"
                user_content += ("Update the DAG based on the NEW observation above: "
                                 "mark achieved goals as 'completed', prune failed paths, "
                                 "then pick the next action.\n")
            else:
                user_content += ("\nOutput your reasoning DAG for the remaining plan, "
                                 "then output the NEXT action in 'Action_Chunks'.\n")

            messages = [
                {"role": "system", "content": (
                    system_prompt + "\n" + in_conext_prompt_instruction + "\n"
                    + "Example:\n" + in_context_example + "\n"
                    + f"Your global goal is: {global_goal}\n"
                )},
                {"role": "user", "content": user_content},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                enable_thinking=False
            )

            # Generate N completions from the SAME prompt
            # Guided decoding: constrain Action_Chunks.Command to the current
            # step's admissible_commands. Falls back to {} (no constraint)
            # when unavailable
            guided_overrides = build_guided_overrides(cur_cmds)
            outputs = generate_rollout_completions(
                trainer,
                [prompt_text] * num_gen,
                generation_overrides=guided_overrides if guided_overrides else None,
            )

            # Helper: replay to current state for env scoring
            def reset_tw_to_current():
                if tw_env is None:
                    return None
                try:
                    gs = tw_env.reset()
                    for pre_cmd in expert_prefix:
                        gs, _, d = tw_env.step(pre_cmd)
                        if d:
                            return gs
                    for exec_cmd in executed_actions[len(expert_prefix):]:
                        adm = getattr(gs, 'admissible_commands', [])
                        matched = fuzzy_match_admissible(exec_cmd, adm)
                        if matched:
                            gs, _, d = tw_env.step(matched)
                            if d:
                                return gs
                    return gs
                except Exception as e:
                    print(f"    [tw reset error] {e}")
                    return None

            # Score each generation
            step_rewards = []
            step_commands = []
            step_valid = []
            step_reward_components = []
            step_phi_afters = []

            for g, out in enumerate(outputs):
                completion_text = out.get("text")
                if completion_text is None:
                    completion_text = tokenizer.decode(out["completion_ids"], skip_special_tokens=True)

                batch_prompt_ids.append(out["prompt_ids"])
                batch_completion_ids.append(out["completion_ids"])
                batch_logprobs.append(out["logprobs"])

                game_state = reset_tw_to_current()
                # Pass expert's expected state diff for this step
                exp_facts_for_step = (expert_step_new_facts[step]
                                      if step < len(expert_step_new_facts) else None)
                reward, command, is_valid, reward_parts, phi_after = score_step_with_potential(
                    completion_text, cur_cmds,
                    game_state, tw_env,
                    recent_actions=executed_actions,
                    phi_max=phi_max,
                    goal_entities=goal_entities,
                    expert_step_new_facts=exp_facts_for_step,
                )
                step_rewards.append(reward)
                step_commands.append(command)
                step_valid.append(is_valid)
                step_reward_components.append(reward_parts)
                step_phi_afters.append(phi_after)
                print(
                    f"    Step {step} Gen {g+1}: cmd='{command}', total={reward:.3f}, "
                    f"fmt_fail={reward_parts['r_format_fail']:.3f}, "
                    f"inadm={reward_parts['r_inadmissible']:.3f}, "
                    f"state_match={reward_parts['r_state_match']:.3f}, "
                    f"potential={reward_parts['r_potential']:.3f}, "
                    f"win={reward_parts['r_win']:.3f}, "
                    f"rep={reward_parts['pen_repeat']:.3f}, "
                    f"noop={reward_parts['pen_noop']:.3f}, "
                    f"len={reward_parts['pen_length']:.3f}, "
                    f"phi_max={phi_max:.2f}, phi_after={phi_after:.2f}"
                )



            batch_precomputed_reward.extend(step_rewards)
            batch_reward_components.extend(step_reward_components)
            groups_collected += 1

            #  Component-level diagnostics and tracking 
            component_keys = [
                "r_format_fail", "r_inadmissible", "r_state_match", "r_potential",
                "r_win", "pen_repeat", "pen_noop", "pen_length", "total"
            ]
            comp_mean = {
                k: float(np.mean([c.get(k, 0.0) for c in step_reward_components]))
                for k in component_keys
            }
            print(
                f"  [reward parts] step={step}, group={groups_collected-1}, "
                f"total={comp_mean['total']:.3f}, "
                f"fmt_fail={comp_mean['r_format_fail']:.3f}, "
                f"inadm={comp_mean['r_inadmissible']:.3f}, "
                f"state_match={comp_mean['r_state_match']:.3f}, "
                f"potential={comp_mean['r_potential']:.3f}, "
                f"win={comp_mean['r_win']:.3f}, "
                f"rep={comp_mean['pen_repeat']:.3f}, "
                f"noop={comp_mean['pen_noop']:.3f}, "
                f"len={comp_mean['pen_length']:.3f}, "
                f"phi_max={phi_max:.2f}"
            )
            try:
                exp = comet_ml.get_global_experiment()
                if exp is not None:
                    exp.log_metrics(
                        {f"reward_parts/{k}": v for k, v in comp_mean.items()},
                        step=groups_collected,
                    )
            except Exception:
                pass

            #  Pick ONE action to advance the shared env(actual env )
            valid_indices = [j for j in range(num_gen)
                             if step_valid[j] and step_commands[j]]
            if not valid_indices:
                print(f"  Step {step}: no valid actions, ending this game")
                episode_done = True
                continue

            # Greedy selection: pick the highest-reward valid action to advance the shared env
            valid_rewards = np.array([step_rewards[j] for j in valid_indices])
            max_r = valid_rewards.max()
            best_indices = [valid_indices[j] for j in range(len(valid_indices))
                            if valid_rewards[j] >= max_r - 1e-6]
            chosen_idx = best_indices[np.random.randint(len(best_indices))]
            chosen_cmd = step_commands[chosen_idx]

            print(f"  Step {step}: chose '{chosen_cmd}' "
                  f"(gen {chosen_idx+1}, r={step_rewards[chosen_idx]:.3f})")

            #  Execute in shared env (actual env)
            next_obs, env_r, done, next_cmds, won, _ = env_interact(env, chosen_cmd)
            executed_actions.append(chosen_cmd)

            # Update potential tracking:
            chosen_phi = step_phi_afters[chosen_idx]
            phi_max = max(phi_max, chosen_phi)

            if won:
                # R_win is already captured per-completion inside

                print(f"  Step {step}: TASK WON!")
                game_won = True
                episode_done = True

            # Extract DAG from chosen completion for next step
            chosen_text = outputs[chosen_idx].get("text")
            if chosen_text is None:
                chosen_text = tokenizer.decode(
                    outputs[chosen_idx]["completion_ids"], skip_special_tokens=True)
            dag_mgr = DAGManager()
            dag_mgr.extract_and_update_DAG(chosen_text)
            cur_dag_context = dag_mgr.get_current_DAG()

            cur_obs = next_obs
            cur_cmds = next_cmds

            if done and not won:
                episode_done = True

        #  Update curriculum (strict level-gated)
        new_level = update_curriculum(game_won, level_played)
        stats = get_curriculum_stats()
        print(f"  [curriculum update] game_won={game_won}, level_played={level_played}, "
              f"current_level={new_level}, "
              f"window={stats['window_size']}/{_CURR_WINDOW}, "
              f"window_win_rate={stats['window_win_rate']:.3f}, "
              f"promotions={stats['promotions']}")

        #  Cleanup textworld
        if tw_env:
            try:
                tw_env.close()
            except Exception:
                pass
            cleanup_textworld_tmp()

    #  Diagnostics for pre-computed rewards
    for g_idx in range(total_groups_needed):
        group_rewards = batch_precomputed_reward[g_idx * num_gen : (g_idx + 1) * num_gen]
        if group_rewards:
            print(f"[GRPO diag] Group {g_idx}: rewards={[f'{r:.3f}' for r in group_rewards]}, "
                  f"mean={np.mean(group_rewards):.3f}, std={np.std(group_rewards):.4f}")

    return {
        "prompt_ids": batch_prompt_ids,
        "completion_ids": batch_completion_ids,
        "logprobs": batch_logprobs,
        "precomputed_reward": batch_precomputed_reward,
    }
def direct_step_reward(completions, precomputed_reward=None, **kwargs):
    if precomputed_reward is None:
        return [0.0] * len(completions)
    return [float(r) for r in precomputed_reward]



#actual traning 
MERGED_MODEL = "./Qwen3-8B-alfworld-sft-dag-merged"

GRPO_OUT = "./Qwen3-8B-alfworld-grpo"


def has_trainer_state(path_str):
    return Path(path_str, "trainer_state.json").exists()


def has_adapter_weights(path_str):
    path = Path(path_str)
    return (
        path.joinpath("adapter_config.json").exists()
        and (
            path.joinpath("adapter_model.safetensors").exists()
            or path.joinpath("adapter_model.bin").exists()
        )
    )

def resolve_resume_checkpoint(grpo_out):
    # Fresh start by default. Set GRPO_RESUME=1 to re-enable checkpoint scanning.
    if os.environ.get("GRPO_RESUME", "").strip() != "1":
        print("[qwen3-8b-grpo] Fresh start (set GRPO_RESUME=1 to resume from checkpoint).")
        return None
    env_ckpt = os.environ.get("GRPO_RESUME_FROM", "").strip()
    if env_ckpt:
        return env_ckpt if has_trainer_state(env_ckpt) else None
    out_dir = Path(grpo_out)
    if not out_dir.exists():
        return None
    checkpoints = []
    for p in out_dir.glob("checkpoint-*"):
        m = re.match(r"checkpoint-(\d+)$", p.name)
        if m and has_trainer_state(str(p)):
            checkpoints.append((int(m.group(1)), p))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


def resolve_model_source(grpo_out, merged_model):
    env_model = os.environ.get("GRPO_LOAD_MODEL_FROM", "").strip()
    if env_model:
        return env_model
    if has_adapter_weights(grpo_out):
        return grpo_out
    return merged_model


def build_training_model(model_source, merged_model, peft_config):
    if has_adapter_weights(model_source):
        print(f"[qwen3-8b-grpo] Loading adapter model from {model_source} over base {merged_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            merged_model,
            torch_dtype="auto",
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_source,
            is_trainable=True,
        )
        return model, None
    print(f"[qwen3-8b-grpo] Loading model source: {model_source}")
    return model_source, peft_config


RESUME_CHECKPOINT = resolve_resume_checkpoint(GRPO_OUT)
MODEL_SOURCE = resolve_model_source(GRPO_OUT, MERGED_MODEL)

DEFAULT_RESUME_TOTAL_STEPS = int(os.environ.get("GRPO_RESUME_TOTAL_STEPS", "4500"))
DEFAULT_ADAPTER_WARMSTART_STEPS = int(os.environ.get("GRPO_ADAPTER_WARMSTART_STEPS", "1350"))

if RESUME_CHECKPOINT:
    TARGET_TOTAL_STEPS = DEFAULT_RESUME_TOTAL_STEPS
    print(f"[qwen3-8b-grpo] Trainer checkpoint mode: total steps set to {TARGET_TOTAL_STEPS}")
else:
    TARGET_TOTAL_STEPS = DEFAULT_ADAPTER_WARMSTART_STEPS
    print(f"[qwen3-8b-grpo] Adapter warm-start mode: total steps set to {TARGET_TOTAL_STEPS}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE)
#add pad
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

#dummy dataset, not actually used for training, since we are using the rollout function to generate data on the fly, but it is required by the trainer
train_dataset = Dataset.from_dict({
    "prompt": ["alfworld"] * 4000     # must be >= max possible TARGET_TOTAL_STEPS
})


peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=16,          # scaling = alpha/r = 1.0 (safe default for GRPO)
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj",
    ],
)



grpo_args = GRPOConfig(
    output_dir=GRPO_OUT,
    use_vllm=True,
    vllm_mode="colocate",   
    learning_rate=1e-5,             
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=10,  # = generation_batch_size / num_gen: must match exactly
    num_generations=8,
    generation_batch_size=80,        # = num_gen(8) × 10: supports curriculum level up to 10
    max_steps=TARGET_TOTAL_STEPS,
    max_completion_length=1024,      #  prevent truncation → invalid JSON → zero-variance groups
    lr_scheduler_type="cosine",      # cosine decay for stable late-training
    warmup_steps=100,                # ~7% warmup (1350 total steps)
    max_grad_norm=1.0,
    logging_steps=1,
    save_steps=40,
    save_total_limit=2,              # only keep 2 most recent checkpoints (prevent disk quota exceeded)
    bf16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to = ["comet_ml"],
    run_name = "Qwen3-8B-alfworld-T27",
    beta=0.04,                       #  entropy collapsed to 0.12 in T26. Higher beta

    vllm_importance_sampling_correction=False,  # guided decoding now disabled; keep False for stability.
    temperature=1.5,                 # keep 1.5: higher causes too many invalid JSON outputs
    top_p=0.95,                      # slightly wider nucleus for more command diversity
)
trainer_model, trainer_peft_config = build_training_model(MODEL_SOURCE, MERGED_MODEL, peft_config)
trainer = GRPOTrainer(
    model=trainer_model,
    processing_class=tokenizer,
    reward_funcs=[direct_step_reward],
    train_dataset=train_dataset,
    args=grpo_args,
    rollout_func=alfworld_rollout_func,
    peft_config=trainer_peft_config,

)


if RESUME_CHECKPOINT and Path(RESUME_CHECKPOINT).exists():
    print(f"[qwen3-8b-grpo] Resuming from checkpoint: {RESUME_CHECKPOINT}")
    trainer.train(resume_from_checkpoint=RESUME_CHECKPOINT)
else:
    print(f"[qwen3-8b-grpo] No trainer checkpoint found, training from loaded model source: {MODEL_SOURCE}")
    trainer.train()
trainer.save_model(GRPO_OUT)
tokenizer.save_pretrained(GRPO_OUT)