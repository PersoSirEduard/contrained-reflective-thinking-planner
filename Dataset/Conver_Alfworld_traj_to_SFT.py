import json
import os
import re

# System prompt must match RL.py exactly so the SFT model learns the expected format
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

# map the action to subgoal by using regex, e.g. "go to dresser 1" -> "reach dresser 1", "take apple from fridge" -> "hold apple"
def action_to_subgoal(action: str) -> str:

    action = re.sub(r"\s+", " ", action).strip()
    if action == "look":
        return "inspect current scene"
    m = re.match(r"^go to (.+)$", action)
    if m:
        return f"reach {m.group(1)}"
    m = re.match(r"^open (.+)$", action)
    if m:
        return f"access {m.group(1)}"
    m = re.match(r"^close (.+)$", action)
    if m:
        return f"finish with {m.group(1)}"
    m = re.match(r"^take (.+) from (.+)$", action)
    if m:
        return f"hold {m.group(1)}"
    m = re.match(r"^put (.+) (?:on|in) (.+)$", action)
    if m:
        return f"place {m.group(1)} at {m.group(2)}"
    m = re.match(r"^clean (.+) with (.+)$", action)
    if m:
        return f"clean {m.group(1)}"
    m = re.match(r"^heat (.+) with (.+)$", action)
    if m:
        return f"heat {m.group(1)}"
    m = re.match(r"^cool (.+) with (.+)$", action)
    if m:
        return f"cool {m.group(1)}"
    m = re.match(r"^slice (.+) with (.+)$", action)
    if m:
        return f"slice {m.group(1)}"
    m = re.match(r"^(?:use|toggle) (.+)$", action)
    if m:
        return f"activate {m.group(1)}"
    return f"execute {action}"

#Covert action to target state, e.g. "go to dresser 1" -> "Agent_At(dresser 1)", "take apple from fridge" -> "Agent_Holding(apple)"
def action_to_target_state(action: str) -> str:
    cmd = re.sub(r"\s+", " ", action).strip().lower()
    if cmd in ("look", "inventory"):
        return "Observation_Update"
    m = re.match(r"^go to (.+)$", cmd)
    if m:
        return f"Agent_At({m.group(1)})"
    m = re.match(r"^take (.+) from (.+)$", cmd)
    if m:
        return f"Agent_Holding({m.group(1)})"
    m = re.match(r"^put (.+) (?:on|in) (.+)$", cmd)
    if m:
        return f"Object_Location({m.group(1)}, {m.group(2)})"
    m = re.match(r"^open (.+)$", cmd)
    if m:
        return f"Receptacle_Opened({m.group(1)})"
    m = re.match(r"^close (.+)$", cmd)
    if m:
        return f"Receptacle_Closed({m.group(1)})"
    m = re.match(r"^heat (.+) with (.+)$", cmd)
    if m:
        return f"Object_Property({m.group(1)}, Hot)"
    m = re.match(r"^cool (.+) with (.+)$", cmd)
    if m:
        return f"Object_Property({m.group(1)}, Cool)"
    m = re.match(r"^clean (.+) with (.+)$", cmd)
    if m:
        return f"Object_Property({m.group(1)}, Clean)"
    m = re.match(r"^slice (.+) with (.+)$", cmd)
    if m:
        return f"Object_Property({m.group(1)}, Sliced)"
    m = re.match(r"^(?:use|toggle) (.+)$", cmd)
    if m:
        return f"Object_Property({m.group(1)}, Toggled)"
    return "Unknown_State"

#covert action to DAG edge, e.g. "go to dresser 1" -> "GOTO(dresser 1)", "take apple from fridge" -> "TAKE(apple, fridge)"
def action_to_dag_edge(action: str) -> str:

    cmd = re.sub(r"\s+", " ", action).strip().lower()
    if cmd in ("look", "inventory"):
        return "OBSERVE()"
    m = re.match(r"^go to (.+)$", cmd)
    if m:
        return f"GOTO({m.group(1)})"
    m = re.match(r"^take (.+) from (.+)$", cmd)
    if m:
        return f"TAKE({m.group(1)}, {m.group(2)})"
    m = re.match(r"^put (.+) (?:on|in) (.+)$", cmd)
    if m:
        return f"PUT({m.group(1)}, {m.group(2)})"
    m = re.match(r"^open (.+)$", cmd)
    if m:
        return f"OPEN({m.group(1)})"
    m = re.match(r"^close (.+)$", cmd)
    if m:
        return f"CLOSE({m.group(1)})"
    m = re.match(r"^heat (.+) with (.+)$", cmd)
    if m:
        return f"HEAT({m.group(1)}, {m.group(2)})"
    m = re.match(r"^cool (.+) with (.+)$", cmd)
    if m:
        return f"COOL({m.group(1)}, {m.group(2)})"
    m = re.match(r"^clean (.+) with (.+)$", cmd)
    if m:
        return f"CLEAN({m.group(1)}, {m.group(2)})"
    m = re.match(r"^slice (.+) with (.+)$", cmd)
    if m:
        return f"SLICE({m.group(1)}, {m.group(2)})"
    m = re.match(r"^(?:use|toggle) (.+)$", cmd)
    if m:
        return f"TOGGLE({m.group(1)})"
    return f"EXECUTE({action})"


def build_dag_for_step(all_actions: list, step_idx: int,
                       window_back: int = 2, window_fwd: int = 4) -> dict:
   # Build a DAG representation for the current step, showing a window of actions around the current step.
    n = len(all_actions)
    show_start = max(0, step_idx - window_back)
    show_end   = min(n, step_idx + window_fwd + 1)

    nodes = {}
    for j in range(show_start, show_end):
        state  = action_to_target_state(all_actions[j])
        status = "completed" if j < step_idx else "pending"
        nodes[f"S{j}"] = [state, status]

    # Edges: Only connect sequential actions in the shown window, and label edges with the corresponding action tags. This keeps the DAG simple and focused on the immediate plan.
    edges = []
    shown = list(range(show_start, show_end))
    for k in range(len(shown) - 1):
        src_j = shown[k]
        dst_j = shown[k + 1]
        edge_tag = action_to_dag_edge(all_actions[dst_j])
        edges.append({"from": f"S{src_j}", "to": f"S{dst_j}", "action": edge_tag})

    target_goal  = f"S{step_idx}"
    current_state = action_to_target_state(all_actions[step_idx])
    checker = (
        f"Target is {target_goal}: achieve {current_state}. "
        f"Execute the corresponding admissible command to progress."
    )

    return {
        "nodes": nodes,
        "edges": edges,
        "Target_Goal": target_goal,
        "Checker": checker,
    }


def build_reflection(main_goal: str, all_actions: list, step_idx: int) -> str:
    #Generate a reflection string that summarizes the current progress and next target in a human-readable format. This helps the model understand the context and reasoning behind the DAG structure.
    current_state = action_to_target_state(all_actions[step_idx])
    if step_idx == 0:
        return f"Starting task: {main_goal}. First I need to achieve: {current_state}."
    prev_state = action_to_target_state(all_actions[step_idx - 1])
    return (
        f"Task: {main_goal}. "
        f"Just achieved: {prev_state}. "
        f"Next target: {current_state}."
    )


def build_user_content(task: str, observation: str, recent_actions: list) -> dict:
    return {
        "role": "user",
        "content": (
            f"Task: {task}\n"
            f"Current Observation: {observation}\n"
            f"Recent Actions: {', '.join(recent_actions)}"
        ),
    }


def Convert_Alfworld_traj_to_SFT(expert_traj_path, output_file_name):
    expert_traj_data = {}
    with open(expert_traj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            expert_traj_data.update(obj)

    converted_data = []
    main_task_re = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)

    for traj_id, traj in expert_traj_data.items():
        if not traj or "Initial_Observation" not in traj[0]:
            continue

        initial_observation = traj[0]["Initial_Observation"]
        match = main_task_re.search(initial_observation)
        if not match:
            raise ValueError(
                f"Could not extract main goal from initial observation: {initial_observation}"
            )
        main_goal = match.group(1).strip()

        # Pre-extract the complete ordered action list for DAG generation
        all_actions = [
            re.sub(r"\s+", " ", traj[j]["Action"]).strip()
            for j in range(1, len(traj), 2)
            if "Action" in traj[j]
        ]

        i = 1                        # trajectory index (odd = Action, even = Observation)
        current_observation = initial_observation

        while i + 1 < len(traj):
            action = re.sub(r"\s+", " ", traj[i]["Action"]).strip()
            next_observation = traj[i + 1]["Observation"]

            # Index of this action inside all_actions
            step_idx = (i - 1) // 2

            recent_actions = [
                re.sub(r"\s+", " ", traj[j]["Action"]).strip()
                for j in range(max(1, i - 6), i, 2)
                if "Action" in traj[j]
            ]

            user_content = build_user_content(main_goal, current_observation, recent_actions)

            subgoal    = action_to_subgoal(action)
            dag        = build_dag_for_step(all_actions, step_idx)
            reflection = build_reflection(main_goal, all_actions, step_idx)

            assistant_content = {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "Reflection": reflection,
                        "DAG": dag,
                        "Action_Chunks": {
                            "Subgoal": subgoal,
                            "Command": action,
                        },
                    },
                    ensure_ascii=False,
                ),
            }

            converted_data.append(
                {
                    "traj_id": traj_id,
                    "step": step_idx,
                    "task": main_goal,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        user_content,
                        assistant_content,
                    ],
                }
            )

            current_observation = next_observation
            i += 2

    with open(output_file_name, "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

    print(f"Converted {len(converted_data)} steps to {output_file_name}")
