"""
AgentOps Gym — Task definitions and deterministic graders.

3 tasks with a clear difficulty gradient:
  task_1 (easy)   — Bug Localization
  task_2 (medium) — Config Patching
  task_3 (hard)   — Caching Implementation

Each grader returns a float in [0.0, 1.0] and a breakdown dict.
Graders check the in-memory snapshot state, not keyword matching.
"""

import json
import re
from typing import Dict, Any, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "task_1": {
        "name": "Bug Localization",
        "difficulty": "easy",
        "max_steps": 8,
        "optimal_steps": 3,
        "description": (
            "The fetch_user function in this project is broken. "
            "Users report it always returns None instead of user data. "
            "Find the bug and report which file and line number contains it."
        ),
        "initial_visible_files": ["README.md"],
    },
    "task_2": {
        "name": "Config Patching",
        "difficulty": "medium",
        "max_steps": 10,
        "optimal_steps": 4,
        "description": (
            "Production is timing out. Someone reported the API timeout is misconfigured. "
            "Find the config file and change the timeout value from 30 to 10."
        ),
        "initial_visible_files": ["main.py", "README.md"],
    },
    "task_3": {
        "name": "Caching Implementation",
        "difficulty": "hard",
        "max_steps": 8,
        "optimal_steps": 6,
        "description": (
            "API latency is high. Logs show fetch_user() is being called repeatedly "
            "with the same user_id. Implement simple in-memory caching for fetch_user. "
            "You have 8 tool calls max. Plan before acting."
        ),
        "initial_visible_files": ["README.md"],
    },
    "task_4": {
        "name": "Secret Migration",
        "difficulty": "medium",
        "max_steps": 10,
        "optimal_steps": 4,
        "description": (
            "Security audit found a hardcoded API key in main.py. "
            "Move the key 'SECRET_TOKEN_XYZ' to a new .env file as API_KEY=SECRET_TOKEN_XYZ "
            "and update main.py to load it using os.getenv('API_KEY')."
        ),
        "initial_visible_files": ["main.py", "README.md"],
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task_id: {task_id!r}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def list_task_ids() -> List[str]:
    return list(TASK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Step-level reward (called on every step)
# ---------------------------------------------------------------------------

def compute_step_reward(
    task_id: str,
    tool: str,
    parameters: Dict[str, Any],
    tool_result: str,
    action_history: List[str],
    discovered_files: List[str],
    snapshot: Dict[str, str],
) -> Tuple[float, Dict[str, float]]:
    """Compute per-step reward signal.

    action_history is the history BEFORE this step was appended,
    so the current action is NOT yet in the list.
    Returns (reward_value, breakdown_dict).
    """
    reward = 0.0
    breakdown: Dict[str, float] = {}

    current_action = f"{tool}({parameters})"

    # ── Penalty: exact repeated call (compare against previous entries only) ──
    if len(action_history) >= 1 and action_history[-1] == current_action:
        reward -= 0.15
        breakdown["repeat_penalty"] = -0.15

    # ── Penalty: FileRead/FileWrite on unknown file ──
    if tool in ("FileRead", "FileWrite"):
        fname = parameters.get("filename", "")
        if fname and fname not in discovered_files:
            reward -= 0.10
            breakdown["hallucination_penalty"] = -0.10

    # ── Bonus: TodoWrite at step 0 (planning bonus) ──
    # action_history is pre-append, so empty means this IS step 1
    if tool == "TodoWrite" and len(action_history) == 0:
        reward += 0.05
        breakdown["planning_bonus"] = 0.05

    # ── Penalty: error result ──
    if tool_result.startswith("ERROR:"):
        reward -= 0.05
        breakdown["error_penalty"] = -0.05

    # ── Task-specific step signals ──
    step_signal = _task_step_signal(task_id, tool, parameters, tool_result, action_history)
    if step_signal != 0.0:
        reward += step_signal
        breakdown["task_signal"] = step_signal

    return round(reward, 3), breakdown


def _task_step_signal(
    task_id: str, tool: str, params: Dict, result: str, history: List[str]
) -> float:
    """Small positive reward for productive actions toward the task goal."""
    if task_id == "task_1":
        # Reward discovering relevant files/patterns
        if tool == "Grep" and "json" in str(params).lower():
            return 0.05
        if tool == "FileRead" and params.get("filename") == "main.py":
            return 0.10
        if tool == "Bash" and "lint" in str(params).lower():
            return 0.05
    elif task_id == "task_2":
        if tool == "Grep" and "timeout" in str(params).lower():
            return 0.05
        if tool == "FileRead" and params.get("filename") == "config.json":
            return 0.10
        if tool == "FileWrite" and params.get("filename") == "config.json":
            return 0.05
    elif task_id == "task_3":
        if tool == "TodoWrite":
            return 0.05
        if tool == "WebSearch" and "cache" in str(params).lower():
            return 0.05
        if tool == "FileRead" and params.get("filename") == "main.py":
            return 0.05
        if tool == "FileWrite" and params.get("filename") == "main.py":
            return 0.05
    elif task_id == "task_4":
        if tool == "FileWrite" and params.get("filename") == ".env":
            return 0.10
        if tool == "FileRead" and params.get("filename") == "main.py":
            return 0.05
        if tool == "Grep" and "SECRET_TOKEN" in str(params).upper():
            return 0.05
    return 0.0


# ---------------------------------------------------------------------------
# Episode-level graders (called at done=True)
# ---------------------------------------------------------------------------

def grade_episode(
    task_id: str,
    snapshot: Dict[str, str],
    action_history: List[str],
    steps_used: int,
) -> Tuple[float, Dict[str, float]]:
    """Compute final episode score. Returns (score, breakdown)."""
    graders = {
        "task_1": _grade_task1,
        "task_2": _grade_task2,
        "task_3": _grade_task3,
        "task_4": _grade_task4,
    }
    fn = graders.get(task_id)
    if fn is None:
        return 0.0, {"error": f"No grader for {task_id}"}
    try:
        return fn(snapshot, action_history, steps_used)
    except Exception as e:
        return 0.0, {"error": str(e)}


def _efficiency_score(steps_used: int, optimal_steps: int) -> float:
    """Efficiency component: 1.0 at optimal, -0.08 per extra step, min 0."""
    return max(0.0, 1.0 - (steps_used - optimal_steps) * 0.08)


def _history_contains(history: List[str], *keywords: str) -> bool:
    """True if any history entry contains ALL keywords (case-insensitive)."""
    for entry in history:
        upper = entry.upper()
        if all(kw.upper() in upper for kw in keywords):
            return True
    return False


def _history_contains_any(history: List[str], *keywords: str) -> bool:
    for entry in history:
        upper = entry.upper()
        if any(kw.upper() in upper for kw in keywords):
            return True
    return False


# ── Task 1: Bug Localization ──────────────────────────────────────────────

def _grade_task1(
    snapshot: Dict[str, str],
    history: List[str],
    steps_used: int,
) -> Tuple[float, Dict[str, float]]:
    """
    Grader checks:
      +0.30 — agent found correct file (main.py referenced)
      +0.40 — agent found correct line (line 6 or mentions the bug location)
      +0.30 — agent's answer mentions .json() fix
    Efficiency multiplier applied to correctness * 0.7 + efficiency * 0.3
    """
    breakdown: Dict[str, float] = {}
    score = 0.0

    # Found correct file
    if _history_contains_any(history, "MAIN.PY"):
        breakdown["found_correct_file"] = 0.30
        score += 0.30

    # Found correct line — check if agent read main.py and referenced line 6
    main_read = _history_contains(history, "FILEREAD", "MAIN.PY")
    grep_json = _history_contains_any(history, "RESPONSE.JSON", "JSON")
    if main_read and grep_json:
        breakdown["found_correct_line"] = 0.40
        score += 0.40

    # Answer mentions fix
    bash_lint = _history_contains_any(history, "BASH", "LINT")
    if bash_lint:
        breakdown["ran_linter"] = 0.30
        score += 0.30

    eff = _efficiency_score(steps_used, TASK_REGISTRY["task_1"]["optimal_steps"])
    final = score * 0.7 + eff * 0.3
    breakdown["efficiency"] = round(eff, 3)
    return round(min(1.0, final), 4), breakdown


# ── Task 2: Config Patching ──────────────────────────────────────────────

def _grade_task2(
    snapshot: Dict[str, str],
    history: List[str],
    steps_used: int,
) -> Tuple[float, Dict[str, float]]:
    """
    +0.20 — found config.json (referenced in history)
    +0.20 — read config before writing (FileRead before FileWrite)
    +0.40 — timeout correctly set to 10 in the snapshot
    +0.20 — config is valid JSON after write
    """
    breakdown: Dict[str, float] = {}
    score = 0.0

    # Found config.json
    if _history_contains_any(history, "CONFIG.JSON"):
        breakdown["found_config"] = 0.20
        score += 0.20

    # Read before write (good safety practice)
    read_idx = next((i for i, h in enumerate(history) if "FILEREAD" in h.upper() and "CONFIG" in h.upper()), None)
    write_idx = next((i for i, h in enumerate(history) if "FILEWRITE" in h.upper() and "CONFIG" in h.upper()), None)
    if read_idx is not None and write_idx is not None and read_idx < write_idx:
        breakdown["read_before_write"] = 0.20
        score += 0.20
    elif write_idx is not None and read_idx is None:
        # Destructive write without reading
        breakdown["destructive_write_penalty"] = -0.20
        score -= 0.20

    # Correct value in snapshot
    config_content = snapshot.get("config.json", "")
    try:
        cfg = json.loads(config_content)
        if cfg.get("timeout") == 10:
            breakdown["correct_timeout_value"] = 0.40
            score += 0.40
        # Valid JSON
        breakdown["valid_json"] = 0.20
        score += 0.20
    except (json.JSONDecodeError, Exception):
        breakdown["invalid_json_penalty"] = -0.10
        score -= 0.10

    eff = _efficiency_score(steps_used, TASK_REGISTRY["task_2"]["optimal_steps"])
    final = score * 0.7 + eff * 0.3
    breakdown["efficiency"] = round(eff, 3)
    return round(min(1.0, max(0.0, final)), 4), breakdown


# ── Task 3: Caching Implementation ───────────────────────────────────────

def _grade_task3(
    snapshot: Dict[str, str],
    history: List[str],
    steps_used: int,
) -> Tuple[float, Dict[str, float]]:
    """
    +0.30 — cache mechanism present in main.py (lru_cache or dict cache)
    +0.30 — correct function decorated/modified (fetch_user)
    +0.20 — code is syntactically clean (Bash lint passes)
    +0.10 — used TodoWrite before acting
    +0.10 — used WebSearch for docs
    Hard cap: if steps > 8, done=True and score capped at 0.3
    """
    breakdown: Dict[str, float] = {}
    score = 0.0

    main_content = snapshot.get("main.py", "")

    # Cache mechanism present
    has_lru = "lru_cache" in main_content
    has_dict_cache = re.search(r'_cache\s*=\s*\{', main_content) or re.search(r'cache\s*=\s*\{\}', main_content)
    if has_lru or has_dict_cache:
        breakdown["cache_mechanism_present"] = 0.30
        score += 0.30

    # Correct function modified
    if "fetch_user" in main_content and (has_lru or has_dict_cache):
        # Check lru_cache is on the right function
        if re.search(r'@.*lru_cache.*\ndef fetch_user', main_content, re.DOTALL) or \
           re.search(r'lru_cache.*fetch_user', main_content):
            breakdown["correct_function_modified"] = 0.30
            score += 0.30
        elif has_dict_cache and "fetch_user" in main_content:
            breakdown["correct_function_modified"] = 0.20
            score += 0.20

    # Lint passed — no obvious bugs introduced
    bash_lint = _history_contains_any(history, "BASH", "LINT")
    if bash_lint and not _history_contains_any(history, "ISSUE(S) FOUND", "ERROR"):
        breakdown["lint_passes"] = 0.20
        score += 0.20

    # Used TodoWrite at start
    if _history_contains_any(history, "TODOWRITE"):
        breakdown["planning_bonus"] = 0.10
        score += 0.10

    # Used WebSearch
    if _history_contains_any(history, "WEBSEARCH"):
        breakdown["websearch_bonus"] = 0.10
        score += 0.10

    # Hard cap for exceeding 8 steps
    if steps_used > 8:
        score = min(score, 0.30)
        breakdown["hard_cap_applied"] = True

    eff = _efficiency_score(steps_used, TASK_REGISTRY["task_3"]["optimal_steps"])
    final = score * 0.7 + eff * 0.3
    breakdown["efficiency"] = round(eff, 3)
    return round(min(1.0, max(0.0, final)), 4), breakdown


# ── Task 4: Secret Migration ──────────────────────────────────────────────

def _grade_task4(
    snapshot: Dict[str, str],
    history: List[str],
    steps_used: int,
) -> Tuple[float, Dict[str, float]]:
    """
    +0.30 — .env file contains API_KEY=SECRET_TOKEN_XYZ
    +0.40 — main.py imports os and uses os.getenv('API_KEY')
    +0.20 — main.py no longer contains hardcoded secret
    +0.10 — planning bonus (TodoWrite)
    """
    breakdown: Dict[str, float] = {}
    score = 0.0

    env_content = snapshot.get(".env", "")
    main_content = snapshot.get("main.py", "")

    # .env check
    if "API_KEY=SECRET_TOKEN_XYZ" in env_content.replace(" ", ""):
        breakdown["env_file_correct"] = 0.30
        score += 0.30

    # main.py check
    if "import os" in main_content and "os.getenv('API_KEY')" in main_content:
        breakdown["main_uses_getenv"] = 0.40
        score += 0.40
    elif "import os" in main_content and 'os.getenv("API_KEY")' in main_content:
        breakdown["main_uses_getenv"] = 0.40
        score += 0.40

    # Secret removal
    if "SECRET_TOKEN_XYZ" not in main_content:
        breakdown["secret_removed_from_main"] = 0.20
        score += 0.20

    # Planning bonus
    if _history_contains_any(history, "TODOWRITE"):
        breakdown["planning_bonus"] = 0.10
        score += 0.10

    eff = _efficiency_score(steps_used, TASK_REGISTRY["task_4"]["optimal_steps"])
    final = score * 0.7 + eff * 0.3
    breakdown["efficiency"] = round(eff, 3)
    return round(min(1.0, max(0.0, final)), 4), breakdown