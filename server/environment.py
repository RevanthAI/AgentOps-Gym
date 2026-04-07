"""
AgentOps Gym — Core Environment class.

Implements the OpenEnv Environment interface: reset(), step(), state.
Orchestrates tool execution, reward shaping, and episode grading.

Each episode is fully deterministic given a task_id:
  - Snapshot is restored from PROJECT_SNAPSHOTS on reset
  - All tool calls operate on the in-memory snapshot
  - No real filesystem, no real subprocess
"""

import copy
import logging
import uuid
from typing import Optional, Any

from openenv.core.env_server.interfaces import Environment

from agentops_gym.models import ToolCall, AgentObservation, AgentState
from agentops_gym.server.tools import run_tool, PROJECT_SNAPSHOTS, AVAILABLE_TOOLS
from agentops_gym.server.tasks import (
    TASK_REGISTRY,
    get_task,
    list_task_ids,
    compute_step_reward,
    grade_episode,
)

logger = logging.getLogger(__name__)

_last_grader_result: Optional[dict] = None


class AgentOpsEnvironment(Environment[ToolCall, AgentObservation, AgentState]):
    """Tool-use efficiency training environment.

    Each episode:
    1. reset() selects a task, initialises the in-memory snapshot, returns initial obs
    2. step() executes a tool call, computes reward, checks completion
    3. state property returns current episode metadata
    """

    def __init__(self):
        super().__init__()
        self._episode_id: str = ""
        self._task_id: str = ""
        self._task: dict = {}
        self._snapshot: dict = {}
        self._visible_files: list = []
        self._discovered_files: list = []
        self._action_history: list = []
        self._step_count: int = 0
        self._max_steps: int = 10
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._grader_score: Optional[float] = None


    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentObservation:
        """Start a new episode.

        kwargs may include 'task_id' to select a specific task.
        If not given, defaults to task_1 (can be cycled externally).
        """
        task_id = kwargs.get("task_id", "task_1")
        if task_id not in TASK_REGISTRY:
            task_id = "task_1"

        self._episode_id = episode_id or str(uuid.uuid4())
        self._task_id = task_id
        self._task = get_task(task_id)
        self._max_steps = self._task["max_steps"]

        self._snapshot = copy.deepcopy(PROJECT_SNAPSHOTS.get(task_id, {}))

        self._visible_files = list(self._task["initial_visible_files"])
        self._discovered_files = list(self._visible_files)

        self._action_history = []
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._grader_score = None

        logger.info("Episode %s started: task=%s", self._episode_id, task_id)

        return AgentObservation(
            visible_files=list(self._visible_files),
            last_tool_result=None,
            action_history=[],
            step_count=0,
            task_description=self._task["description"],
            message=f"Episode started. Available tools: {', '.join(AVAILABLE_TOOLS.keys())}",
            done=False,
            reward=0.0,
            metadata={
                "task_id": task_id,
                "difficulty": self._task["difficulty"],
                "max_steps": self._max_steps,
                "available_tools": list(AVAILABLE_TOOLS.keys()),
            },
        )

    def step(
        self,
        action: ToolCall,
        **kwargs: Any,
    ) -> AgentObservation:
        """Execute one tool call and return updated observation."""
        if self._done:
            return self._terminal_obs("Episode already done. Call reset() first.")

        self._step_count += 1
        tool = action.tool
        params = action.parameters

        tool_result, self._snapshot, self._discovered_files = run_tool(
            tool=tool,
            parameters=params,
            snapshot=self._snapshot,
            discovered_files=self._discovered_files,
        )

        history_before = list(self._action_history)

        action_str = f"{tool}({params})"
        self._action_history.append(action_str)

        for f in self._discovered_files:
            if f not in self._visible_files:
                self._visible_files.append(f)

        step_reward, reward_breakdown = compute_step_reward(
            task_id=self._task_id,
            tool=tool,
            parameters=params,
            tool_result=tool_result,
            action_history=history_before,
            discovered_files=self._discovered_files,
            snapshot=self._snapshot,
        )
        self._cumulative_reward += step_reward
        self._cumulative_reward = max(0.0, min(1.0, self._cumulative_reward))

        done = False
        message = None

        if self._step_count >= self._max_steps:
            done = True
            message = f"Max steps ({self._max_steps}) reached."

        # Hard cap for task_3
        if self._task_id == "task_3" and self._step_count > 8:
            done = True
            message = "Hard step cap (8) exceeded. Score capped at 0.3."

        # ── Task completion detection ──────────────────────────────────
        # task_1: linter ran and found the bug (or agent read main.py + grepped json)
        if self._task_id == "task_1":
            linted = any("BASH" in h.upper() and "LINT" in h.upper() for h in self._action_history)
            read_main = any("FILEREAD" in h.upper() and "MAIN.PY" in h.upper() for h in self._action_history)
            found_json = any("GREP" in h.upper() and "JSON" in h.upper() for h in self._action_history)
            if linted or (read_main and found_json):
                done = True
                message = "Bug identified — grading episode."

        # task_2: config.json was written with timeout=10
        elif self._task_id == "task_2":
            import json as _json
            try:
                cfg = _json.loads(self._snapshot.get("config.json", "{}"))
                if cfg.get("timeout") == 10:
                    done = True
                    message = "Config patched successfully — grading episode."
            except Exception:
                pass

        # task_3: main.py now contains a cache mechanism
        elif self._task_id == "task_3":
            main_src = self._snapshot.get("main.py", "")
            if "lru_cache" in main_src or "_cache" in main_src:
                done = True
                message = "Caching implemented — grading episode."

        # task_4: .env contains API_KEY and main.py uses os.getenv
        elif self._task_id == "task_4":
            main_src = self._snapshot.get("main.py", "")
            env_src = self._snapshot.get(".env", "")
            if "API_KEY=SECRET_TOKEN_XYZ" in env_src.replace(" ", "") and \
               "os.getenv" in main_src and \
               "SECRET_TOKEN_XYZ" not in main_src:
                done = True
                message = "Secret migrated successfully — grading episode."

        # Redundant call message (non-terminating)
        if len(self._action_history) >= 2 and self._action_history[-1] == self._action_history[-2]:
            message = (message or "") + " Redundant call detected."

        self._done = done

        # Compute final grader score at episode end
        grader_score = None
        if done:
            grader_score, breakdown = grade_episode(
                task_id=self._task_id,
                snapshot=self._snapshot,
                action_history=self._action_history,
                steps_used=self._step_count,
            )
            self._grader_score = grader_score
            # Store globally for /grader endpoint
            global _last_grader_result
            _last_grader_result = {
                "task_id": self._task_id,
                "episode_id": self._episode_id,
                "score": grader_score,
                "breakdown": breakdown,
                "steps_used": self._step_count,
            }
            # Add completion bonus proportional to grader score
            step_reward += grader_score * 0.5
            logger.info(
                "Episode %s done: task=%s score=%.3f steps=%d",
                self._episode_id, self._task_id, grader_score, self._step_count,
            )

        return AgentObservation(
            visible_files=list(self._visible_files),
            last_tool_result=tool_result,
            action_history=list(self._action_history),
            step_count=self._step_count,
            task_description=self._task["description"],
            message=message,
            done=done,
            reward=round(step_reward, 4),
            metadata={
                "task_id": self._task_id,
                "difficulty": self._task["difficulty"],
                "cumulative_reward": round(self._cumulative_reward, 4),
                "grader_score": grader_score,
                "reward_breakdown": reward_breakdown,
                "steps_remaining": self._max_steps - self._step_count,
            },
        )

    @property
    def state(self) -> AgentState:
        return AgentState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_id=self._task_id,
            task_description=self._task.get("description", ""),
            difficulty=self._task.get("difficulty", ""),
            max_steps=self._max_steps,
            visible_files=list(self._visible_files),
            discovered_files=list(self._discovered_files),
            action_history=list(self._action_history),
            current_reward=round(self._cumulative_reward, 4),
            completed=self._done,
            grader_score=self._grader_score,
        )

    def close(self) -> None:
        pass


    def _terminal_obs(self, msg: str) -> AgentObservation:
        return AgentObservation(
            visible_files=list(self._visible_files),
            last_tool_result=msg,
            action_history=list(self._action_history),
            step_count=self._step_count,
            task_description=self._task.get("description", ""),
            message=msg,
            done=True,
            reward=0.0,
            metadata={"task_id": self._task_id, "grader_score": self._grader_score},
        )


def get_last_grader_result() -> Optional[dict]:
    return _last_grader_result