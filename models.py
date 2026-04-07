"""
AgentOps Gym — Pydantic models for Action, Observation, and State.

The agent operates on a simulated Python codebase by calling tools.
The environment is partially observable, stateful, and efficiency-aware.
Rewards shrink with wasteful or redundant tool calls.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class ToolCall(Action):
    """Agent submits a tool call with a name and parameters.

    Open action space: any valid tool name from AVAILABLE_TOOLS with
    any parameters. This mirrors how real agents interact with tool-use
    environments — no artificial discretization.
    """
    tool: str = Field(
        ...,
        description="Tool name (FileRead, FileWrite, Grep, Bash, WebSearch, TodoWrite)"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters, e.g. {'filename': 'main.py'} or {'pattern': 'def fetch'}"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional: why the agent is calling this tool (for interpretability)"
    )


class AgentObservation(Observation):
    """What the agent sees after each action.

    Inherits from Observation which provides:
        - done: bool
        - reward: Optional[float]
        - metadata: Dict[str, Any]
    """
    # Files the agent has discovered so far (partial observability)
    visible_files: List[str] = Field(
        default_factory=list,
        description="Files the agent currently knows exist in the project"
    )
    # Output of the most recent tool call
    last_tool_result: Optional[str] = Field(
        default=None,
        description="Output string from the last tool call"
    )
    # Sequential history of tool calls made this episode
    action_history: List[str] = Field(
        default_factory=list,
        description="e.g. ['Grep(pattern=timeout)', 'FileRead(config.json)']"
    )
    step_count: int = Field(default=0, description="How many steps taken so far")
    task_description: str = Field(default="", description="The task the agent must solve")
    # Feedback from the environment on quality of last action
    message: Optional[str] = Field(
        default=None,
        description="Environment feedback e.g. 'redundant call detected'"
    )


class AgentState(State):
    """Episode metadata for training harnesses and curriculum schedulers.

    Inherits from State which provides:
        - episode_id: Optional[str]
        - step_count: int
    """
    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="Human-readable task description")
    difficulty: str = Field(default="", description="easy / medium / hard")
    max_steps: int = Field(default=10, description="Max steps allowed this episode")
    visible_files: List[str] = Field(default_factory=list)
    discovered_files: List[str] = Field(default_factory=list)
    action_history: List[str] = Field(default_factory=list)
    current_reward: float = Field(default=0.0, description="Cumulative reward so far")
    completed: bool = Field(default=False)
    grader_score: Optional[float] = Field(
        default=None,
        description="Final grader score (0.0-1.0), set at end of episode"
    )