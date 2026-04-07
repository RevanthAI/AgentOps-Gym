"""
AgentOps Gym — Environment client.

Wraps WebSocket communication with the environment server.
Provides typed step/reset/state methods for the agent.
"""

from typing import Dict, Any
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from agentops_gym.models import ToolCall, AgentObservation, AgentState


class AgentOpsEnv(EnvClient[ToolCall, AgentObservation, AgentState]):
    """Client for the AgentOps Gym environment."""

    def _step_payload(self, action: ToolCall) -> Dict[str, Any]:
        """Convert a ToolCall action to the JSON payload expected by the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AgentObservation]:
        """Parse server response into a StepResult with typed observation."""
        obs_data = payload.get("observation", {})
        obs = AgentObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AgentState:
        """Parse server state response into typed State object."""
        return AgentState(**payload)