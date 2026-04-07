"""
AgentOps Gym — FastAPI application.

Exposes the OpenEnv-compatible HTTP + WebSocket API via openenv-core's
create_app(), plus custom endpoints: /tasks, /grader, /health.

A persistent singleton environment handles HTTP /reset and /step (for
the baseline script and interactive testing). WebSocket connections each
get their own AgentOpsEnvironment instance (via create_app factory pattern).
"""

import threading
import logging
from typing import Optional

from fastapi.responses import JSONResponse

from openenv.core.env_server.http_server import create_app

from agentops_gym.models import ToolCall, AgentObservation
from agentops_gym.server.environment import AgentOpsEnvironment, get_last_grader_result
from agentops_gym.server.tasks import TASK_REGISTRY

logger = logging.getLogger(__name__)

app = create_app(
    AgentOpsEnvironment,
    ToolCall,
    AgentObservation,
    env_name="agentops-gym",
)

_env = AgentOpsEnvironment()
_env_lock = threading.Lock()


def _serialize(obs: AgentObservation) -> dict:
    return obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()


app.router.routes = [
    r for r in app.router.routes
    if not (hasattr(r, "path") and r.path in ("/reset", "/step"))
]


@app.post("/reset")
async def stateful_reset(request: dict = None):
    """Reset environment for a new episode. Pass {'task_id': 'task_1'} etc."""
    import asyncio
    request = request or {}
    task_id = request.get("task_id", "task_1")

    def _do():
        with _env_lock:
            obs = _env.reset(task_id=task_id)
        return _serialize(obs)

    loop = asyncio.get_event_loop()
    obs_dict = await loop.run_in_executor(None, _do)
    return {"observation": obs_dict, "reward": 0.0, "done": False}


@app.post("/step")
async def stateful_step(request: dict = None):
    """Execute one tool call.

    Accepts two body shapes:
      1. {"action": {"tool": "...", "parameters": {...}}}   ← inference script
      2. {"tool": "...", "parameters": {...}}               ← direct curl
    """
    import asyncio
    request = request or {}

    if "action" in request:
        action_data = request["action"]
    else:
        action_data = request

    tool = action_data.get("tool", "")
    parameters = action_data.get("parameters", {})
    reasoning = action_data.get("reasoning", "")

    if not tool:
        return JSONResponse(
            status_code=400,
            content={"error": "'tool' field is required. Body must be {'action': {'tool': '...', 'parameters': {...}}}"},
        )

    def _do():
        with _env_lock:
            obs = _env.step(ToolCall(tool=tool, parameters=parameters, reasoning=reasoning))
        return _serialize(obs)

    loop = asyncio.get_event_loop()
    obs_dict = await loop.run_in_executor(None, _do)
    return {
        "observation": obs_dict,
        "reward": obs_dict.get("reward", 0.0),
        "done": obs_dict.get("done", False),
    }



@app.get("/tasks")
async def list_tasks():
    """List all available tasks with metadata."""
    tasks = []
    for tid, t in TASK_REGISTRY.items():
        tasks.append({
            "id": tid,
            "name": t["name"],
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_steps": t["max_steps"],
            "optimal_steps": t["optimal_steps"],
        })
    return {
        "tasks": tasks,
        "action_schema": {
            "tool": "string — one of FileRead|FileWrite|Grep|Bash|WebSearch|TodoWrite",
            "parameters": "dict — tool-specific params",
            "reasoning": "string (optional) — agent's reasoning",
        },
    }


@app.get("/grader")
async def grader_score():
    """Return the grader score for the last completed episode."""
    result = get_last_grader_result()
    if result is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No episode graded yet. Complete an episode first."},
        )
    return result


@app.get("/health")
async def health():
    return {"status": "ok", "env": "agentops-gym"}


def main():
    import uvicorn
    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()


