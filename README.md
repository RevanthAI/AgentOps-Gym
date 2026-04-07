---
title: Agentops Gym Environment Server
emoji: 🏏
colorFrom: gray
colorTo: pink
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Agentops Gym Environment

Stateful, partially observable, efficiency-penalizing RL environment for training agents on software engineering tool-use tasks.

## Quick Start

The simplest way to use the Agentops Gym environment is through the `AgentopsGymEnv` class:

```python
from agentops_gym import AgentopsGymAction, AgentopsGymEnv
from agentops_gym.models import ToolCall

try:
    # Create environment from Docker image
    agentops_gymenv = AgentopsGymEnv.from_docker_image("agentops_gym-env:latest")

    # Reset to start a task
    result = agentops_gymenv.reset(task_id="task_1")
    print(f"Task: {result.observation.task_description}")

    # Use tools to complete the task
    # Example: Search for a pattern
    action = AgentopsGymAction(
        tool_call=ToolCall(tool="Grep", parameters={"pattern": "json"})
    )
    result = agentops_gymenv.step(action)
    print(f"Grep Result: {result.observation.last_tool_result}")

finally:
    # Always clean up
    agentops_gymenv.close()
```

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t agentops_gym-env:latest -f agentops_gym/server/Dockerfile .
```

## Environment Details

### Action
**AgentopsGymAction**:
- `tool_call` (ToolCall) - The tool to execute (Grep, FileRead, FileWrite, Bash, TodoWrite, Submit)
- `reasoning` (str, optional) - Agent's explanation for the action

### Observation
**AgentopsGymObservation**:
- `task_description` (str) - The task objective
- `visible_files` (list[str]) - Files discovered so far
- `last_tool_result` (str) - Output of the last tool call
- `action_history` (list[str]) - Previous actions in this episode
- `step_count` (int) - Current step number
- `max_steps` (int) - Maximum allowed steps
- `done` (bool) - Whether the episode is complete
- `feedback` (str, optional) - Warnings or penalties from the environment

### Available Tools
- **Grep**: Search for patterns in the virtual filesystem.
- **FileRead**: Read file contents.
- **FileWrite**: Modify file contents.
- **Bash**: Run simulated commands (lint, test).
- **TodoWrite**: Save a plan for the task.
- **Submit**: Submit the final answer.

## Advanced Usage

### Using the Context Manager

```python
from agentops_gym import AgentopsGymAction, AgentopsGymEnv
from agentops_gym.models import ToolCall

with AgentopsGymEnv(base_url="http://localhost:8000") as env:
    result = env.reset(task_id="task_1")
    # Execute steps...
    action = AgentopsGymAction(tool_call=ToolCall(tool="FileRead", parameters={"filename": "README.md"}))
    result = env.step(action)
```

## Running Locally

Run the server locally for development:

```bash
cd agentops_gym
uvicorn server.app:app --reload
```

## Project Structure

```
agentops_gym/
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── agentops_gym_environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    └── Dockerfile         # Container image definition
```
