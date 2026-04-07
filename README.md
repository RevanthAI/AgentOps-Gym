# Agentops Gym: Optimizing Tool-Use Efficiency

**"LLMs burn tokens via inefficient tool usage."**

Agentops Gym is a stateful, partially observable, efficiency-penalizing RL environment designed to train and evaluate agents on software engineering tasks. While many environments focus solely on task completion, Agentops Gym prioritizes **efficiency**—penalizing redundant calls, reward-hacking, and "hallucinated" file reads to help you build agents that solve problems with minimal token consumption.

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

## Docker Build & Run

### 1. Build the Image
Build the environment server from the project root:
```bash
docker build -t agentops-gym -f agentops_gym/server/Dockerfile .
```

### 2. Run the Container
Start the server on port 8000:
```bash
# Remove existing container if necessary
docker stop agentops-gym && docker rm agentops-gym

# Run new container
docker run -d --name agentops-gym -p 8000:8000 agentops-gym
```

### 3. Verify & Logs
```bash
# Check health
curl http://localhost:8000/health

# Tail logs
docker logs -f agentops-gym
```

## Run Baseline Inference

The project includes a baseline inference script to evaluate agents across all tasks (including the new Task 4: Secret Migration).

### Setup
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
export IMAGE_NAME=agentops-gym

# Optional overrides:
# export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
# export API_BASE_URL=https://router.huggingface.co/v1
```

### Run
```bash
python agentops_gym/inference.py
```

### Expected Output
```text
============================================================
AgentOps Gym — Baseline Inference
Model: gpt-4.1  |  Server: http://localhost:8000
============================================================
────────────────────────────────────────
[START] task=task_1 env=agentops-gym model=gpt-4.1
[STEP] step=1 action=Grep({"pattern": "def fetch_user"}) reward=0.00 done=false error=null
[STEP] step=2 action=Grep({"pattern": "return"}) reward=0.00 done=false error=null
[STEP] step=3 action=FileRead({"filename": "main.py"}) reward=0.10 done=false error=null
...
[STEP] step=8 action=FileRead({"filename": "main.py"}) reward=0.14 done=true error=null
[END] success=false steps=8 rewards=0.00,0.00,0.10,-0.05,-0.05,-0.05,-0.05,0.14
────────────────────────────────────────
[START] task=task_2 env=agentops-gym model=gpt-4.1
[STEP] step=1 action=Grep({"pattern": "timeout"}) reward=0.05 done=false error=null
[STEP] step=2 action=FileRead({"filename": "config.json"}) reward=0.10 done=false error=null
[STEP] step=3 action=FileWrite({"filename": "config.json", "content": "{\"api_url\": \"https://api.example.com\", \"timeout\": 10}"}) reward=0.55 done=true error=null
[END] success=true steps=3 rewards=0.05,0.10,0.55
────────────────────────────────────────
[START] task=task_3 env=agentops-gym model=gpt-4.1
...
[STEP] step=8 action=Grep({"pattern": "def "}) reward=0.20 done=true error=null
[END] success=false steps=8 rewards=0.10,0.00,0.05,0.05,0.05,0.00,0.05,0.20
────────────────────────────────────────
[START] task=task_4 env=agentops-gym model=gpt-4.1
[STEP] step=1 action=TodoWrite({"plan": "..."}) reward=0.05 done=false error=null
[STEP] step=2 action=Grep({"pattern": "SECRET_TOKEN_XYZ"}) reward=0.05 done=false error=null
[STEP] step=3 action=FileRead({"filename": "main.py"}) reward=0.05 done=false error=null
[STEP] step=4 action=FileWrite({"filename": ".env", "content": "API_KEY=SECRET_TOKEN_XYZ\n"}) reward=0.10 done=false error=null
[STEP] step=10 action=FileWrite({"filename": "main.py", "content": "import os\n..."}) reward=0.43 done=true error=null
[END] success=true steps=10 rewards=0.05,0.05,0.05,0.10,0.05,0.00,0.05,0.05,0.10,0.43

============================================================
BASELINE SUMMARY
============================================================
    task_1    score=0.390  steps= 8  ❌ FAIL
    task_2    score=1.000  steps= 3  ✅ PASS
    task_3    score=0.392  steps= 8  ❌ FAIL
    task_4    score=0.856  steps=10  ✅ PASS

  Average score: 0.659
  Solved: 2 / 4
============================================================
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
