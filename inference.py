#!/usr/bin/env python3
"""
AgentOps Gym — Baseline inference script.

Runs an LLM agent against all 3 AgentOps Gym tasks (tool-use efficiency)
and reports per-task scores in the mandatory OpenEnv stdout format.

Environment variables (MANDATORY):
    API_BASE_URL   The API endpoint for the LLM (default: HF router)
    MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Your Hugging Face / API key (must be set)
    IMAGE_NAME     Docker image name for the environment (must be set)

Usage:
    IMAGE_NAME=agentops-gym HF_TOKEN=xxx python inference.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from agentops_gym.client import AgentOpsEnv
from agentops_gym.models import ToolCall

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME  = os.getenv("IMAGE_NAME")
API_KEY     = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME  = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK  = "agentops-gym"
MAX_STEPS  = 10
TEMPERATURE = 0.0
MAX_TOKENS  = 600

ALL_TASKS = ["task_1", "task_2", "task_3", "task_4"]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert software engineer agent. You solve coding tasks by calling tools.

Available tools:
  FileRead   — Read a file. Parameters: {"filename": "path/to/file.py"}
  FileWrite  — Write/overwrite a file. Parameters: {"filename": "...", "content": "..."}
  Grep       — Search files for a pattern. Parameters: {"pattern": "regex_or_string"}
  Bash       — Run simulated shell command. Parameters: {"command": "lint main.py"}
  WebSearch  — Search documentation. Parameters: {"query": "python lru_cache"}
  TodoWrite  — Write a plan. Parameters: {"plan": "1. Do X\\n2. Do Y"}

RULES:
1. Respond ONLY with a single JSON object — no markdown, no explanation.
2. Format: {"tool": "ToolName", "parameters": {...}, "reasoning": "why"}
3. Be efficient — minimize total tool calls.
4. For hard tasks: use TodoWrite FIRST to plan, then act.
5. Never call the exact same tool+parameters twice.

Example response:
{"tool": "Grep", "parameters": {"pattern": "def fetch"}, "reasoning": "Find the function location"}
"""

# ---------------------------------------------------------------------------
# Logging helpers (mandatory OpenEnv stdout format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val = error if error else "null"
    done_val = str(done).lower()
    action_short = action.replace("\n", " ")[:200]
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(obs_data: Dict[str, Any]) -> str:
    parts = [f"TASK: {obs_data.get('task_description', '')}"]
    parts.append(f"\nVisible files: {obs_data.get('visible_files', [])}")
    if obs_data.get("last_tool_result"):
        parts.append(f"\nLast tool result:\n{obs_data['last_tool_result']}")
    history = obs_data.get("action_history", [])
    if history:
        parts.append(f"\nHistory ({len(history)} calls): {history[-3:]}")  # last 3
    if obs_data.get("message"):
        parts.append(f"\nEnvironment message: {obs_data['message']}")
    meta = obs_data.get("metadata", {})
    parts.append(f"\nStep {obs_data.get('step_count', 0)}/{meta.get('max_steps', 10)}, "
                 f"steps remaining: {meta.get('steps_remaining', '?')}")
    parts.append("\nRespond with a single JSON tool call:")
    return "\n".join(parts)


def extract_tool_call(text: str) -> Optional[Dict]:
    """Extract JSON tool call from model response."""
    text = text.strip()
    # Strip markdown fences if present
    if "```" in text:
        blocks = text.split("```")
        for b in blocks:
            b = b.strip().lstrip("json").strip()
            if b.startswith("{"):
                text = b
                break
    # Try direct JSON parse
    try:
        obj = json.loads(text)
        if "tool" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Try to extract first {...} block
    import re
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if "tool" in obj:
                return obj
        except json.JSONDecodeError:
            pass
    return None

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    env: AgentOpsEnv,
    client: OpenAI,
    task_id: str,
) -> Dict[str, Any]:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = await env.reset(seed=None, task_id=task_id)
        obs = result.observation
        obs_data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            prompt = build_prompt(obs_data)
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )

            raw = (completion.choices[0].message.content or "").strip()
            tool_call = extract_tool_call(raw)

            if tool_call is None:
                # Fallback: emit a safe no-op
                tool_call = {"tool": "Grep", "parameters": {"pattern": "def "}, "reasoning": "fallback"}

            tool = tool_call.get("tool", "Grep")
            parameters = tool_call.get("parameters", {})
            reasoning = tool_call.get("reasoning", "")
            action_str = f"{tool}({json.dumps(parameters)})"

            result = await env.step(ToolCall(tool=tool, parameters=parameters, reasoning=reasoning))
            obs = result.observation
            obs_data = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()

            reward = result.reward or 0.0
            done = result.done
            error = None  # tools return errors inside last_tool_result, not as exceptions

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        meta = obs_data.get("metadata", {})
        score = meta.get("grader_score") or 0.0
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Episode error for {task_id}: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "steps": steps_taken,
        "success": success,
        "rewards": rewards,
    }

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def async_main() -> None:
    if not API_KEY:
        raise SystemExit(
            "HF_TOKEN (or API_KEY) must be set.\n"
            "  export HF_TOKEN=your_token_here"
        )
    if not IMAGE_NAME:
        raise SystemExit(
            "IMAGE_NAME must be set.\n"
            "  export IMAGE_NAME=agentops-gym"
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with AgentOpsEnv.from_docker_image(IMAGE_NAME) as env:
        results = []
        for task_id in ALL_TASKS:
            result = await run_episode(env, client, task_id)
            results.append(result)

        # Summary
        print(f"\n{'='*60}", flush=True)
        print("SUMMARY", flush=True)
        print(f"{'='*60}", flush=True)

        total = sum(r["score"] for r in results)
        resolved = sum(1 for r in results if r["success"])
        avg = total / len(results) if results else 0.0

        for r in results:
            status = "SOLVED" if r["success"] else "FAILED"
            print(f"  {r['task_id']:>8}: score={r['score']:.3f}  steps={r['steps']}  {status}", flush=True)

        print(f"\n  Total:    {total:.3f} / {len(results)}", flush=True)
        print(f"  Average:  {avg:.3f}", flush=True)
        print(f"  Solved:   {resolved} / {len(results)}", flush=True)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()