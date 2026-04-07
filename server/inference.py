#!/usr/bin/env python3
"""
AgentOps Gym — Baseline inference script.

Runs an LLM agent against all 3 tasks and reports per-task scores
in the mandatory OpenEnv stdout format.

Environment variables (MANDATORY):
    API_BASE_URL   LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace / API key (must be set)
    IMAGE_NAME     Docker image name (must be set)

Usage:
    IMAGE_NAME=agentops-gym HF_TOKEN=xxx python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


# Load .env file if present (works without it too)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME   = os.getenv("IMAGE_NAME")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") 
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:8000")

BENCHMARK   = "agentops-gym"
MAX_STEPS   = 10
TEMPERATURE = 0.3
MAX_TOKENS  = 600

ALL_TASKS = ["task_1", "task_2", "task_3", "task_4"]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert software engineer agent. You solve coding tasks by calling tools.

Available tools:
  FileRead   — Read a file.         Parameters: {"filename": "path/to/file.py"}
  FileWrite  — Write/overwrite.     Parameters: {"filename": "...", "content": "..."}
  Grep       — Search all files.    Parameters: {"pattern": "regex_or_string"}
  Bash       — Simulated shell.     Parameters: {"command": "lint main.py"}
  WebSearch  — Search docs.         Parameters: {"query": "python lru_cache"}
  TodoWrite  — Record a plan.       Parameters: {"plan": "1. Do X\\n2. Do Y"}

RULES:
1. Respond ONLY with a single JSON object — no markdown, no extra text.
2. Format exactly: {"tool": "ToolName", "parameters": {...}, "reasoning": "why"}
3. Be efficient — minimize total tool calls.
4. For hard tasks: call TodoWrite FIRST to plan, then act.
5. Never repeat the exact same tool + parameters twice in a row.

Example:
{"tool": "Grep", "parameters": {"pattern": "def fetch"}, "reasoning": "Find the function"}
"""

# ---------------------------------------------------------------------------
# Mandatory stdout log helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val = error if error else "null"
    action_short = str(action).replace("\n", " ")[:200]
    print(
        f"[STEP] step={step} action={action_short} "
        f"reward={reward:.2f} done={str(done).lower()} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_reset(task_id: str) -> Dict:
    """POST /reset and return the observation dict."""
    resp = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def http_step(tool: str, parameters: Dict, reasoning: str = "") -> Dict:
    """POST /step with the correct body shape and return the response dict."""
    body = {
        "action": {
            "tool": tool,
            "parameters": parameters,
            "reasoning": reasoning,
        }
    }
    resp = requests.post(
        f"{BASE_URL}/step",
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def http_grader() -> Dict:
    resp = requests.get(f"{BASE_URL}/grader", timeout=10)
    if resp.status_code == 200:
        return resp.json()
    return {}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(obs: Dict) -> str:
    parts = [f"TASK: {obs.get('task_description', '')}"]
    parts.append(f"\nVisible files: {obs.get('visible_files', [])}")
    last = obs.get("last_tool_result")
    if last:
        # Truncate long outputs
        parts.append(f"\nLast tool result:\n{str(last)[:1500]}")
    history = obs.get("action_history", [])
    if history:
        parts.append(f"\nHistory (last 3): {history[-3:]}")
    if obs.get("message"):
        parts.append(f"\nEnv message: {obs['message']}")
    meta = obs.get("metadata", {})
    steps_rem = meta.get("steps_remaining", "?")
    parts.append(f"\nStep {obs.get('step_count', 0)}, steps remaining: {steps_rem}")
    parts.append("\nRespond with a single JSON tool call:")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def extract_tool_call(text: str) -> Optional[Dict]:
    """Extract a valid JSON tool call from model output."""
    text = text.strip()
    # Strip markdown fences
    if "```" in text:
        for block in text.split("```"):
            block = block.strip().lstrip("json").strip()
            if block.startswith("{"):
                text = block
                break
    # Direct parse
    try:
        obj = json.loads(text)
        if "tool" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
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

def run_episode(client: OpenAI, task_id: str) -> Dict:
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    try:
        # Reset
        reset_resp = http_reset(task_id)
        obs = reset_resp.get("observation", {})

        for step in range(1, MAX_STEPS + 1):
            if reset_resp.get("done") or obs.get("done"):
                break

            # Ask the model
            prompt = build_prompt(obs)
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                raw = (completion.choices[0].message.content or "").strip()
            except Exception as e:
                error_msg = f"LLM error: {e}"
                log_step(step=step, action="(llm_error)", reward=0.0, done=True, error=str(e))
                break

            tool_call = extract_tool_call(raw)
            if tool_call is None:
                # Fallback: safe no-op grep
                tool_call = {
                    "tool": "Grep",
                    "parameters": {"pattern": "def "},
                    "reasoning": "fallback — could not parse model output",
                }

            tool      = tool_call.get("tool", "Grep")
            params    = tool_call.get("parameters", {})
            reasoning = tool_call.get("reasoning", "")
            action_str = f"{tool}({json.dumps(params)})"

            # Execute
            try:
                step_resp = http_step(tool, params, reasoning)
            except requests.HTTPError as e:
                error_msg = str(e)
                log_step(step=step, action=action_short, reward=0.0, done=True, error=error_msg)
                break

            obs     = step_resp.get("observation", {})
            reward  = float(step_resp.get("reward", 0.0) or 0.0)
            done    = bool(step_resp.get("done", False))
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        # Fetch grader score
        grader = http_grader()
        score = float(grader.get("score", 0.0) or 0.0)
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Episode error for {task_id}: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {
        "task_id":  task_id,
        "score":    score,
        "steps":    steps_taken,
        "success":  success,
        "rewards":  rewards,
    }


def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN (or API_KEY) must be set.", file=sys.stderr)
        print("  export HF_TOKEN=hf_xxx", file=sys.stderr)
        sys.exit(1)

    for attempt in range(10):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                break
        except Exception:
            pass
        print(f"[DEBUG] Waiting for server... attempt {attempt+1}/10", flush=True)
        time.sleep(2)
    else:
        print("ERROR: Server did not become ready.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 60, flush=True)
    print(f"AgentOps Gym — Baseline Inference", flush=True)
    print(f"Model: {MODEL_NAME}  |  Server: {BASE_URL}", flush=True)
    print("=" * 60, flush=True)

    results = []
    for task_id in ALL_TASKS:
        print("─" * 40, flush=True)
        result = run_episode(client, task_id)
        results.append(result)

    print("=" * 60, flush=True)
    print("BASELINE SUMMARY", flush=True)
    print("=" * 60, flush=True)

    total   = sum(r["score"] for r in results)
    solved  = sum(1 for r in results if r["success"])
    avg     = total / len(results) if results else 0.0

    for r in results:
        status = "✅ PASS" if r["success"] else "❌ FAIL"
        print(f"  {r['task_id']:>8}    score={r['score']:.3f}  steps={r['steps']:2d}  {status}", flush=True)

    print(f"\n  Average score: {avg:.3f}", flush=True)
    print(f"  Solved: {solved} / {len(results)}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()