"""
AgentOps Gym — Simulated tool implementations.

All tools operate on an in-memory filesystem snapshot. No real subprocess,
no real filesystem, fully deterministic and reproducible. The fake linter/
test runner uses static analysis of the snapshot strings.
"""

import re
import json
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# In-memory project snapshots (one per task)
# ---------------------------------------------------------------------------

PROJECT_SNAPSHOTS: Dict[str, Dict[str, str]] = {
    "task_1": {
        "main.py": """\
import requests

def fetch_user(user_id):
    url = f"https://api.example.com/users/{user_id}"
    response = requests.get(url)
    return response.json          # BUG: missing () — should be response.json()

def main():
    user = fetch_user(123)
    print(user['name'])

if __name__ == "__main__":
    main()
""",
        "utils.py": "def helper(): pass\n",
        "config.json": '{"api_url": "https://api.example.com", "timeout": 30}\n',
        "README.md": "# Example Project\n",
    },
    "task_2": {
        "main.py": """\
import requests
import json

def fetch_data(endpoint):
    url = f"https://api.example.com/{endpoint}"
    response = requests.get(url, timeout=30)
    return response.json()

def main():
    data = fetch_data("data")
    print(data)
""",
        "utils.py": "def helper(): pass\n",
        "config.json": '{"api_url": "https://api.example.com", "timeout": 30}\n',
        "README.md": "# Example Project\n",
    },
    "task_3": {
        "main.py": """\
import requests

def fetch_user(user_id):
    url = f"https://api.example.com/users/{user_id}"
    response = requests.get(url)
    return response.json()

def main():
    for uid in range(100):
        user = fetch_user(uid)
        print(user['name'])

if __name__ == "__main__":
    main()
""",
        "utils.py": "def helper(): pass\n",
        "config.json": '{"api_url": "https://api.example.com", "timeout": 30}\n',
        "README.md": "# Example Project\n",
        "tests/test_main.py": """\
from main import fetch_user

def test_fetch_user():
    result = fetch_user(1)
    assert result is not None
""",
    },
    "task_4": {
        "main.py": """\
import requests

API_KEY = "SECRET_TOKEN_XYZ"

def fetch_data():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get("https://api.example.com/data", headers=headers)
    return response.json()

if __name__ == "__main__":
    print(fetch_data())
""",
        "README.md": "# Project Alpha\nSecure the API key.\n",
    },
}

# ---------------------------------------------------------------------------
# Simulated web search index
# ---------------------------------------------------------------------------

WEB_SEARCH_DOCS: Dict[str, str] = {
    "lru_cache": """\
functools.lru_cache — Python docs
  @functools.lru_cache(maxsize=128)
  def my_function(arg): ...
  Caches results of function calls. Use maxsize=None for unlimited cache.
""",
    "response.json": """\
requests.Response.json() — requests docs
  response.json() returns the JSON-encoded content of the response.
  Note: json is a method, must be called with parentheses: response.json()
""",
    "timeout": """\
requests timeout — requests docs
  Set timeout in seconds: requests.get(url, timeout=10)
  Recommended: keep timeout low (5-15s) for production APIs.
""",
    "python caching": """\
Python caching patterns:
  1. functools.lru_cache — in-memory memoization decorator
  2. dict-based cache    — manual dict for full control
  3. joblib.Memory       — disk-backed cache
  For simple in-memory caching, lru_cache is idiomatic Python.
""",
    "getenv": """\
os.getenv(key, default=None) — Python docs
  Return the value of the environment variable key if it exists, or default if it doesn't.
  Example:
    import os
    api_key = os.getenv('API_KEY')
""",
    ".env": """\
.env files — Best Practices
  Store secrets and configuration in a .env file:
    API_KEY=your_secret_here
  Never commit .env files to version control.
""",
}

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

AVAILABLE_TOOLS = {
    "FileRead":  "Read contents of a specific file",
    "FileWrite": "Write/edit a specific file with new content",
    "Grep":      "Search for a pattern across all files",
    "Bash":      "Run a shell command (simulated: lint, test runner)",
    "WebSearch": "Search for documentation (simulated)",
    "TodoWrite": "Write a plan/todo list before acting",
}


def run_tool(
    tool: str,
    parameters: Dict,
    snapshot: Dict[str, str],
    discovered_files: list,
) -> Tuple[str, Dict[str, str], list]:
    """
    Execute a simulated tool and return (result_string, updated_snapshot, updated_discovered).
    All mutations to the snapshot are returned as a new dict.
    """
    snapshot = dict(snapshot)
    discovered = list(discovered_files)

    if tool == "FileRead":
        return _file_read(parameters, snapshot, discovered)
    elif tool == "FileWrite":
        return _file_write(parameters, snapshot, discovered)
    elif tool == "Grep":
        return _grep(parameters, snapshot, discovered)
    elif tool == "Bash":
        return _bash(parameters, snapshot)
    elif tool == "WebSearch":
        return _web_search(parameters), snapshot, discovered
    elif tool == "TodoWrite":
        return _todo_write(parameters), snapshot, discovered
    else:
        return f"ERROR: Unknown tool '{tool}'. Available: {list(AVAILABLE_TOOLS.keys())}", snapshot, discovered


def _file_read(params, snapshot, discovered):
    fname = params.get("filename", "")
    if not fname:
        return "ERROR: 'filename' parameter required for FileRead.", snapshot, discovered
    if fname not in snapshot:
        return f"ERROR: File '{fname}' not found in project.", snapshot, discovered
    # Reveal file in discovered list
    if fname not in discovered:
        discovered.append(fname)
    content = snapshot[fname]
    lines = content.splitlines()
    numbered = "\n".join(f"{i+1:3}: {line}" for i, line in enumerate(lines))
    return f"=== {fname} ===\n{numbered}", snapshot, discovered


def _file_write(params, snapshot, discovered):
    fname = params.get("filename", "")
    content = params.get("content", "")
    if not fname:
        return "ERROR: 'filename' parameter required for FileWrite.", snapshot, discovered
    snapshot[fname] = content
    if fname not in discovered:
        discovered.append(fname)
    return f"Write successful: {fname} ({len(content)} bytes written)", snapshot, discovered


def _grep(params, snapshot, discovered):
    pattern = params.get("pattern", "")
    if not pattern:
        return "ERROR: 'pattern' parameter required for Grep.", snapshot, discovered
    results = []
    for fname, content in snapshot.items():
        for i, line in enumerate(content.splitlines(), 1):
            if re.search(pattern, line, re.IGNORECASE):
                results.append(f"{fname}:{i} → {line.strip()}")
                # Discovering a file via grep reveals it
                if fname not in discovered:
                    discovered.append(fname)
    if not results:
        return f"No matches for pattern '{pattern}'.", snapshot, discovered
    return "\n".join(results), snapshot, discovered


def _bash(params, snapshot):
    cmd = params.get("command", "")
    if not cmd:
        return "ERROR: 'command' parameter required for Bash.", snapshot, []

    cmd_lower = cmd.lower()

    # Simulated linter
    if "lint" in cmd_lower or "flake8" in cmd_lower or "pylint" in cmd_lower:
        fname = None
        for f in snapshot:
            if f.endswith(".py") and f in cmd:
                fname = f
                break
        if fname and fname in snapshot:
            return _lint_file(fname, snapshot[fname]), snapshot, []
        # Lint all py files
        out = []
        for f, content in snapshot.items():
            if f.endswith(".py"):
                out.append(_lint_file(f, content))
        return "\n".join(out) if out else "No Python files found.", snapshot, []

    # Simulated test runner
    if "pytest" in cmd_lower or "test" in cmd_lower:
        test_files = [f for f in snapshot if "test" in f]
        if not test_files:
            return "No test files found.", snapshot, []
        # Check if main.py has obvious bugs
        main_content = snapshot.get("main.py", "")
        if "response.json\n" in main_content or "response.json " in main_content:
            return '{"status": "error", "file": "main.py", "line": 6, "message": "AttributeError: method object is not subscriptable — did you forget response.json()?"}'
        return '{"status": "pass", "passed": 1, "failed": 0}', snapshot, []

    # Simulated validate (for config check)
    if "validate" in cmd_lower or "json" in cmd_lower:
        for fname, content in snapshot.items():
            if fname.endswith(".json") and fname in cmd:
                try:
                    json.loads(content)
                    return f"✓ {fname} is valid JSON", snapshot, []
                except json.JSONDecodeError as e:
                    return f"✗ {fname} invalid JSON: {e}", snapshot, []
        return "Validation complete.", snapshot, []

    return f"$ {cmd}\n(simulated) Command executed. No output.", snapshot, []


def _lint_file(fname: str, content: str) -> str:
    errors = []
    for i, line in enumerate(content.splitlines(), 1):
        # Check for common bug: response.json without ()
        if re.search(r'response\.json\b(?!\()', line):
            errors.append(f'  {fname}:{i}: E001 response.json called without parentheses — should be response.json()')
        # Check for bare except
        if re.match(r'\s*except\s*:', line):
            errors.append(f'  {fname}:{i}: W001 Bare except clause detected')
        # Check for hardcoded secrets (task_4)
        if "SECRET_TOKEN_XYZ" in line and fname == "main.py":
            errors.append(f'  {fname}:{i}: E002 Hardcoded secret detected — use environment variables')
    if errors:
        return f'{fname}: {len(errors)} issue(s) found\n' + '\n'.join(errors)
    return f'{fname}: OK'


def _web_search(params) -> str:
    query = params.get("query", "").lower()
    for key, doc in WEB_SEARCH_DOCS.items():
        if key in query:
            return doc
    return f"No results found for '{params.get('query', '')}'. Try more specific terms."


def _todo_write(params) -> str:
    plan = params.get("plan", params.get("content", ""))
    if not plan:
        return "ERROR: 'plan' parameter required for TodoWrite."
    return f"✓ Plan recorded:\n{plan}"