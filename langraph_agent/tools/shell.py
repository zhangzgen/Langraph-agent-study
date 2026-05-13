from __future__ import annotations

import re
import shlex
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from langchain_core.tools import tool

from langraph_agent.config import config


_GRAPH_APPROVED_EXECUTION = ContextVar("graph_approved_execution", default=False)


@contextmanager
def graph_approved_execution():
    """标记当前 bash 调用已经通过 LangGraph 人工审批。"""
    token = _GRAPH_APPROVED_EXECUTION.set(True)
    try:
        yield
    finally:
        _GRAPH_APPROVED_EXECUTION.reset(token)


# 危险名单：匹配到这些模式时直接拒绝执行。
# 这里故意使用“偏保守”的规则，因为 bash 是能修改本机文件系统的高权限工具。
DANGEROUS_COMMAND_PATTERNS = [
    r"\brm\s+.*(-r|-R|--recursive).*(-f|--force)",
    r"\bsudo\b",
    r"\bsu\b",
    r"\bchmod\s+.*777\b",
    r"\bchown\s+.*(-r|-R|--recursive)",
    r"\bfind\b.*\s-delete\b",
    r"\bxargs\b.*\brm\b",
    r"\bgit\s+reset\b.*--hard\b",
    r"\bgit\s+clean\b.*-[^\s]*f",
    r"\bmkfs\b",
    r"\bdd\s+.*\bof=",
    r"\bdiskutil\s+erase",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
    r"\bkill\s+-9\s+-1\b",
    r"\bpkill\b",
    r"\bkillall\b",
    r":\s*\(\s*\)\s*\{",
    r"(curl|wget)\b.*\|\s*(sh|bash|zsh|python|python3)\b",
]

# 如果命令包含管道、重定向、命令替换或多命令串联，就不走直接执行白名单。
# 例如 `cat file | sh` 即使以 cat 开头，也必须要求用户确认。
SHELL_CONTROL_OPERATORS = (";", "&&", "||", "|", ">", "<", "`", "$(", "\n")

# 直接执行白名单：主要放只读或低风险命令。
# 注意：命令在进入白名单前还会经过危险名单和参数级检查。
DIRECT_ALLOWLIST_COMMANDS = {
    "pwd",
    "ls",
    "find",
    "rg",
    "grep",
    "cat",
    "sed",
    "awk",
    "head",
    "tail",
    "wc",
    "date",
    "whoami",
    "which",
    "echo",
}
DIRECT_ALLOWLIST_PREFIXES = (
    ("git", "status"),
    ("git", "diff"),
    ("git", "log"),
    ("git", "show"),
    ("git", "branch"),
    ("git", "remote"),
    ("python", "--version"),
    ("python3", "--version"),
    ("uv", "--version"),
    ("uv", "run", "python", "-m", "compileall"),
)


@tool
def bash(command: str, timeout_seconds: int = config.COMMAND_TIMEOUT_SECONDS) -> str:
    """执行 bash 命令。白名单命令直接执行，非白名单命令需要用户确认，危险命令会被拦截。

    这是高权限工具，安全策略放在工具内部作为第二层保护。Skill 脚本也通过
    这个工具执行，例如 bash("uv run python skills/python-debug-helper/scripts/environment_report.py")。
    """
    command = command.strip()
    if not command:
        return "命令为空，未执行。"

    if _is_dangerous_command(command):
        return f"已拦截危险命令，未执行: {command}"

    if not _is_directly_allowed_command(command) and not _GRAPH_APPROVED_EXECUTION.get():
        if not _confirm_execution("bash", command):
            return f"用户未确认，命令未执行: {command}"

    return _run_shell_command(command, timeout_seconds=timeout_seconds)


def _is_dangerous_command(command: str) -> bool:
    """判断命令是否命中明显破坏性的危险模式。"""
    lowered = command.lower()
    if _is_dangerous_rm_command(command):
        return True
    return any(re.search(pattern, lowered) for pattern in DANGEROUS_COMMAND_PATTERNS)


def _is_dangerous_rm_command(command: str) -> bool:
    """专门识别 rm -rf 这类递归强制删除命令。"""
    try:
        parts = shlex.split(command)
    except ValueError:
        return False

    if not parts or parts[0] != "rm":
        return False

    options = [part for part in parts[1:] if part.startswith("-")]
    has_recursive = any(
        option in {"-r", "-R", "--recursive"}
        or (option.startswith("-") and "r" in option.lower())
        for option in options
    )
    has_force = any(
        option == "--force" or (option.startswith("-") and "f" in option.lower())
        for option in options
    )
    return has_recursive and has_force


def _is_directly_allowed_command(command: str) -> bool:
    """判断命令是否属于可以跳过确认的简单低风险命令。"""
    if any(operator in command for operator in SHELL_CONTROL_OPERATORS):
        return False

    try:
        parts = shlex.split(command)
    except ValueError:
        return False

    if not parts:
        return False
    if _has_disallowed_allowlist_options(parts):
        return False
    if parts[0] in DIRECT_ALLOWLIST_COMMANDS:
        return True
    return any(tuple(parts[: len(prefix)]) == prefix for prefix in DIRECT_ALLOWLIST_PREFIXES)


def _has_disallowed_allowlist_options(parts: list[str]) -> bool:
    """识别会让白名单命令产生副作用的危险参数。"""
    if parts[0] == "sed" and "-i" in parts:
        return True
    if parts[0] == "find" and any(part in {"-delete", "-exec", "-execdir"} for part in parts):
        return True
    if parts[:2] == ["git", "branch"] and any(part in {"-d", "-D", "--delete"} for part in parts):
        return True
    if parts[:2] == ["git", "remote"] and len(parts) > 2:
        return parts[2] not in {"-v", "show", "get-url"}
    return False


def _confirm_execution(tool_name: str, command: str) -> bool:
    """在终端请求用户确认非白名单命令。"""
    print(f"\n[{tool_name}] 即将执行需要确认的命令:")
    print(command)
    try:
        answer = input("确认执行？输入 y 或 yes 继续: ").strip().lower()
    except (EOFError, OSError):
        return False
    return answer in {"y", "yes"}


def _run_shell_command(command: str, timeout_seconds: int) -> str:
    """在项目根目录运行 bash 命令。"""
    return _run_process(
        ["bash", "-lc", command],
        cwd=config.PROJECT_ROOT,
        timeout_seconds=timeout_seconds,
    )


def _run_process(command: list[str], cwd: Path, timeout_seconds: int) -> str:
    """执行子进程并统一处理 timeout、stdout/stderr 和输出截断。"""
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=max(1, timeout_seconds),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return f"命令超时，已终止。timeout={exc.timeout}s"
    except ValueError as exc:
        return str(exc)
    except OSError as exc:
        return f"命令执行失败: {exc}"

    stdout = _truncate_output(completed.stdout.strip())
    stderr = _truncate_output(completed.stderr.strip())
    sections = [f"exit_code: {completed.returncode}"]
    if stdout:
        sections.append(f"stdout:\n{stdout}")
    if stderr:
        sections.append(f"stderr:\n{stderr}")
    return "\n\n".join(sections)


def _truncate_output(output: str) -> str:
    """限制命令输出长度，避免工具结果过大。"""
    if len(output) <= config.OUTPUT_LIMIT:
        return output
    return output[: config.OUTPUT_LIMIT] + "\n...[output truncated]"
