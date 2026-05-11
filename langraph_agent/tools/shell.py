from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from langchain_core.tools import tool

from langraph_agent.config import COMMAND_TIMEOUT_SECONDS, OUTPUT_LIMIT, PROJECT_ROOT


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
def bash(command: str, timeout_seconds: int = COMMAND_TIMEOUT_SECONDS) -> str:
    """执行 bash 命令。白名单命令直接执行，非白名单命令需要用户确认，危险命令会被拦截。"""
    # 这是一个高权限工具，故意把安全策略放在工具内部，而不是完全依赖模型自觉。
    # Skill 脚本也通过这个工具执行，例如：
    # bash("uv run python skills/python-debug-helper/scripts/environment_report.py")
    command = command.strip()
    if not command:
        return "命令为空，未执行。"

    if _is_dangerous_command(command):
        return f"已拦截危险命令，未执行: {command}"

    if not _is_directly_allowed_command(command):
        if not _confirm_execution("bash", command):
            return f"用户未确认，命令未执行: {command}"

    return _run_shell_command(command, timeout_seconds=timeout_seconds)


def _is_dangerous_command(command: str) -> bool:
    # 对整条命令做正则匹配，覆盖明显破坏性操作。
    # 这一步优先于白名单判断，避免 `find . -delete` 这类命令被 find 白名单放过。
    lowered = command.lower()
    return any(re.search(pattern, lowered) for pattern in DANGEROUS_COMMAND_PATTERNS)


def _is_directly_allowed_command(command: str) -> bool:
    # 白名单只允许“单条简单命令”直接执行。
    # 只要出现 shell 控制符，就必须走用户确认。
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
    # 某些命令整体看似只读，但特定参数会产生破坏性副作用。
    # 这些参数会让命令退出白名单，转入用户确认或危险拦截。
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
    # 非白名单命令需要人类在终端显式输入 y/yes。
    # 如果运行环境没有 stdin，EOFError 会被视为拒绝执行。
    print(f"\n[{tool_name}] 即将执行需要确认的命令:")
    print(command)
    try:
        answer = input("确认执行？输入 y 或 yes 继续: ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _run_shell_command(command: str, timeout_seconds: int) -> str:
    # 统一在项目根目录执行，便于命令使用相对路径。
    return _run_process(
        ["bash", "-lc", command],
        cwd=PROJECT_ROOT,
        timeout_seconds=timeout_seconds,
    )


def _run_process(command: list[str], cwd: Path, timeout_seconds: int) -> str:
    # 所有进程执行都收敛到这里，统一处理 timeout、stdout/stderr 和输出截断。
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
    if len(output) <= OUTPUT_LIMIT:
        return output
    return output[:OUTPUT_LIMIT] + "\n...[output truncated]"
