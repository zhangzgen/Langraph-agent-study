from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Literal

from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool

from langraph_agent.config import config


_FILESYSTEM_BACKEND = FilesystemBackend(
    root_dir=config.PROJECT_ROOT,
    virtual_mode=True,
)


@tool
def ls(path: str = "/") -> str:
    """列出项目虚拟文件系统中的目录内容。

    Description:
        使用 Deep Agents 的 FilesystemBackend 读取指定目录下的文件和子目录。
        路径会被限制在当前项目根目录映射出的虚拟根路径内。
    Args:
        path (str): 要列出的目录路径，支持项目内相对路径、虚拟绝对路径或项目内真实绝对路径。
    Returns:
        str: JSON 字符串，包含目录项路径、类型、大小和修改时间；失败时返回错误文本。
    """
    normalized_path = _normalize_virtual_path(path)
    result = _FILESYSTEM_BACKEND.ls(normalized_path)
    if result.error:
        return f"Error: {result.error}"
    return _dump_json(result.entries or [])


@tool
def read_file(file_path: str, offset: int = 0, limit: int = 100) -> str:
    """读取项目虚拟文件系统中的文本文件。

    Description:
        使用 Deep Agents 的 FilesystemBackend 按行读取文件内容，并在输出中保留行号，
        便于模型后续定位和编辑文件。
    Args:
        file_path (str): 要读取的文件路径，支持项目内相对路径、虚拟绝对路径或项目内真实绝对路径。
        offset (int): 读取起始行，使用 0 起始计数。
        limit (int): 最多读取的行数，用于控制大文件输出规模。
    Returns:
        str: 带行号的文件内容；文件不存在、越权或读取失败时返回错误文本。
    """
    normalized_path = _normalize_virtual_path(file_path)
    bounded_offset = max(offset, 0)
    bounded_limit = min(max(limit, 1), 1000)
    result = _FILESYSTEM_BACKEND.read(
        normalized_path,
        offset=bounded_offset,
        limit=bounded_limit,
    )
    if result.error:
        return f"Error: {result.error}"
    if result.file_data is None:
        return f"Error: no data returned for {normalized_path}"

    content = result.file_data["content"]
    if not content:
        return f"File {normalized_path} is empty."
    return _format_lines(content, start_line=bounded_offset + 1)


@tool
def write_file(file_path: str, content: str) -> str:
    """在项目虚拟文件系统中写入文件。

    Description:
        使用 Deep Agents 的 FilesystemBackend 创建或覆盖指定文件。该工具会修改本地文件，
        因此默认必须经过 tool_guard 的人工审核后才会执行。
    Args:
        file_path (str): 要写入的文件路径，支持项目内相对路径、虚拟绝对路径或项目内真实绝对路径。
        content (str): 要写入文件的完整文本内容。
    Returns:
        str: 写入成功时返回更新后的虚拟路径；失败时返回错误文本。
    """
    normalized_path = _normalize_virtual_path(file_path)
    result = _FILESYSTEM_BACKEND.write(normalized_path, content)
    if result.error:
        return f"Error: {result.error}"
    return f"Updated file {result.path or normalized_path}"


@tool
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> str:
    """对项目虚拟文件系统中的文件执行精确字符串替换。

    Description:
        使用 Deep Agents 的 FilesystemBackend 在已有文件中替换指定文本。该工具适合
        小范围修改已有文件，默认必须经过 tool_guard 的人工审核后才会执行。
    Args:
        file_path (str): 要编辑的文件路径，支持项目内相对路径、虚拟绝对路径或项目内真实绝对路径。
        old_string (str): 需要被替换的原始字符串。
        new_string (str): 替换后的新字符串。
        replace_all (bool): 是否替换文件中的全部匹配项；为 False 时要求原始字符串唯一。
    Returns:
        str: 编辑成功时返回替换次数和文件路径；失败时返回错误文本。
    """
    normalized_path = _normalize_virtual_path(file_path)
    result = _FILESYSTEM_BACKEND.edit(
        normalized_path,
        old_string,
        new_string,
        replace_all=replace_all,
    )
    if result.error:
        return f"Error: {result.error}"
    occurrences = result.occurrences if result.occurrences is not None else 0
    return f"Successfully replaced {occurrences} instance(s) in {result.path or normalized_path}"


@tool
def glob(pattern: str, path: str = "/") -> str:
    """按 glob 模式查找项目文件。

    Description:
        使用 Deep Agents 的 FilesystemBackend 在项目虚拟文件系统中查找匹配路径。
        适合替代旧的 file_search 文件名搜索能力。
    Args:
        pattern (str): glob 匹配模式，例如 `**/*.py`、`*.md`。
        path (str): 搜索起始目录，默认为项目虚拟根路径 `/`。
    Returns:
        str: JSON 字符串，包含匹配文件的路径、类型、大小和修改时间；失败时返回错误文本。
    """
    normalized_path = _normalize_virtual_path(path)
    result = _FILESYSTEM_BACKEND.glob(pattern, path=normalized_path)
    if result.error:
        return f"Error: {result.error}"
    return _dump_json(result.matches or [])


@tool
def grep(
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
) -> str:
    """在项目文件中搜索文字内容。

    Description:
        使用 Deep Agents 的 FilesystemBackend 执行字面量文本搜索。该工具适合替代旧的
        file_search 内容搜索场景，并支持按 glob 模式缩小搜索范围。
    Args:
        pattern (str): 要搜索的字面量文本，不按正则表达式解释。
        path (str | None): 搜索目录；为空时从项目虚拟根路径开始。
        glob (str | None): 文件 glob 过滤模式，例如 `*.py`。
        output_mode (Literal["files_with_matches", "content", "count"]): 输出模式。
    Returns:
        str: 搜索结果文本；失败时返回错误文本。
    """
    normalized_path = _normalize_virtual_path(path or "/")
    result = _FILESYSTEM_BACKEND.grep(
        pattern,
        path=normalized_path,
        glob=glob,
    )
    if result.error:
        return f"Error: {result.error}"

    matches = result.matches or []
    if output_mode == "content":
        return "\n".join(
            f"{item['path']}:{item['line']}: {item['text']}" for item in matches
        )
    if output_mode == "count":
        counts: dict[str, int] = {}
        for item in matches:
            counts[item["path"]] = counts.get(item["path"], 0) + 1
        return _dump_json(counts)
    return _dump_json(sorted({item["path"] for item in matches}))


def _normalize_virtual_path(path: str) -> str:
    """将用户提供的路径标准化为 Deep Agents 虚拟绝对路径。

    Description:
        兼容项目内相对路径、虚拟绝对路径和项目内真实绝对路径，并阻止路径逃逸到
        项目根目录之外。
    Args:
        path (str): 用户或模型传入的原始路径。
    Returns:
        str: 标准化后的虚拟绝对路径。
    """
    if not path:
        return "/"

    raw_path = Path(path).expanduser()
    if raw_path.is_absolute():
        try:
            relative_path = raw_path.resolve().relative_to(config.PROJECT_ROOT)
        except ValueError:
            return str(PurePosixPath(path))
        return f"/{relative_path.as_posix()}"

    normalized = PurePosixPath("/") / PurePosixPath(path)
    return str(normalized)


def _format_lines(content: str, start_line: int) -> str:
    """把文本内容格式化为带行号的输出。

    Description:
        将读取到的文件内容拆分为行，并用固定宽度行号前缀展示，方便模型准确引用。
    Args:
        content (str): 文件文本内容。
        start_line (int): 第一行内容对应的实际行号。
    Returns:
        str: 带行号的多行文本。
    """
    lines = content.splitlines()
    return "\n".join(
        f"{line_number:>6}\t{line}"
        for line_number, line in enumerate(lines, start=start_line)
    )


def _dump_json(value: object) -> str:
    """将工具结果序列化为中文环境友好的 JSON 字符串。

    Description:
        统一处理 Deep Agents backend 返回的结构化数据，避免中文路径或内容被转义。
    Args:
        value (object): 需要序列化的工具结果。
    Returns:
        str: 缩进后的 JSON 字符串。
    """
    return json.dumps(value, ensure_ascii=False, indent=2)


FILESYSTEM_TOOLS = [ls, read_file, write_file, edit_file, glob, grep]
