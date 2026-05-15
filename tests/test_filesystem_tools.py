from __future__ import annotations

from langraph_agent.tools.filesystem import glob, grep, ls, read_file


def test_ls_uses_deepagents_filesystem_backend() -> None:
    result = ls.invoke({"path": "/"})

    assert '"/README.md"' in result
    assert '"/langraph_agent/"' in result


def test_read_file_supports_relative_project_path() -> None:
    result = read_file.invoke({"file_path": "README.md", "limit": 1})

    assert "1\t# LangGraph ReAct Agent Study" in result


def test_glob_finds_project_files() -> None:
    result = glob.invoke({"pattern": "langraph_agent/tools/*.py", "path": "/"})

    assert '"/langraph_agent/tools/filesystem.py"' in result


def test_grep_searches_project_file_content() -> None:
    result = grep.invoke(
        {
            "pattern": "Deep Agents",
            "path": "/langraph_agent/tools",
            "glob": "*.py",
            "output_mode": "files_with_matches",
        }
    )

    assert '"/langraph_agent/tools/filesystem.py"' in result
