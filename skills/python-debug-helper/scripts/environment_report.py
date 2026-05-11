from __future__ import annotations

import importlib.metadata
import platform
import sys


PACKAGES = [
    "langgraph",
    "langchain-core",
    "langchain-openai",
    "openai",
    "pyyaml",
]


def main() -> None:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        print(f"{package}: {version}")


if __name__ == "__main__":
    main()
