#!/usr/bin/env python
"""
Create a single text snapshot of the project.

- Walks the project directory tree.
- Skips common junk (e.g., .git, __pycache__, venvs, logs, large data folders).
- Includes only selected file types by default (.py, .txt, .md, .yaml, .yml, .toml, .ini, .json).
- Writes everything into one file with headers showing the file path.

Usage (from project root):

    python create_project_snapshot.py
    # or:
    python create_project_snapshot.py --root . --output project_snapshot.txt
"""

import os
import argparse
from pathlib import Path
from typing import Iterable, Set


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    ".env",
    "dist",
    "build",
    "logs",
    "data",
    "resultings",
    ".terraform",
    "tests", 
    "create_project_snapshot.py"
}

DEFAULT_INCLUDED_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".rst",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".json",
    ".sh",
    ".ps1",
    ".bat",
    ".dockerfile",
    ".Dockerfile",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single snapshot file containing all project sources."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory of the project (default: current directory).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="project_snapshot.txt",
        help="Output snapshot file (default: project_snapshot.txt).",
    )
    parser.add_argument(
        "--include-ext",
        type=str,
        default="",
        help=(
            "Comma-separated list of file extensions to include. "
            "If empty, a sensible default set is used. Example: .py,.md,.yaml"
        ),
    )
    parser.add_argument(
        "--exclude-dirs",
        type=str,
        default="",
        help=(
            "Comma-separated list of directory names to exclude "
            "(in addition to the default list). Example: .git,venv,logs"
        ),
    )
    return parser.parse_args()


def parse_extension_list(ext_list: str, default: Set[str]) -> Set[str]:
    if not ext_list.strip():
        return default
    result = set()
    for raw in ext_list.split(","):
        e = raw.strip()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        result.add(e)
    return result


def parse_dir_list(dir_list: str, default: Set[str]) -> Set[str]:
    if not dir_list.strip():
        return default
    result = set(default)
    for raw in dir_list.split(","):
        d = raw.strip()
        if d:
            result.add(d)
    return result


def iter_files(root: Path, include_exts: Set[str], exclude_dirs: Set[str]) -> Iterable[Path]:
    """Yield all files under root that match extensions and are not in excluded dirs."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Modify dirnames in-place to stop os.walk from descending into excluded dirs
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_dirs and not d.startswith(".git")
        ]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            # Decide by extension
            ext = fpath.suffix
            # Special cases: Dockerfile or similar without suffix
            if not ext and fname.lower() in {"dockerfile"}:
                ext = ".Dockerfile"

            if include_exts and ext not in include_exts:
                continue

            yield fpath


def write_snapshot(root: Path, output: Path, files: Iterable[Path]) -> None:
    """Write all files into the output snapshot with headers."""
    root = root.resolve()
    output = output.resolve()

    # Ensure output dir exists
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", errors="replace") as out_f:
        out_f.write("# Project Snapshot\n")
        out_f.write(f"# Root: {root}\n\n")

        for fpath in sorted(files):
            try:
                rel_path = fpath.relative_to(root)
            except ValueError:
                # Fallback if for some reason not under root
                rel_path = fpath

            header = "\n" + "=" * 80 + "\n"
            header += f"FILE: {rel_path}\n"
            header += "=" * 80 + "\n\n"
            out_f.write(header)

            try:
                with fpath.open("r", encoding="utf-8", errors="replace") as in_f:
                    content = in_f.read()
            except Exception as e:
                out_f.write(f"<< ERROR READING FILE: {e} >>\n")
                continue

            out_f.write(content)
            if not content.endswith("\n"):
                out_f.write("\n")
            out_f.write("\n")  # extra blank line between files

    print(f"Snapshot written to: {output}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)

    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    include_exts = parse_extension_list(args.include_ext, DEFAULT_INCLUDED_EXTENSIONS)
    exclude_dirs = parse_dir_list(args.exclude_dirs, DEFAULT_EXCLUDED_DIRS)

    files = list(iter_files(root, include_exts, exclude_dirs))
    print(f"Found {len(files)} files to include in snapshot.")

    write_snapshot(root, output, files)


if __name__ == "__main__":
    main()
