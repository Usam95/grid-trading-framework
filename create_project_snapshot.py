#!/usr/bin/env python
"""
Create a single text snapshot of the project.

Features:
- Walks the project directory tree.
- Excludes directories (default + CLI).
- Excludes files by basename anywhere and/or by relative path from root (default + CLI).
- Includes files by extension (default + CLI) OR (optionally) by an allowlist of filenames/paths.
- Allows defining the allowlist directly in code (lists below), not only via CLI.
- Prints the included files list to stdout (with relative paths).

Modes:
- Default (extensions mode): includes files whose extension is in DEFAULT_INCLUDED_EXTENSIONS.
- Allowlist mode: if any of these are non-empty:
    - DEFAULT_ONLY_FILENAMES
    - DEFAULT_ONLY_PATHS
    - --only-filenames
    - --only-paths
  then ONLY those specified files are included (union of filename and path allowlists).

Examples:
  python create_project_snapshot.py
  python create_project_snapshot.py --only-filenames "CMakeLists.txt,Dockerfile"
  python create_project_snapshot.py --only-paths "src/main.py,core/config/settings.yaml"
"""

import os
import argparse
from pathlib import Path
from typing import Iterable, Set, Tuple


# ----------------------------
# Exclusions (directories)
# ----------------------------
DEFAULT_EXCLUDED_DIRS: Set[str] = {
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
    "output",
    "nlohmann",
    "docs",
    "historical_data",
    "runs"
#    "core",
#    "config",
}

# ----------------------------
# Exclusions (files)
# ----------------------------
# Exclude by *basename* anywhere in the tree
DEFAULT_EXCLUDED_FILES: Set[str] = {
    "create_project_snapshot.py",
    "project_snapshot.txt",
}

# Exclude by *relative path* from root (POSIX style: folder/file.ext)
DEFAULT_EXCLUDED_PATHS: Set[str] = {
    # Example:
    # "core/generated/big.json",
}

# ----------------------------
# Inclusion by extension (default mode)
# ----------------------------
DEFAULT_INCLUDED_EXTENSIONS: Set[str] = {
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
    ".h",
    ".hpp",
    ".c",
    ".cpp",
    ".arxml",
}

# ----------------------------
# Inclusion allowlist (IN CODE)
# ----------------------------
# If you want "ONLY these files", add them here.
# Any non-empty value here switches the script into allowlist mode.
DEFAULT_ONLY_FILENAMES: Set[str] = {
    # Examples:
    # "CMakeLists.txt",
    # "Dockerfile",
#    "utils.py",
#    "account_state.py",
#    "execution.py",
#    "user_stream.py",
#    "runtime.py",
#    "live_trade.py",
#    "strategy_builder.py",
#    "live_trade.yml",
#    "trading_config.py",
#    "equity_tracker.py"

}

"""Included files:
--------------------------------------------------------------------------------
app/__init__.py
app/grid_research.py
app/live_trade.py
app/main.py
app/simple_grid_backtest.py
backtest/__init__.py
backtest/engine.py
config/__init__.py
config/grid_run.yml
config/live_trade.yml
conftest.py
core/__init__.py
core/engine_actions.py
core/execution/bootstrap.py
core/execution/constraints.py
core/execution/reservations.py
core/models.py
core/research/grid_search.py
core/results/__init__.py
core/results/benchmarks.py
core/results/metrics.py
core/results/models.py
core/results/repository.py
core/results/summary.py
core/results/trade_builder.py
core/strategy/__init__.py
core/strategy/base.py
core/strategy/grid_strategy_dynamic.py
core/strategy/grid_strategy_simple.py
core/strategy/policies/filter.py
core/strategy/policies/range.py
core/strategy/policies/recenter.py
core/strategy/policies/sltp.py
core/strategy/policies/space.py
environment.yml
infra/__init__.py
infra/binance_downloader.py
infra/config/__init__.py
infra/config/binance_live_data_config.py
infra/config/data_config.py
infra/config/engine_config.py
infra/config/logging_config.py
infra/config/research_config.py
infra/config/run_config.py
infra/config/strategy_base.py
infra/config/strategy_grid.py
infra/config_loader.py
infra/data_source.py
infra/exchange/__init__.py
infra/exchange/base.py
infra/exchange/binance_spot.py
infra/indicators.py
infra/logging_setup.py
infra/marketdata/__init__.py
infra/marketdata/binance_kline_stream.py
infra/marketdata/binance_user_stream.py
infra/secrets.py
infra/splits.py
"""


DEFAULT_ONLY_PATHS: Set[str] = {
    # Examples (relative to --root):
    # "src/main.py",
    # "core/config/settings.yaml",
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

    # Extension inclusion mode (used only if NOT in allowlist mode)
    parser.add_argument(
        "--include-ext",
        type=str,
        default="",
        help=(
            "Comma-separated list of file extensions to include. "
            "If empty, a sensible default set is used. Example: .py,.md,.yaml\n"
            "Ignored in allowlist mode."
        ),
    )

    # Exclusions
    parser.add_argument(
        "--exclude-dirs",
        type=str,
        default="",
        help=(
            "Comma-separated list of directory names to exclude "
            "(in addition to the default list). Example: .git,venv,logs"
        ),
    )
    parser.add_argument(
        "--exclude-files",
        type=str,
        default="",
        help=(
            "Comma-separated list of files to exclude. Each entry can be either:\n"
            "  - a filename (basename) like 'secrets.json' (excluded anywhere), OR\n"
            "  - a relative path from --root like 'core/generated/big.json'.\n"
            "Paths can use '/' or '\\\\' (they will be normalized)."
        ),
    )

    # Allowlist mode (CLI)
    parser.add_argument(
        "--only-filenames",
        type=str,
        default="",
        help=(
            "Comma-separated list of filenames (basenames) to include (anywhere in tree). "
            "If set (or if --only-paths is set), switches to allowlist mode."
        ),
    )
    parser.add_argument(
        "--only-paths",
        type=str,
        default="",
        help=(
            "Comma-separated list of relative paths (from --root) to include exactly. "
            "If set (or if --only-filenames is set), switches to allowlist mode."
        ),
    )

    # Printing control
    parser.add_argument(
        "--print-files",
        action="store_true",
        help="Print the list of included files (relative paths) before writing snapshot.",
    )

    return parser.parse_args()


def parse_extension_list(ext_list: str, default: Set[str]) -> Set[str]:
    if not ext_list.strip():
        return set(default)
    result: Set[str] = set()
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
        return set(default)
    result = set(default)
    for raw in dir_list.split(","):
        d = raw.strip()
        if d:
            result.add(d)
    return result


def parse_exclude_files_list(
    raw_list: str,
    default_files: Set[str],
    default_paths: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """
    Returns: (excluded_basenames, excluded_relative_paths)

    - If an entry contains a path separator, treat it as a relative path.
    - Otherwise treat it as a basename.
    """
    excluded_files = set(default_files)
    excluded_paths = set(default_paths)

    if not raw_list.strip():
        return excluded_files, excluded_paths

    for raw in raw_list.split(","):
        item = raw.strip()
        if not item:
            continue

        item_norm = item.replace("\\", "/").lstrip("./")

        if "/" in item_norm:
            excluded_paths.add(item_norm)
        else:
            excluded_files.add(item_norm)

    return excluded_files, excluded_paths


def parse_only_lists(only_filenames: str, only_paths: str) -> Tuple[Set[str], Set[str]]:
    """
    Returns: (allowed_basenames, allowed_relative_paths)
    """
    allowed_files: Set[str] = set()
    allowed_paths: Set[str] = set()

    if only_filenames.strip():
        for raw in only_filenames.split(","):
            name = raw.strip()
            if name:
                allowed_files.add(name)

    if only_paths.strip():
        for raw in only_paths.split(","):
            p = raw.strip()
            if not p:
                continue
            p_norm = p.replace("\\", "/").lstrip("./")
            allowed_paths.add(p_norm)

    return allowed_files, allowed_paths


def iter_files(
    root: Path,
    include_exts: Set[str],
    exclude_dirs: Set[str],
    exclude_files: Set[str],
    exclude_paths: Set[str],
    allow_only_files: Set[str],
    allow_only_paths: Set[str],
) -> Iterable[Path]:
    """
    Yield all files under root that match inclusion rules and are not excluded.

    Allowlist mode: if allow_only_files or allow_only_paths is non-empty,
    include ONLY those basenames/relative paths.
    Otherwise: include by extension (include_exts).
    """
    root = root.resolve()
    allowlist_mode = bool(allow_only_files or allow_only_paths)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]

        for fname in filenames:
            # Exclude by basename anywhere
            if fname in exclude_files:
                continue

            fpath = (Path(dirpath) / fname).resolve()

            # Must be under root
            try:
                rel_posix = fpath.relative_to(root).as_posix()
            except ValueError:
                continue

            # Exclude by relative path
            if rel_posix in exclude_paths:
                continue

            if allowlist_mode:
                if rel_posix in allow_only_paths or fname in allow_only_files:
                    yield fpath
                continue

            # Default mode: include by extension
            ext = fpath.suffix

            # Special case: Dockerfile has no suffix
            if not ext and fname.lower() == "dockerfile":
                ext = ".Dockerfile"

            if include_exts and ext not in include_exts:
                continue

            yield fpath


def print_included_files(root: Path, files: Iterable[Path]) -> None:
    root = root.resolve()
    file_list = []
    for f in files:
        try:
            file_list.append(f.resolve().relative_to(root).as_posix())
        except ValueError:
            file_list.append(str(f))
    file_list.sort()

    print("\nIncluded files:")
    print("-" * 80)
    for p in file_list:
        print(p)
    print("-" * 80)
    print(f"Total included: {len(file_list)}\n")


def write_snapshot(root: Path, output: Path, files: Iterable[Path]) -> None:
    root = root.resolve()
    output = output.resolve()

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", errors="replace") as out_f:
        out_f.write("# Project Snapshot\n")
        out_f.write(f"# Root: {root}\n\n")

        for fpath in sorted(files):
            try:
                rel_path = fpath.relative_to(root)
            except ValueError:
                rel_path = fpath

            out_f.write("\n" + "=" * 80 + "\n")
            out_f.write(f"FILE: {rel_path}\n")
            out_f.write("=" * 80 + "\n\n")

            try:
                with fpath.open("r", encoding="utf-8", errors="replace") as in_f:
                    content = in_f.read()
            except Exception as e:
                out_f.write(f"<< ERROR READING FILE: {e} >>\n")
                continue

            out_f.write(content)
            if not content.endswith("\n"):
                out_f.write("\n")
            out_f.write("\n")

    print(f"Snapshot written to: {output}")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)

    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")

    include_exts = parse_extension_list(args.include_ext, DEFAULT_INCLUDED_EXTENSIONS)
    exclude_dirs = parse_dir_list(args.exclude_dirs, DEFAULT_EXCLUDED_DIRS)
    exclude_files, exclude_paths = parse_exclude_files_list(
        args.exclude_files,
        DEFAULT_EXCLUDED_FILES,
        DEFAULT_EXCLUDED_PATHS,
    )

    # Allowlist from CLI
    cli_only_files, cli_only_paths = parse_only_lists(args.only_filenames, args.only_paths)

    # Allowlist from code (defaults) + CLI (union)
    only_files = set(DEFAULT_ONLY_FILENAMES) | set(cli_only_files)
    only_paths = set(DEFAULT_ONLY_PATHS) | set(cli_only_paths)

    files = list(
        iter_files(
            root=root,
            include_exts=include_exts,
            exclude_dirs=exclude_dirs,
            exclude_files=exclude_files,
            exclude_paths=exclude_paths,
            allow_only_files=only_files,
            allow_only_paths=only_paths,
        )
    )

    mode = "ALLOWLIST (only specified filenames/paths)" if (only_files or only_paths) else "EXTENSIONS"
    print(f"Mode: {mode}")
    print(f"Found {len(files)} files to include in snapshot.")

    # Print included files if requested (or always, if you prefer)
    if args.print_files:
        print_included_files(root, files)

    write_snapshot(root, output, files)


if __name__ == "__main__":
    main()
