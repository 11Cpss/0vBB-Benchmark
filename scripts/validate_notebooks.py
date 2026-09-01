#!/usr/bin/env python3
"""Validate committed notebooks without executing detector-data workflows."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def validate(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{path}: expected notebook format 4")
    for index, cell in enumerate(notebook.get("cells", [])):
        if not cell.get("id"):
            raise ValueError(f"{path}: cell {index} has no stable ID")
        if cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None or cell.get("outputs"):
            raise ValueError(f"{path}: code cell {index} contains saved output")
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source, filename=f"{path}:cell-{index}")
        except SyntaxError as error:
            raise SyntaxError(f"{path}: invalid code cell {index}: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    notebooks: list[Path] = []
    for path in args.paths:
        notebooks.extend(sorted(path.rglob("*.ipynb")) if path.is_dir() else [path])
    for notebook in notebooks:
        validate(notebook)
        print(f"ok: {notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
