#!/usr/bin/env python3
"""
Saccade API Layering Audit.

Scans src/saccade/ for cross-boundary import violations:
- saccade.perception.* importing redis/chromadb/fastapi directly
  (should go through saccade.storage / saccade.api abstractions)
- Non-lazy circular imports between tracking ↔ temporal_yolo

Currently warning-only (exit 0). Will become blocking once existing violations
are resolved.

Usage:
    python scripts/tools/check_api_layers.py
"""
# status: stable

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PY_TARGET_DIR = os.path.join(ROOT, "src", "saccade")

WARN_YELLOW = "\033[1;33m"
BOLD_YELLOW = "\033[1;33m"
RESET = "\033[0m"

FORBIDDEN_TOP_LEVEL = {"redis", "chromadb", "fastapi"}


class ImportLayerVisitor(ast.NodeVisitor):
    def __init__(self, filename, lines, rel_path):
        self.filename = filename
        self.lines = lines
        self.rel_path = rel_path
        self.violations = []
        self._func_depth = 0
        self._in_tracking = "tracking" in rel_path.split(os.sep)
        self._in_temporal_yolo = "temporal_yolo" in rel_path.split(os.sep)

    def visit_FunctionDef(self, node):
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    def visit_AsyncFunctionDef(self, node):
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    def visit_ClassDef(self, node):
        self._func_depth += 1
        self.generic_visit(node)
        self._func_depth -= 1

    def _is_module_level(self):
        return self._func_depth == 0

    def _check_top_level_package(self, module_path, lineno):
        if not self._is_module_level():
            return
        top = module_path.split(".")[0]
        if top in FORBIDDEN_TOP_LEVEL:
            line_content = (
                self.lines[lineno - 1].strip() if lineno - 1 < len(self.lines) else ""
            )
            self.violations.append(
                (
                    lineno,
                    f"direct import of '{module_path}' in perception layer",
                    line_content,
                )
            )

    def _check_circular(self, module_path, lineno):
        if not self._is_module_level():
            return
        if self._in_tracking and module_path.startswith(
            "saccade.perception.temporal_yolo"
        ):
            line_content = (
                self.lines[lineno - 1].strip() if lineno - 1 < len(self.lines) else ""
            )
            self.violations.append(
                (lineno, "tracking → temporal_yolo (circular dependency)", line_content)
            )
        if self._in_temporal_yolo and module_path.startswith(
            "saccade.perception.tracking"
        ):
            line_content = (
                self.lines[lineno - 1].strip() if lineno - 1 < len(self.lines) else ""
            )
            self.violations.append(
                (lineno, "temporal_yolo → tracking (circular dependency)", line_content)
            )

    def visit_Import(self, node):
        for alias in node.names:
            self._check_top_level_package(alias.name, node.lineno)
            self._check_circular(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module is not None:
            self._check_top_level_package(node.module, node.lineno)
            self._check_circular(node.module, node.lineno)
        self.generic_visit(node)


def check_file(filepath):
    rel_path = os.path.relpath(filepath)
    parts = rel_path.split(os.sep)

    if "perception" not in parts:
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        lines = code.splitlines()
        tree = ast.parse(code, filepath)
    except Exception as e:
        print(f"  [Error] Failed to parse {rel_path}: {e}", file=sys.stderr)
        return []

    visitor = ImportLayerVisitor(filepath, lines, rel_path)
    visitor.visit(tree)
    return [
        (rel_path, lineno, msg, content) for lineno, msg, content in visitor.violations
    ]


def main():
    print(f"{WARN_YELLOW}── API layering audit (warning-only){RESET}")
    all_violations = []

    for root, _, files in os.walk(PY_TARGET_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                violations = check_file(filepath)
                all_violations.extend(violations)

    if all_violations:
        print(
            f"\n{BOLD_YELLOW}! Found {len(all_violations)} API layering concern(s):{RESET}"
        )
        for filepath, lineno, msg, content in all_violations:
            print(
                f"  {WARN_YELLOW}{filepath}:{lineno}{RESET} — {BOLD_YELLOW}{msg}{RESET}"
            )
            print(f"    Code: {content}")
        print(
            f"\n{WARN_YELLOW}Note: This check is currently warning-only. It will become blocking{RESET}"
        )
        print(f"{WARN_YELLOW}once existing violations are resolved.{RESET}")

    print(f"{WARN_YELLOW}── API layering audit done{RESET}")
    sys.exit(0)


if __name__ == "__main__":
    main()
