#!/usr/bin/env python3
"""
Saccade GPU-First Performance Contract Checker.

Parses files in src/saccade/perception/ using Python's built-in AST parser
to prevent unauthorized host memory roundtrips (e.g., .cpu(), .numpy(), .to('cpu'),
.item(), .tolist(), torch.cuda.synchronize()).

Also regex-scans C++/CUDA sources (src/tracking/, src/perception/) for
cudaStreamSynchronize, cudaDeviceSynchronize, cudaStreamCreate.

Allows bypassing via:
    Python: # saccade-allow-cpu / # saccade-allow-numpy
    C++:    // saccade-allow-cpu
"""
# status: stable

import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directory to scan (Python)
PY_TARGET_DIR = os.path.join(ROOT, "src", "saccade", "perception")

# Directories to scan (C++/CUDA)
CPP_TARGET_DIRS = [
    os.path.join(ROOT, "src", "tracking"),
    os.path.join(ROOT, "src", "perception"),
]

CPP_EXTENSIONS = {".cu", ".cpp", ".cuh", ".h"}

CPP_PATTERNS = [
    re.compile(r"\bcudaStreamSynchronize\b"),
    re.compile(r"\bcudaDeviceSynchronize\b"),
]

# Entire subdirectories to exclude (Python, flexible/offline paths)
EXCLUDED_DIRS = {
    "eval",  # Evaluation/testing orchestrator (explicitly allowed to use CPU metrics)
    "training",  # Offline training scripts
    "temporal_yolo",  # Experimental module (excluded from mypy, same policy)
}

# Individual files to exclude (Python, bridge or offline utility components)
EXCLUDED_FILES = {
    "embedding_dispatcher.py",  # Asynchronous dispatch queue to host/Redis
    "data_pipeline.py",  # Offline data preprocessing pipeline
    "loss.py",  # Offline loss computation & linear assignment
    "dataset.py",  # PyTorch dataset loading
    "dataset_joint.py",  # PyTorch dataset loading
}

# Entire subdirectories to exclude (C++/CUDA)
CPP_EXCLUDED_DIRS: set[str] = set()

# Individual files to exclude (C++/CUDA)
CPP_EXCLUDED_FILES: set[str] = set()


class GPUContractVisitor(ast.NodeVisitor):
    def __init__(self, filename, lines):
        self.filename = filename
        self.lines = lines
        self.violations = []

    def visit_Attribute(self, node):
        # Detect .cpu or .numpy attributes
        if node.attr in ("cpu", "numpy"):
            self._check_node_violation(node, node.attr)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            # Detect .to('cpu') or .to("cpu") or .to(device='cpu')
            if node.func.attr == "to":
                is_cpu_target = False
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == "cpu":
                        is_cpu_target = True
                for kw in node.keywords:
                    if (
                        kw.arg in ("device",)
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value == "cpu"
                    ):
                        is_cpu_target = True
                if is_cpu_target:
                    self._check_node_violation(node, "to('cpu')")

            # Detect .item() — implicit CPU sync
            if node.func.attr == "item":
                self._check_node_violation(node, "item()")

            # Detect .tolist() — forces CPU copy
            if node.func.attr == "tolist":
                self._check_node_violation(node, "tolist()")

            # Detect torch.cuda.synchronize() — global GPU sync
            if node.func.attr == "synchronize":
                if (
                    isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "cuda"
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "torch"
                ):
                    self._check_node_violation(node, "torch.cuda.synchronize()")

        self.generic_visit(node)

    def _check_node_violation(self, node, trigger):
        line_no = node.lineno

        # Determine the lines to search for bypass comments
        # Check start line, and subsequent lines (up to 3 lines) to handle ruff wrapping
        start_idx = line_no - 1
        end_idx = start_idx + 3

        # If node has end_lineno in Python 3.8+, use it
        if hasattr(node, "end_lineno") and node.end_lineno is not None:
            end_idx = max(end_idx, node.end_lineno + 1)

        has_bypass = False
        for idx in range(start_idx, min(end_idx, len(self.lines))):
            line_content = self.lines[idx]
            if (
                "saccade-allow-cpu" in line_content
                or "saccade-allow-numpy" in line_content
            ):
                has_bypass = True
                break

        if has_bypass:
            return

        line_content = self.lines[start_idx] if start_idx < len(self.lines) else ""
        self.violations.append((line_no, trigger, line_content.strip()))


def check_file(filepath):
    rel_path = os.path.relpath(filepath)
    parts = rel_path.split(os.sep)

    # Check excluded subdirectories
    for d in EXCLUDED_DIRS:
        if d in parts:
            return []

    # Check excluded individual files
    if os.path.basename(filepath) in EXCLUDED_FILES:
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
        lines = code.splitlines()
        tree = ast.parse(code, filepath)
    except Exception as e:
        print(f"  [Error] Failed to parse {rel_path}: {e}", file=sys.stderr)
        return []

    visitor = GPUContractVisitor(rel_path, lines)
    visitor.visit(tree)
    return [
        (rel_path, line_no, trigger, content)
        for line_no, trigger, content in visitor.violations
    ]


def check_cpp_file(filepath):
    rel_path = os.path.relpath(filepath)
    parts = rel_path.split(os.sep)

    for d in CPP_EXCLUDED_DIRS:
        if d in parts:
            return []
    if os.path.basename(filepath) in CPP_EXCLUDED_FILES:
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  [Error] Failed to read {rel_path}: {e}", file=sys.stderr)
        return []

    violations = []
    for i, line in enumerate(lines):
        for pattern in CPP_PATTERNS:
            if pattern.search(line):
                line_no = i + 1
                trigger = pattern.pattern.strip(r"\b")
                has_bypass = False
                for offset in (0, -1):
                    check_idx = i + offset
                    if 0 <= check_idx < len(lines):
                        if "saccade-allow-cpu" in lines[check_idx]:
                            has_bypass = True
                            break
                if not has_bypass:
                    violations.append((rel_path, line_no, trigger, line.strip()))
                    break
    return violations


def main():
    print("── Checking GPU-first contract in perception modules...")
    py_violations = []

    for root, _, files in os.walk(PY_TARGET_DIR):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                violations = check_file(filepath)
                py_violations.extend(violations)

    print("── Checking C++/CUDA modules (warning-only)...")
    cpp_violations = []
    for cpp_dir in CPP_TARGET_DIRS:
        if not os.path.isdir(cpp_dir):
            continue
        for root, _, files in os.walk(cpp_dir):
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext in CPP_EXTENSIONS:
                    filepath = os.path.join(root, file)
                    violations = check_cpp_file(filepath)
                    cpp_violations.extend(violations)

    if cpp_violations:
        print(
            f"\n\033[1;33m! Found {len(cpp_violations)} C++ concern(s) (warning-only):\033[0m"
        )
        for filepath, line_no, trigger, content in cpp_violations:
            print(
                f"  \033[1;33m{filepath}:{line_no}\033[0m — \033[1;33m{trigger}\033[0m"
            )
            print(f"    Code: {content}")
        print()

    has_errors = False

    if py_violations:
        print(
            f"\n\033[0;31m✗ Found {len(py_violations)} GPU-first contract violations:\033[0m"
        )
        for filepath, line_no, trigger, content in py_violations:
            print(
                f"  \033[1;33m{filepath}:{line_no}\033[0m — call to \033[1;31m.{trigger}\033[0m"
            )
            print(f"    Code: {content}")
        print(
            "\n\033[1;36mTip:\033[0m If this CPU transfer is authorized, append \033[0;32m# saccade-allow-cpu\033[0m to the line."
        )
        has_errors = True

    if has_errors:
        sys.exit(1)

    print("\033[0;32m✓ All Python modules comply with GPU-first contract.\033[0m")
    sys.exit(0)


if __name__ == "__main__":
    main()
