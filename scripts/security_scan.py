"""Lightweight repository secret scanner for CI.

The scanner only checks the current working tree, not git history. That keeps
CI focused on whether the proposed state is safe to publish while still
catching accidental API keys, GitHub PATs, and common private key blocks.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFAULT_EXCLUDES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "data/cache",
    "data/historical",
    "logs",
}

TOKEN_PATTERNS = {
    "github_pat": re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
    "minimax_key": re.compile(r"sk-cp-[A-Za-z0-9_-]{40,}"),
    "generic_sk": re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{32,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def _is_excluded(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    parts = set(path.relative_to(root).parts)
    return "__pycache__" in parts or any(
        rel == item or rel.startswith(item.rstrip("/") + "/") for item in DEFAULT_EXCLUDES
    )


def _iter_text_files(root: Path):
    for path in root.rglob("*"):
        if path.is_dir() or _is_excluded(path, root):
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
        except FileNotFoundError:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        yield path, text


def scan(root: Path) -> list[tuple[str, str, int]]:
    findings: list[tuple[str, str, int]] = []
    for path, text in _iter_text_files(root):
        rel = path.relative_to(root).as_posix()
        for name, pattern in TOKEN_PATTERNS.items():
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                findings.append((name, rel, line_no))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan current tree for committed secrets.")
    parser.add_argument("root", nargs="?", default=".", help="repository root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = scan(root)
    if findings:
        print("Potential secrets found:")
        for name, rel, line_no in findings:
            print(f"  - {name}: {rel}:{line_no}")
        return 1
    print("Secret scan passed: no high-confidence tokens found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
