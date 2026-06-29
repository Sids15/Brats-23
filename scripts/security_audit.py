"""Pre-commit security audit (roadmap workflow rule).

Scans files for secrets, API keys, and machine-identifying information that must
never enter the git history of a public research toolkit. Run this BEFORE every
commit (see rules.md).

    python scripts/security_audit.py            # scan staged files (default)
    python scripts/security_audit.py --all      # scan all git-tracked files

Severity:
    HIGH  -> exit code 1 (block the commit): private keys, cloud keys, secret literals.
    WARN  -> reported only: absolute user paths and bare email addresses.

The script skips itself (its own patterns would otherwise self-match).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# (name, severity, compiled pattern). Patterns are written so they never match
# their own source text in this file.
RULES: list[tuple[str, str, re.Pattern[str]]] = [
    ("private_key_block", "HIGH",
     re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("aws_access_key", "HIGH", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("generic_secret_literal", "HIGH",
     re.compile(r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|access[_-]?key)\b"
                r"\s*[:=]\s*['\"][^'\"\n]{8,}['\"]")),
    ("bearer_token", "HIGH", re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}")),
    ("windows_user_path", "WARN", re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s'\"]+")),
    ("unix_home_path", "WARN", re.compile(r"/(?:home|Users)/[^/\s'\"]+")),
    ("email_address", "WARN",
     re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
]

# Binary / data extensions never worth scanning.
SKIP_EXT = {".nii", ".gz", ".png", ".jpg", ".jpeg", ".pdf", ".pt", ".pth", ".ckpt", ".pyc"}
SELF = Path(__file__).resolve()


def _git_files(staged: bool) -> list[Path]:
    if staged:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    else:
        cmd = ["git", "ls-files"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [Path(p) for p in out.stdout.splitlines() if p.strip()]


def scan_file(path: Path) -> list[tuple[int, str, str, str]]:
    findings: list[tuple[int, str, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return findings
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, severity, pattern in RULES:
            if pattern.search(line):
                findings.append((lineno, severity, name, line.strip()[:120]))
    return findings


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan for secrets / machine info before commit.")
    ap.add_argument("--all", action="store_true", help="Scan all tracked files (default: staged).")
    args = ap.parse_args()

    files = _git_files(staged=not args.all)
    high, warn = 0, 0
    for path in files:
        if path.resolve() == SELF or path.suffix in SKIP_EXT:
            continue
        for lineno, severity, name, snippet in scan_file(path):
            counter = "HIGH" if severity == "HIGH" else "WARN"
            if counter == "HIGH":
                high += 1
            else:
                warn += 1
            print(f"{severity:4} | {path}:{lineno} | {name} | {snippet}")

    scope = "all tracked" if args.all else "staged"
    print(f"\nscanned {len(files)} {scope} file(s): {high} HIGH, {warn} WARN")
    if high:
        print("BLOCKED: resolve HIGH findings before committing.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
