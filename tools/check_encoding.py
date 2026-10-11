#!/usr/bin/env python
"""Encoding guard for every tracked file.

Motivation: this project once lost 208 lines / 540 characters from README.md
because a shell round-trip decoded a UTF-8 file as GBK and wrote it back as
UTF-8 (see the work log, section 4.6 and section 7 item 20). This check makes
that entire class of damage impossible to merge unnoticed.

Every tracked file must be:
  1. valid UTF-8 (strict, no replacement decoding)
  2. free of a UTF-8 BOM
  3. free of U+FFFD REPLACEMENT CHARACTER
  4. free of classic mojibake signatures
  5. free of lone CR line endings

Exception for rule 2: PowerShell and batch scripts are allowed to carry a
BOM. Windows PowerShell 5.1 decodes a BOM-less .ps1 as ANSI, which turns any
non-ASCII string in the script into mojibake, so the BOM is a requirement
there rather than a defect.

Usage:

    python tools/check_encoding.py            # check tracked files
    python tools/check_encoding.py --all      # walk the tree instead
    python tools/check_encoding.py --quiet    # only print failures

Exit code 0 when clean, 1 when any file fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Make output survive consoles whose default encoding is not UTF-8 (this
# project's development console is GBK, where printing a Latin-1 accented
# character raises UnicodeEncodeError).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - very old Python
        pass

ROOT = Path(__file__).resolve().parent.parent

# Directories never worth scanning (build products, VCS metadata, caches).
SKIP_DIRS = {".git", "__pycache__", "build", "dist", "out", ".venv", "venv"}
SKIP_SUFFIXES = {
    ".pyc", ".pyo", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".gz",
}

# A BOM is legitimate in script formats that old Windows hosts read as ANSI.
BOM_ALLOWED_SUFFIXES = {".ps1", ".psm1", ".bat", ".cmd"}

# Signatures that cannot occur in healthy source or documentation. Both the
# markers and the labels are written as escape sequences / plain ASCII, so this
# file never matches its own rules.
MOJIBAKE: tuple[tuple[str, str], ...] = (
    ("\ufffd", "U+FFFD REPLACEMENT CHARACTER (lossy decode signature)"),
    ("\u9535\u65a4\u62f7", "GBK double-encoded garbage (kua-jin-kao)"),
    ("\u00ef\u00bb\u00bf", "UTF-8 BOM misread as Latin-1"),
    ("\u00e3\u20ac", "UTF-8 CJK bytes misread as Latin-1 (euro signature)"),
    ("\u00e4\u00b8", "UTF-8 CJK bytes misread as Latin-1 (cedilla signature)"),
    ("\u00e6\u02dc", "UTF-8 CJK bytes misread as Latin-1 (tilde signature)"),
    ("\u00e5\u00a4", "UTF-8 CJK bytes misread as Latin-1 (currency signature)"),
    ("\u00e8\u00af", "UTF-8 CJK bytes misread as Latin-1 (macron signature)"),
    ("\u00e7\u0161", "UTF-8 CJK bytes misread as Latin-1 (caron signature)"),
)


def tracked_files() -> list[Path]:
    """Files git tracks, which is exactly the set that gets published."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return walk_files()
    names = [n for n in out.decode("utf-8", "surrogateescape").split("\0") if n]
    return [ROOT / n for n in names] if names else walk_files()


def walk_files() -> list[Path]:
    return [
        p
        for p in sorted(ROOT.rglob("*"))
        if p.is_file() and not any(part in SKIP_DIRS for part in p.parts)
    ]


def check(path: Path) -> list[str]:
    """Return a list of human-readable problems; empty means clean."""
    if path.suffix.lower() in SKIP_SUFFIXES:
        return []

    try:
        raw = path.read_bytes()
    except OSError as exc:
        return [f"cannot read: {exc}"]

    problems: list[str] = []

    if raw.startswith(b"\xef\xbb\xbf"):
        if path.suffix.lower() not in BOM_ALLOWED_SUFFIXES:
            problems.append("has a UTF-8 BOM")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(f"not valid UTF-8: {exc}")
        text = raw.decode("utf-8", errors="replace")

    for marker, label in MOJIBAKE:
        count = text.count(marker)
        if count:
            problems.append(f"{label} x{count}")

    lone_cr = text.count("\r") - text.count("\r\n")
    if lone_cr:
        problems.append(f"lone CR line ending x{lone_cr}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="walk the tree")
    parser.add_argument("--quiet", action="store_true", help="only show failures")
    args = parser.parse_args()

    files = walk_files() if args.all else tracked_files()
    failures = 0

    for path in files:
        problems = check(path)
        if not problems:
            continue
        failures += 1
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"FAIL  {shown}")
        for problem in problems:
            print(f"        - {problem}")

    if not args.quiet:
        print()
        print(f"checked {len(files)} file(s); {failures} with problems")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
