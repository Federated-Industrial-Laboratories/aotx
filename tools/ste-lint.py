#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ste-lint.py: the register gate for repository text.
#
# The gate examines prose in documents and comments in source files. It refuses a sentence
# that has more than 25 words, a paragraph that has more than 6 sentences, a word or pattern
# that ste-words.txt lists, and punctuation that is not ASCII.
#   ste-lint.py PATH [PATH...]   examine the given files or directories.
#   ste-lint.py --staged         examine staged content only.
#   ste-lint.py                  examine all git-tracked files under the current directory.
# Exit codes: 0 clean, 1 findings, 2 usage or environment error.

import re
import subprocess
import sys
from pathlib import Path

WORD_LIMIT = 25
SENTENCE_LIMIT = 6
PROSE_SUFFIXES = {".md", ".txt", ".head"}
CODE_SUFFIXES = {".cu", ".cuh", ".c", ".h", ".ptx", ".cmake", ".py", ".sh"}
SELF_EXEMPT = {"ste-lint.py", "ste-words.txt", "LICENSE", "NOTICE"}
NON_ASCII = re.compile(r"[\u2013\u2014\u2018\u2019\u201c\u201d\u2026]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`(\"'])")


def load_words(script_dir):
    path = script_dir / "ste-words.txt"
    if not path.is_file():
        print(f"ste-lint: word list not found at {path}", file=sys.stderr)
        sys.exit(2)
    patterns = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line))
    return patterns


def git_files(base, staged):
    cmd = ["git", "-C", str(base)]
    cmd += ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"] if staged else ["ls-files", "-z"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print("ste-lint: git command failed; give paths", file=sys.stderr)
        sys.exit(2)
    names = [p for p in out.stdout.split("\0") if p]
    if not staged:
        return [(n, (base / n).read_bytes()) for n in names if (base / n).is_file()]
    entries = []
    for rel in names:
        show = subprocess.run(["git", "-C", str(base), "show", f":{rel}"], capture_output=True)
        if show.returncode == 0:
            entries.append((rel, show.stdout))
    return entries


def prose_paragraphs(text):
    # A paragraph is a run of lines between blank lines. Fenced code, tables, headings and
    # front matter are not prose and are skipped. List items become their own sentences.
    paragraphs, current, start, fence = [], [], 0, False
    list_marker = re.compile(r"^([-*]|\d+\.)\s+")
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = not fence
            continue
        if fence or stripped.startswith("|") or stripped.startswith("#") or stripped == "---":
            continue
        if not stripped:
            if current:
                paragraphs.append((start, " ".join(current)))
            current = []
            continue
        # A list item is its own paragraph: a list is not prose, and an item is not a sentence
        # of the item before it.
        if list_marker.match(stripped):
            if current:
                paragraphs.append((start, " ".join(current)))
            current, start = [list_marker.sub("", stripped)], lineno
            continue
        if not current:
            start = lineno
        current.append(stripped)
    if current:
        paragraphs.append((start, " ".join(current)))
    return paragraphs


def comment_paragraphs(text, suffix):
    # Each comment block is one paragraph. Strings and code are not examined.
    marker = "#" if suffix in {".py", ".sh", ".cmake"} else None
    paragraphs, current, start = [], [], 0
    in_block = False
    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        piece = None
        if marker:
            if stripped.startswith(marker) and not stripped.startswith("#!"):
                body = stripped.lstrip("#")
                # An indented comment line is a table row (a usage line, a field list), not a
                # sentence of the prose before it; it stands as its own paragraph.
                if body.startswith("  ") and body.strip():
                    if current:
                        paragraphs.append((start, " ".join(current)))
                    paragraphs.append((lineno, body.strip()))
                    current = []
                    continue
                piece = body.strip()
        else:
            if in_block:
                end = stripped.find("*/")
                piece = (stripped[:end] if end != -1 else stripped).lstrip("*").strip()
                if end != -1:
                    in_block = False
                # A blank line inside a block comment ends a paragraph.
                if not piece:
                    if current:
                        paragraphs.append((start, " ".join(current)))
                    current = []
                    continue
            elif "//" in stripped or "/*" in stripped:
                opener = "//" if "//" in stripped and (
                    "/*" not in stripped or stripped.find("//") < stripped.find("/*")) else "/*"
                code, body = stripped.split(opener, 1)
                if opener == "/*":
                    end = body.find("*/")
                    piece = (body[:end] if end != -1 else body).strip()
                    in_block = end == -1
                else:
                    piece = body.strip()
                # A comment after code on the same line annotates that line. It is a table
                # entry, not a sentence of the comment above it, and stands alone.
                if code.strip():
                    if current:
                        paragraphs.append((start, " ".join(current)))
                    current = []
                    if piece:
                        paragraphs.append((lineno, piece))
                    continue
        if piece is None:
            if current:
                paragraphs.append((start, " ".join(current)))
            current = []
            continue
        if not current:
            start = lineno
        if piece:
            current.append(piece)
    if current:
        paragraphs.append((start, " ".join(current)))
    return paragraphs


def scan(name, data, patterns):
    findings = []
    suffix = Path(name).suffix.lower()
    if Path(name).name == "CMakeLists.txt":
        suffix = ".cmake"
    # A file which is neither prose nor code holds no register. The suffix comes first, so a
    # data file or a model file is not read as text.
    if suffix not in PROSE_SUFFIXES and suffix not in CODE_SUFFIXES:
        return findings
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [f"{name}: not UTF-8"]
    if suffix in PROSE_SUFFIXES:
        paragraphs = prose_paragraphs(text)
    else:
        paragraphs = comment_paragraphs(text, suffix)
    for lineno, line in enumerate(text.splitlines(), 1):
        m = NON_ASCII.search(line)
        if m:
            findings.append(f"{name}:{lineno}: punctuation that is not ASCII (U+{ord(m.group(0)):04X})")
    for start, paragraph in paragraphs:
        sentences = [s for s in SENTENCE_END.split(paragraph) if s.strip()]
        if len(sentences) > SENTENCE_LIMIT:
            findings.append(f"{name}:{start}: paragraph has {len(sentences)} sentences (limit {SENTENCE_LIMIT})")
        for sentence in sentences:
            words = len(sentence.split())
            if words > WORD_LIMIT:
                findings.append(f"{name}:{start}: sentence has {words} words (limit {WORD_LIMIT}): {sentence[:60]}")
        for pat in patterns:
            m = pat.search(paragraph)
            if m:
                findings.append(f"{name}:{start}: refused word or pattern '{m.group(0)}'")
    return findings


def skipped(path):
    # Version control data and build trees hold generated files that are not the repository's.
    parts = path.resolve().parts
    vendor = any(parts[i:i + 2] == ("ctrl", "vendor") for i in range(len(parts)))
    return vendor or any(part == ".git" or part.startswith("build") for part in path.parts)


def main():
    script_dir = Path(__file__).resolve().parent
    patterns = load_words(script_dir)
    base = Path.cwd()
    args = sys.argv[1:]
    if args == ["--staged"]:
        entries = git_files(base, staged=True)
    elif args:
        entries = []
        for a in args:
            p = Path(a)
            if not p.exists():
                print(f"ste-lint: no such path: {a}", file=sys.stderr)
                return 2
            files = [q for q in p.rglob("*") if q.is_file()] if p.is_dir() else [p]
            entries += [(str(q), q.read_bytes()) for q in files if not skipped(q)]
    else:
        entries = git_files(base, staged=False)
    findings = []
    for name, data in entries:
        if skipped(Path(name)):
            continue
        if Path(name).name in SELF_EXEMPT:
            continue
        findings += scan(name, data, patterns)
    for f in findings:
        print(f)
    if findings:
        print(f"ste-lint: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print(f"ste-lint: clean ({len(entries)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
