#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Purpose: Compare exact turn fields and agent order after raw manifest chain checks.
# Owns: Parsed manifest rows and one digest for each input chain.
# Threading: One test process reads all agent and turn rows.
# Lifetime: One comparison of two complete manifest byte strings.

# Inputs: Two raw JSONL manifests. Output: Equality result. Exit: Import only.
import hashlib
import json
import re

FIELDS = ("agent", "turn", "input_hash", "output_hash", "tokens", "finish", "tool", "request")


def unique(pairs):
    row = {}
    for key, value in pairs:
        if key in row:
            raise ValueError("duplicate manifest key")
        row[key] = value
    return row


def manifest(raw):
    if not isinstance(raw, bytes) or (raw and not raw.endswith(b"\n")):
        raise ValueError("manifest requires complete byte lines")
    previous = "0" * 64
    agents, seen = {}, set()
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            raise ValueError("manifest line has no newline")
        row = json.loads(line.decode("utf-8"), object_pairs_hook=unique)
        if not isinstance(row, dict) or set(row) != set(FIELDS) | {"prev"}:
            raise ValueError("manifest row keys do not match")
        if row["prev"] != previous:
            raise ValueError("manifest chain does not match raw line bytes")
        for key in ("agent", "turn", "tokens", "request"):
            if type(row[key]) is not int or not 0 <= row[key] <= 0xffffffff:
                raise ValueError("manifest integer is invalid")
        for key in ("input_hash", "output_hash"):
            if not isinstance(row[key], str) or re.fullmatch("[0-9a-f]{16}", row[key]) is None:
                raise ValueError("manifest content hash is invalid")
        if any(not isinstance(row[key], str) for key in ("finish", "tool")):
            raise ValueError("manifest text field is invalid")
        identity = row["agent"], row["turn"]
        if identity in seen:
            raise ValueError("duplicate agent turn")
        seen.add(identity)
        agents.setdefault(row["agent"], []).append(tuple(row[key] for key in FIELDS))
        previous = hashlib.sha256(line).hexdigest()
    return agents


def same_turns(original, recovered):
    try:
        return manifest(original) == manifest(recovered)
    except (ValueError, TypeError):
        return False
