#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compare journals from two builds with the affect settings absent.
# Inputs: two build directories and one model store. Outputs: comparison lines and files.
# Exit codes: 0 pass, 1 comparison or run failure, 2 usage or build error.
set -u
set -o pipefail

if [ "$#" -ne 3 ]; then
    echo "usage: affect_identity.sh <build-on> <build-off> <models>" >&2
    exit 2
fi

build_on="$1"
build_off="$2"
models="$3"
for item in "$build_on/aotx_boot" "$build_on/aotx_journal" \
            "$build_off/aotx_boot" "$build_off/aotx_journal"; do
    if [ ! -x "$item" ]; then
        echo "affect_identity: the build program is not executable: $item" >&2
        exit 2
    fi
done
if [ ! -d "$models" ]; then
    echo "affect_identity: the model store is not a directory: $models" >&2
    exit 2
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/aotx-affect-identity-XXXXXX") || exit 2
trap 'rm -rf "$work"' EXIT HUP INT TERM
fail=0

echo "masked fields: boot id, wall clock, globaltimer, CARD record"
echo "dependent fields: state hash and tick duration"

run_one()
{
    local build="$1"
    local name="$2"
    local journal="$work/$name"
    local settings="$work/$name.settings"
    local feed="$work/$name.feed"
    local boot_dir
    local boot
    local boot_status
    mkdir -p "$journal" || return 1
    mkfifo "$feed" || return 1
    printf '%s\n' 'sample.temperature = 0' 'sample.seed = 7' \
        'derive.list = tokens,pages' >"$settings"
    "$build/aotx_boot" --settings "$settings" --journal "$journal" --models "$models" \
        --ticks 32 <"$feed" >"$work/$name.log" 2>&1 &
    boot=$!
    exec 9>"$feed"
    printf '%s\n' 'say name one color and nothing else' >&9
    exec 9>&-
    wait "$boot"
    boot_status=$?
    if [ "$boot_status" -ne 0 ]; then
        echo "affect_identity: the run $name ended with status $boot_status" >&2
        sed -n '1,160p' "$work/$name.log" >&2
        return 1
    fi
    if ! grep -qhs '"output_hash"' "$journal"/manifest/*.jsonl; then
        echo "affect_identity: the run $name completed no turn" >&2
        sed -n '1,160p' "$work/$name.log" >&2
        return 1
    fi
    boot_dir=$(find "$journal" -mindepth 1 -maxdepth 1 -type d \
        -name '[0-9a-f][0-9a-f]*' | sort | tail -1)
    if [ -z "$boot_dir" ]; then
        echo "affect_identity: the run $name made no boot directory" >&2
        return 1
    fi
    if ! "$build/aotx_journal" records "$boot_dir" >"$work/$name.dump" \
        2>"$work/$name.journal.log"; then
        echo "affect_identity: the journal dump of $name failed" >&2
        return 1
    fi
    for stream in tokens pages; do
        if [ ! -f "$boot_dir/$stream.jsonl" ]; then
            echo "affect_identity: $name has no $stream.jsonl" >&2
            return 1
        fi
        cp "$boot_dir/$stream.jsonl" "$work/$name.$stream.jsonl" || return 1
    done
    return 0
}

mask_dump()
{
    awk '
        / class=2 type=22 / { next }
        {
            sub(/boot=[0-9a-f]+/, "boot=MASKED")
            sub(/globaltimer=[0-9]+/, "globaltimer=MASKED")
            if ($0 ~ / type=1 / || $0 ~ / type=2 /) sub(/body=.*/, "body=MASKED")
            if ($0 ~ / type=3 / || $0 ~ / type=7 /) {
                at = index($0, "body=")
                if (at > 0) $0 = substr($0, 1, at + 4) "MASKED" substr($0, at + 21)
            }
            print
        }
    ' "$1" >"$2"
}

before_tokens()
{
    awk '/ type=14 / { exit } { print }' "$1"
}

class_a_counts()
{
    awk '{
        cls = ""; type = ""
        for (i = 1; i <= NF; i++) {
            if ($i ~ /^class=/) cls = substr($i, 7)
            if ($i ~ /^type=/) type = substr($i, 6)
        }
        if (cls == "1") print type
    }' "$1" | sort -n | uniq -c
}

first_gap()
{
    python3 - "$1" "$2" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as one:
    left = [json.loads(line) for line in one]
with open(sys.argv[2], encoding="utf-8") as two:
    right = [json.loads(line) for line in two]
for a, b in zip(left, right):
    if a != b:
        gap = abs(float(a.get("logprob", 0.0)) - float(b.get("logprob", 0.0)))
        print(f"first differing tick {min(a.get('tick', 0), b.get('tick', 0))}, "
              f"logit gap {gap:.9g}")
        break
else:
    if len(left) != len(right):
        row = left[len(right)] if len(left) > len(right) else right[len(left)]
        print(f"first differing tick {row.get('tick', 0)}, logit gap unavailable")
    else:
        print("first differing tick unavailable, logit gap unavailable")
PY
}

compare_pair()
{
    local one="$1"
    local two="$2"
    local label="$3"
    local dump_same=0
    local streams_same=0
    mask_dump "$work/$one.dump" "$work/$one.masked"
    mask_dump "$work/$two.dump" "$work/$two.masked"
    if cmp -s "$work/$one.masked" "$work/$two.masked"; then dump_same=1; fi
    if cmp -s "$work/$one.tokens.jsonl" "$work/$two.tokens.jsonl" \
       && cmp -s "$work/$one.pages.jsonl" "$work/$two.pages.jsonl"; then
        streams_same=1
    fi
    if [ "$dump_same" -eq 1 ] && [ "$streams_same" -eq 1 ]; then
        echo "$label: PASS byte-identical masked journal, tokens.jsonl, and pages.jsonl"
        return 0
    fi
    if [ "$dump_same" -eq 1 ]; then
        echo "$label: FAIL a derived stream differs with an identical journal" >&2
        return 1
    fi
    before_tokens "$work/$one.masked" >"$work/$one.before"
    before_tokens "$work/$two.masked" >"$work/$two.before"
    if ! cmp -s "$work/$one.before" "$work/$two.before"; then
        echo "$label: FAIL the masked journals differ before the first token record" >&2
        return 1
    fi
    class_a_counts "$work/$one.masked" >"$work/$one.class-a"
    class_a_counts "$work/$two.masked" >"$work/$two.class-a"
    if ! cmp -s "$work/$one.class-a" "$work/$two.class-a"; then
        echo "$label: FAIL a class A record kind or count differs" >&2
        return 1
    fi
    first_gap "$work/$one.tokens.jsonl" "$work/$two.tokens.jsonl"
    echo "$label fallback: equal class A kinds and counts"
    echo "$label: PASS by class A kind-and-count fallback"
    return 0
}

run_one "$build_on" control-one || exit 1
run_one "$build_on" control-two || exit 1
compare_pair control-one control-two "same-build control" || fail=1

run_one "$build_on" pair-on || exit 1
run_one "$build_off" pair-off || exit 1
compare_pair pair-on pair-off "cross-build pair" || fail=1

exit "$fail"
