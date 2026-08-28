#!/usr/bin/env bash
# replay_test.sh: the replay gate. Each scenario runs the system, kills it with SIGKILL,
# restores it from the journal, and compares the state hash before and after. The first
# scenario feeds command lines, and the second asks the language model for a reply.
#   replay_test.sh <build dir> <journal dir> [model dir]
# The journal directory, and the directory beside it that ends with "-say", are removed first.
# Exit codes: 0 when every scenario that ran passed, 1 when one failed, 2 on usage.
set -u

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "usage: replay_test.sh <build dir> <journal dir> [model dir]" >&2
    exit 2
fi
build="$1"
journal="$2"
models="${3:-models}"
say_journal="${journal}-say"
fnv_basis="cbf29ce484222325"
fail=0
skipped=""

# Reads one field of the line that aotx_restore prints.
field() {
    sed -n "s/.*$1=\\([0-9a-f]*\\).*/\\1/p" <<<"$2"
}

# ---- the first scenario: command lines ----

# Sixty-four distinct lines, then the pipe stays open so the run is killed mid-flight.
feed_lines() {
    local i
    for i in $(seq 1 64); do
        printf 'line %03d %08x\n' "$i" $(( (i * 2654435761) & 0xffffffff ))
        sleep 0.02
    done
    sleep 5
}

scenario_lines() {
    local before after hash_before hash_after applied_before echoes drained last_tick bad=0
    rm -rf "$journal"
    mkdir -p "$journal"

    feed_lines | "$build/aotx_boot" --journal "$journal" >"$journal/run-1.log" 2>&1 &
    local boot=$!
    sleep 3
    kill -9 "$boot"
    wait "$boot" 2>/dev/null
    # The disk-side programs die with their parent and finish the published blocks first.
    sleep 1

    before=$("$build/aotx_restore" --journal "$journal" --summary) || {
        echo "replay_test: no restorable journal after the kill" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    applied_before=$(sed -n 's/.*replayed=\([0-9]*\).*/\1/p' <<<"$before")
    echo "lines before: $before"

    "$build/aotx_boot" --journal "$journal" --restore --ticks 20 </dev/null \
        >"$journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $journal/run-2.log" >&2
        return 1
    }

    after=$("$build/aotx_restore" --journal "$journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    echo "lines after:  $after"

    # A gate that can pass on an empty run is not a gate. The run must have applied the 64
    # lines and at least one clock record. The hash must have moved off the FNV-1a basis. The
    # echoes must be on disk. The drain must have written every block the device published.
    echoes=$(cat "$journal"/*/console.log 2>/dev/null | grep -c '^> line ' || true)
    drained=$(sed -n 's/^drain: blocks to \([0-9]*\).*/\1/p' "$journal/run-1.log" | head -1)
    last_tick=$(sed -n 's/.*last_tick=\([0-9]*\).*/\1/p' <<<"$before")
    echo "lines cases: 1 kill, 1 restore, $applied_before class A records replayed," \
         "$echoes echoes, blocks drained $drained, last tick $last_tick"
    [ "${applied_before:-0}" -ge 65 ] || { echo "replay_test: FAIL only $applied_before records applied before the kill" >&2; bad=1; }
    [ "$hash_before" != "$fnv_basis" ] || { echo "replay_test: FAIL the state hash is the empty basis" >&2; bad=1; }
    [ "$echoes" -eq 64 ] || { echo "replay_test: FAIL $echoes echoes in console.log, 64 expected" >&2; bad=1; }
    [ -n "$drained" ] && [ "$drained" -eq "$last_tick" ] || { echo "replay_test: FAIL drained blocks $drained differ from last complete tick $last_tick" >&2; bad=1; }
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    [ "$bad" -eq 0 ] && echo "replay_test: PASS lines, state_hash $hash_before"
    return "$bad"
}

# ---- the second scenario: a reply that a kill cuts short ----

# The kill falls in the middle of the reply. The comparison that follows reads the token
# records of both runs. This scenario runs only when the model directory holds a manifest
# with the language role, and reports a skip when it does not.

# Waits for the console to name the agent that takes a reply. The model load reads and hashes
# 5.3 GB, so the wait is 60 seconds long. A wait that ends without the name goes on anyway,
# and the checks that follow state what the run did.
wait_prompt() {
    local i
    for i in $(seq 1 600); do
        if grep -q 'conductor:' "$say_journal"/*/console.log 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# The line that the feeder reads. The pipe stays open after it, so the run is killed while
# the reply is still forming. The question asks for a long answer, because a short one ends
# before the kill and the restore then has nothing left to sample.
feed_say() {
    wait_prompt
    printf 'say count from one to one hundred, one number for each line\n'
    sleep 20
}

# Prints the first four fields of each token line, which are the token itself: the slot, the
# position, the token and the flags. A line above the tick limit is left out, because the
# journal of the killed run can hold part of a tick that never committed.
token_key() {
    awk -v limit="$2" '{
        tick = "";
        for (i = 1; i <= NF; i++) {
            if ($i ~ /^tick=/) { tick = substr($i, 6); }
        }
        if (tick + 0 <= limit + 0) { print $1, $2, $3, $4; }
    }' "$1"
}

scenario_say() {
    local before after hash_before hash_after boot_1 boot_2 restored tick_1 bad=0
    local keys made again refused held first
    rm -rf "$say_journal"
    mkdir -p "$say_journal"

    feed_say | "$build/aotx_boot" --journal "$say_journal" --models "$models" \
        >"$say_journal/run-1.log" 2>&1 &
    local boot=$!
    wait_prompt || echo "replay_test: the console did not name the agent in 60 seconds"
    sleep 2
    kill -9 "$boot"
    wait "$boot" 2>/dev/null
    sleep 1

    before=$("$build/aotx_restore" --journal "$say_journal" --summary) || {
        echo "replay_test: no restorable journal after the kill; see $say_journal/run-1.log" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    boot_1=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$before")
    tick_1=$(sed -n 's/.*last_tick=\([0-9]*\).*/\1/p' <<<"$before")
    echo "say before: $before"

    "$build/aotx_boot" --journal "$say_journal" --restore --ticks 300 --models "$models" \
        </dev/null >"$say_journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $say_journal/run-2.log" >&2
        return 1
    }

    after=$("$build/aotx_restore" --journal "$say_journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    boot_2=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$after")
    restored=$(field restore_of "$after")
    echo "say after:  $after"

    "$build/aotx_journal" tokens "$say_journal" --boot "$boot_1" >"$say_journal/tokens-1.txt" \
        2>"$say_journal/tokens-1.err" || { echo "replay_test: FAIL the first journal does not read" >&2; return 1; }
    "$build/aotx_journal" tokens "$say_journal" --boot "$boot_2" >"$say_journal/tokens-2.txt" \
        2>"$say_journal/tokens-2.err" || { echo "replay_test: FAIL the second journal does not read" >&2; return 1; }

    token_key "$say_journal/tokens-1.txt" "$tick_1" >"$say_journal/key-1.txt"
    keys=$(wc -l <"$say_journal/key-1.txt")
    awk '{ print $1, $2, $3, $4 }' "$say_journal/tokens-2.txt" | head -n "$keys" \
        >"$say_journal/key-2.txt"
    made=$(grep -c ' sampled=1 ' "$say_journal/tokens-1.txt" || true)
    again=$(tail -n +"$((keys + 1))" "$say_journal/tokens-2.txt" \
        | grep -c ' sampled=1 replayed=0' || true)
    # The two journals hold the records the device wrote again, so a record the device
    # refused leaves no mark in them. The report of the restored run states the count of
    # sequence opens the decode refused, and a replay that lands needs none. The first
    # token the model made on slot 0 must sit at the position after the records that were
    # applied again there.
    refused=$(sed -n 's/^restore: applied [0-9]* hash [0-9a-f]* refused \([0-9]*\).*/\1/p' \
        "$say_journal/run-2.log" | head -1)
    held=$(grep -c '^slot=0 .* replayed=1$' "$say_journal/tokens-2.txt" || true)
    first=$(grep -m1 '^slot=0 .* sampled=1 replayed=0$' "$say_journal/tokens-2.txt" \
        | sed -n 's/^slot=0 position=\([0-9]*\) .*/\1/p')

    echo "say cases: 1 kill, 1 restore, $keys token records in the replayed prefix," \
         "$made sampled before the kill, $again sampled after the restore point," \
         "$held applied again on slot 0, first new token at position ${first:-none}," \
         "refused ${refused:-not stated}"
    [ "$keys" -ge 1 ] || { echo "replay_test: FAIL the killed run wrote no token record" >&2; bad=1; }
    [ "$made" -ge 1 ] || { echo "replay_test: FAIL the killed run sampled no token" >&2; bad=1; }
    [ "$restored" = "$boot_1" ] || { echo "replay_test: FAIL the restored run names boot $restored and not $boot_1" >&2; bad=1; }
    if ! diff -u "$say_journal/key-1.txt" "$say_journal/key-2.txt" >"$say_journal/key.diff"; then
        echo "replay_test: FAIL the token records differ; see $say_journal/key.diff" >&2
        head -20 "$say_journal/key.diff" >&2
        bad=1
    fi
    [ "$again" -ge 1 ] || { echo "replay_test: FAIL no token was sampled after the restore point" >&2; bad=1; }
    if [ -z "$refused" ]; then
        echo "replay_test: FAIL the restore report states no refused count" >&2
        bad=1
    elif [ "$refused" -ne 0 ]; then
        echo "replay_test: FAIL the restored run refused $refused sequence opens" >&2
        bad=1
    fi
    if [ -z "$first" ]; then
        echo "replay_test: FAIL slot 0 sampled no token after the restore" >&2
        bad=1
    elif [ "$first" -ne "$held" ]; then
        echo "replay_test: FAIL the first token of slot 0 sits at position $first and $held records were applied again there" >&2
        bad=1
    fi
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    [ "$bad" -eq 0 ] && echo "replay_test: PASS say, state_hash $hash_before, tokens $keys"
    return "$bad"
}

# ---- both scenarios ----

scenario_lines || fail=1

if [ ! -f "$models/manifest.jsonl" ]; then
    skipped="say (no $models/manifest.jsonl)"
elif ! grep -q '"name":"language"' "$models/manifest.jsonl"; then
    skipped="say (the manifest holds no language role)"
else
    scenario_say || fail=1
fi

if [ -n "$skipped" ]; then
    echo "replay_test: scenarios applied 1, skipped 1: $skipped"
else
    echo "replay_test: scenarios applied 2, skipped 0"
fi
[ "$fail" -eq 0 ] || exit 1
exit 0
