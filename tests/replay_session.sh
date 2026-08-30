#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# replay_session.sh: define the long-session arm of the replay check.
# Inputs: replay_test.sh functions and variables. Outputs: scenario_session. Not run alone.

session_journal="${journal}-session"

session_line() {
    local fill="$1" prefix count
    prefix="say Reply OK only. Context ${fill}: "
    count=$((2000 - ${#prefix}))
    printf '%s' "$prefix"
    printf '%*s' "$count" '' | tr ' ' "$fill"
    printf '\n'
}

feed_session() {
    session_line a
    wait_turns "$session_journal" 1 || return 0
    session_line b
    wait_turns "$session_journal" 2 || return 0
    session_line c
    wait_turns "$session_journal" 3 || return 0
    session_line d
    wait_turns "$session_journal" 4 || return 0
    session_line e
    wait_turns "$session_journal" 5 || return 0
    printf 'agent 0 pages 120\n'
    printf 'agent 0 compact\n'
    wait_turns "$session_journal" 6 || return 0
    printf 'agent 0 pages 160\n'
    session_line e
    wait_turns "$session_journal" 7 || return 0
    wait_killed "$session_journal"
}

scenario_session() {
    local before after hash_before hash_after boot_1 boot_2 tick_1 turns lines bad=0
    rm -rf "$session_journal"
    mkdir -p "$session_journal"
    feed_session | "$build/aotx_boot" --journal "$session_journal" --models "$models" \
        >"$session_journal/run-1.log" 2>&1 &
    local boot=$!
    wait_turns "$session_journal" 7 \
        || echo "replay_test: session made no turn after compaction in 360 seconds"
    kill -9 "$boot"
    : >"$session_journal/killed"
    wait "$boot" 2>/dev/null
    sleep 1
    before=$("$build/aotx_restore" --journal "$session_journal" --summary) || {
        echo "replay_test: no restorable session journal after the kill" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    boot_1=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$before")
    tick_1=$(sed -n 's/.*last_tick=\([0-9]*\).*/\1/p' <<<"$before")
    echo "session before: $before"
    "$build/aotx_boot" --journal "$session_journal" --restore --ticks 300 \
        --models "$models" </dev/null >"$session_journal/run-2.log" 2>&1 || {
        echo "replay_test: the session restore failed; see $session_journal/run-2.log" >&2
        return 1
    }
    after=$("$build/aotx_restore" --journal "$session_journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    boot_2=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$after")
    echo "session after:  $after"
    turns=$(wc -l <"$session_journal/manifest/$boot_1.jsonl")
    head -n "$turns" "$session_journal/manifest/$boot_2.jsonl" \
        >"$session_journal/manifest-2-head.jsonl"
    if ! diff -u "$session_journal/manifest/$boot_1.jsonl" \
        "$session_journal/manifest-2-head.jsonl" >"$session_journal/manifest.diff"; then
        echo "replay_test: FAIL the session manifests differ" >&2
        head -20 "$session_journal/manifest.diff" >&2
        bad=1
    fi
    lines=$(grep -h '"kind":"line"' "$session_journal"/*/transcript/0.jsonl \
        2>/dev/null | grep -c 'Context [abcde]:' || true)
    echo "session cases: 1 kill, 1 restore, $turns manifests, $lines long transcript lines," \
         "last tick $tick_1"
    [ "$turns" -ge 7 ] \
        || { echo "replay_test: FAIL only $turns session turns ended" >&2; bad=1; }
    [ "$lines" -ge 12 ] \
        || { echo "replay_test: FAIL only $lines long lines reached the transcript" >&2; bad=1; }
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL session state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    compare_turns "$session_journal" "$boot_1" "$boot_2" "session" || bad=1
    "$(dirname "${BASH_SOURCE[0]}")/replay_session_check.sh" \
        "$session_journal" "$boot_1" "$boot_2" \
        || bad=1
    [ "$bad" -eq 0 ] && echo "replay_test: PASS session, state_hash $hash_before"
    return "$bad"
}
