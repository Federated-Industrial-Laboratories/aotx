#!/usr/bin/env bash
# replay_test.sh: the replay gate. Runs the system, feeds it lines, kills it with SIGKILL,
# restores it from the journal, and compares the state hash before and after.
#   replay_test.sh <build dir> <journal dir>   the journal directory is removed first.
# Exit codes: 0 when the hashes are equal, 1 when they differ or a step fails, 2 on usage.
set -u

if [ "$#" -ne 2 ]; then
    echo "usage: replay_test.sh <build dir> <journal dir>" >&2
    exit 2
fi
build="$1"
journal="$2"
rm -rf "$journal"
mkdir -p "$journal"

# Sixty-four distinct lines, then the pipe stays open so the run is killed mid-flight.
feed_lines() {
    local i
    for i in $(seq 1 64); do
        printf 'line %03d %08x\n' "$i" $(( (i * 2654435761) & 0xffffffff ))
        sleep 0.02
    done
    sleep 5
}

feed_lines | "$build/aotx_boot" --journal "$journal" >"$journal/run-1.log" 2>&1 &
boot=$!
sleep 3
kill -9 "$boot"
wait "$boot" 2>/dev/null
# The disk-side programs die with their parent and finish the published blocks first.
sleep 1

before=$("$build/aotx_restore" --journal "$journal" --summary) || {
    echo "replay_test: no restorable journal after the kill" >&2
    exit 1
}
hash_before=$(sed -n 's/.*state_hash=\([0-9a-f]*\).*/\1/p' <<<"$before")
applied_before=$(sed -n 's/.*replayed=\([0-9]*\).*/\1/p' <<<"$before")
echo "before: $before"

"$build/aotx_boot" --journal "$journal" --restore --ticks 20 </dev/null >"$journal/run-2.log" 2>&1 || {
    echo "replay_test: the restore run failed; see $journal/run-2.log" >&2
    exit 1
}

after=$("$build/aotx_restore" --journal "$journal" --summary) || exit 1
hash_after=$(sed -n 's/.*restore_hash=\([0-9a-f]*\).*/\1/p' <<<"$after")
echo "after:  $after"

# A gate that can pass on an empty run is not a gate. The run must have applied the 64 lines
# and at least one clock record. The hash must have moved off the FNV-1a basis. The echoes
# must be on disk. The drain must have written every block the device published.
fnv_basis="cbf29ce484222325"
echoes=$(cat "$journal"/*/console.log 2>/dev/null | grep -c '^> line ' || true)
drained=$(sed -n 's/^drain: blocks to \([0-9]*\).*/\1/p' "$journal/run-1.log" | head -1)
last_tick=$(sed -n 's/.*last_tick=\([0-9]*\).*/\1/p' <<<"$before")
echo "cases applied: 1 kill, 1 restore, $applied_before class A records replayed, $echoes echoes, blocks drained $drained, last tick $last_tick"
fail=0
[ "${applied_before:-0}" -ge 65 ] || { echo "replay_test: FAIL only $applied_before records applied before the kill" >&2; fail=1; }
[ "$hash_before" != "$fnv_basis" ] || { echo "replay_test: FAIL the state hash is the empty basis" >&2; fail=1; }
[ "$echoes" -eq 64 ] || { echo "replay_test: FAIL $echoes echoes in console.log, 64 expected" >&2; fail=1; }
[ -n "$drained" ] && [ "$drained" -eq "$last_tick" ] || { echo "replay_test: FAIL drained blocks $drained differ from last complete tick $last_tick" >&2; fail=1; }
if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
    echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
    fail=1
fi
[ "$fail" -eq 0 ] || exit 1
echo "replay_test: PASS state_hash $hash_before"
exit 0
