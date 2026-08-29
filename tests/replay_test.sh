#!/usr/bin/env bash
# replay_test.sh: the replay gate. Each scenario runs the system, kills it with SIGKILL,
# restores it from the journal, and compares the state hash before and after. The first
# scenario feeds command lines. The second asks the language model for a reply. The third
# leaves a tool request waiting for the operator over the kill. The fourth grants a request
# and lets its reply land before the kill.

# The fifth grants a request that no reply reaches, so the device writes a late verdict
# before the kill. The sixth runs sixteen workers at once. The seventh imports a device tool
# module and restores it by its digest. The eighth sends a long three-turn session and
# compacts it. Every scenario with a model
# compares the turns and replay pace over every turn the killed run completed.
#   replay_test.sh <build dir> <journal dir> [model dir]
# The journal directory, and the directories beside it that carry its name, are removed first.
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
auth_journal="${journal}-auth"
auth_root="${journal}-root"
fnv_basis="cbf29ce484222325"
fail=0
skipped=""

# Holds the write end of the pipe open until the scenario states that it killed the run.
# The pipe must stay open. A run whose input ends closes on its own, and the kill would
# then land on a run that already stopped. The scenario makes the file at the kill, so no
# time is lost after it.
wait_killed() {
    local i
    for i in $(seq 1 6000); do
        if [ -f "$1/killed" ]; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# Reads one field of the line that aotx_restore prints.
field() {
    sed -n "s/.*$1=\\([0-9a-f]*\\).*/\\1/p" <<<"$2"
}

# Prints the fields of every turn line that a restore must give again. They are the agent,
# the turn, the token count, the finish, the tool, the request and the hash of the output.
# The hash of the input is left out, because it names the prompt and not the turn.
turn_key() {
    awk '{
        agent = ""; turn = ""; tokens = ""; finish = ""; tool = ""; request = "";
        output = "";
        count = split($0, part, ",");
        for (i = 1; i <= count; i++) {
            split(part[i], pair, ":");
            key = pair[1];
            value = pair[2];
            gsub(/[{}"]/, "", key);
            gsub(/[{}"]/, "", value);
            if (key == "agent") { agent = value; }
            if (key == "turn") { turn = value; }
            if (key == "tokens") { tokens = value; }
            if (key == "finish") { finish = value; }
            if (key == "tool") { tool = value; }
            if (key == "request") { request = value; }
            if (key == "output_hash") { output = value; }
        }
        if (agent != "") {
            print "agent=" agent, "turn=" turn, "tokens=" tokens, "finish=" finish,
                  "tool=" tool, "request=" request, "output=" output;
        }
    }' "$1"
}

# Compares the turns of the killed run with the turns of the restored run, one by one. Every
# turn that the killed run completed must stand again in the restored run, at the same place
# and with the same fields. A run that completed no turn proves nothing, so the comparison
# fails on it. Returns 0 when every turn is the same.
compare_turns() {
    local dir="$1" one="$2" two="$3" name="$4" count kept
    turn_key "$dir/manifest/$one.jsonl" >"$dir/turns-1.txt"
    turn_key "$dir/manifest/$two.jsonl" >"$dir/turns-2.txt"
    count=$(wc -l <"$dir/turns-1.txt")
    if [ "$count" -eq 0 ]; then
        echo "replay_test: FAIL $name completed no turn before the kill, so the turn" \
             "comparison proves nothing" >&2
        return 1
    fi
    head -n "$count" "$dir/turns-2.txt" >"$dir/turns-2-head.txt"
    kept=$(wc -l <"$dir/turns-2-head.txt")
    if [ "$kept" -ne "$count" ]; then
        echo "replay_test: FAIL $name gave $kept turns and the killed run completed $count" >&2
        return 1
    fi
    if ! diff -u "$dir/turns-1.txt" "$dir/turns-2-head.txt" >"$dir/turns.diff"; then
        echo "replay_test: FAIL the turns of $name differ; see $dir/turns.diff" >&2
        head -20 "$dir/turns.diff" >&2
        return 1
    fi
    echo "$name turns: $count completed before the kill and every field is the same again"
    return 0
}

# Compares the pace of the replay. The replayed token records of the restored run are paired
# with the records of the killed run in order. The offset of a pair is the tick the restored
# run applied the record at, less the tick the killed run wrote it at. A tick of the journal
# that spills raises the offset, and a tick that merges with the next one lowers it. The
# offset therefore never decreases. Returns 0 when it holds, and prints the first decrease.
compare_offsets() {
    local dir="$1" one="$2" two="$3" limit="$4" name="$5" keys
    awk -v limit="$limit" '{
        tick = "";
        for (i = 1; i <= NF; i++) { if ($i ~ /^tick=/) { tick = substr($i, 6); } }
        if (tick != "" && tick + 0 <= limit + 0) { print tick; }
    }' "$one" >"$dir/ticks-1.txt"
    keys=$(wc -l <"$dir/ticks-1.txt")
    awk '{
        for (i = 1; i <= NF; i++) { if ($i ~ /^tick=/) { print substr($i, 6); } }
    }' "$two" | head -n "$keys" >"$dir/ticks-2.txt"
    paste "$dir/ticks-1.txt" "$dir/ticks-2.txt" | awk -v name="$name" '
        NF == 2 {
            off = $2 - $1;
            if (NR == 1) { low = off; high = off; }
            if (NR > 1 && off < last) {
                bad += 1;
                if (bad == 1) { first = "record " NR ", " last " to " off; }
            }
            if (off < low) { low = off; }
            if (off > high) { high = off; }
            last = off;
        }
        END {
            printf "%s pace: %d records paired, offset %d to %d ticks, decreases %d%s\n",
                   name, NR, low, high, bad, (bad ? " (first at " first ")" : "");
            exit (bad ? 1 : 0);
        }'
}

# Waits for a count of turns that ended in a journal. Returns 1 when the wait ends first.
wait_turns() {
    local dir="$1" want="$2" i got
    for i in $(seq 1 3600); do
        got=$(cat "$dir"/manifest/*.jsonl 2>/dev/null | grep -c '"output_hash"' || true)
        if [ "${got:-0}" -ge "$want" ]; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# Prints the number of the first file read request in the manifest of a journal.
first_request() {
    grep -h '"tool":"fs_read"' "$1"/manifest/*.jsonl 2>/dev/null \
        | sed -n 's/.*"request":\([0-9]*\).*/\1/p' | head -1
}

# Waits for a file read request in the manifest of a journal, and prints its number.
wait_first_request() {
    local dir="$1" i id
    for i in $(seq 1 1800); do
        id=$(first_request "$dir")
        if [ -n "$id" ]; then
            echo "$id"
            return 0
        fi
        sleep 0.1
    done
    return 1
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

# Waits for the run to make its first block. The drain makes the boot directory of a run
# at that block, which comes after the model files are read. A page cache that holds none
# of those 5.3 GB makes the read take a minute or more, so every wait here is 180 seconds.
wait_ticking() {
    local i
    for i in $(seq 1 1800); do
        if ls "$say_journal"/*/seg-000000.seg >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# Waits for the console to name the agent that takes a reply. A wait that ends without the
# name goes on anyway, and the checks that follow state what the run did.
wait_prompt() {
    local i
    for i in $(seq 1 1800); do
        if grep -q 'conductor:' "$say_journal"/*/console.log 2>/dev/null; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# Waits for the journal to hold a token that the model made. The kill must land inside the
# reply, because the checks read the tokens the killed run sampled.
wait_sampled() {
    local i
    for i in $(seq 1 360); do
        if "$build/aotx_journal" tokens "$say_journal" 2>/dev/null \
           | grep -q ' sampled=1 '; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# Waits for a turn that ended. The chain of a run takes one line for each turn that ended.
# A line there names a turn the run completed before the kill.
wait_turn() {
    local i
    for i in $(seq 1 1800); do
        if grep -qs '"output_hash"' "$say_journal"/manifest/*.jsonl; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# Waits for the reply of the second turn to make a token. The first turn ends before the
# kill and gives the comparison of the turns its lines. The kill must then land inside the
# second reply, so the checks of the tokens read a reply that the kill cut short.
wait_second() {
    local i want got
    want=$(sed -n 's/.*"tokens":\([0-9]*\).*/\1/p' "$say_journal"/manifest/*.jsonl \
        2>/dev/null | head -1)
    want=${want:-1}
    for i in $(seq 1 720); do
        got=$("$build/aotx_journal" tokens "$say_journal" 2>/dev/null \
            | grep -c ' sampled=1 ' || true)
        if [ "${got:-0}" -gt "$want" ]; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# The line that the feeder reads. The pipe stays open after it, so the run is killed while
# the reply is still forming. The question asks for a long answer, because a short one ends
# before the kill and the restore then has nothing left to sample.
feed_say() {
    wait_ticking
    # A first reply that ends well before the kill. The comparison of the turns needs a turn
    # the killed run completed, and a short question gives one.
    printf 'say name one colour and nothing else\n'
    wait_turn
    sleep 1
    printf 'say count from one to one hundred, one number for each line\n'
    wait_killed "$say_journal"
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
    local keys made again refused held first paced
    rm -rf "$say_journal"
    mkdir -p "$say_journal"

    feed_say | "$build/aotx_boot" --journal "$say_journal" --models "$models" \
        >"$say_journal/run-1.log" 2>&1 &
    local boot=$!
    wait_prompt || echo "replay_test: the console did not name the agent in 180 seconds"
    wait_sampled || echo "replay_test: the reply made no token in 180 seconds"
    wait_turn || echo "replay_test: no turn ended in 180 seconds"
    wait_second || echo "replay_test: the second reply made no token in 360 seconds"
    sleep 2
    kill -9 "$boot"
    : >"$say_journal/killed"
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
    # The journal holds one sequence for each turn, and the position of a token starts again
    # at zero with every sequence. The place the replay left slot 0 is therefore the largest
    # position it applied again there, and the first new token stands after it.
    held=$(grep '^slot=0 .* replayed=1$' "$say_journal/tokens-2.txt" \
        | sed -n 's/^slot=0 position=\([0-9]*\) .*/\1/p' | sort -n | tail -1)
    held=$(( ${held:--1} + 1 ))
    first=$(grep -m1 '^slot=0 .* sampled=1 replayed=0$' "$say_journal/tokens-2.txt" \
        | sed -n 's/^slot=0 position=\([0-9]*\) .*/\1/p')
    # The pace of the replay. The apply takes the records of one tick of the journal in one
    # tick of the restored run. A tick of the journal with no record therefore gives a tick
    # here that takes none.
    paced=$(sed -n 's/^restore: .* paced \([0-9]*\).*/\1/p' "$say_journal/run-2.log" \
        | head -1)

    echo "say cases: 1 kill, 1 restore, $keys token records in the replayed prefix," \
         "$made sampled before the kill, $again sampled after the restore point," \
         "slot 0 left at position $held, first new token at position ${first:-none}," \
         "refused ${refused:-not stated}, paced ${paced:-not stated}"
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
        echo "replay_test: FAIL the first token of slot 0 sits at position $first and the replay left the slot at position $held" >&2
        bad=1
    fi
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    compare_offsets "$say_journal" "$say_journal/tokens-1.txt" "$say_journal/tokens-2.txt" \
        "$tick_1" "say" || { echo "replay_test: FAIL the pace of the say replay merged ticks of the journal" >&2; bad=1; }
    compare_turns "$say_journal" "$boot_1" "$boot_2" "say" || bad=1
    [ "$bad" -eq 0 ] && echo "replay_test: PASS say, state_hash $hash_before, tokens $keys"
    return "$bad"
}

# ---- the third scenario: a request that waits for the operator ----

# Waits for a turn that made a file read request. The requests file holds a request only
# after the operator grants it. A feeder that took a line earlier would execute a tool that
# nobody authorized. The manifest of the turn is therefore the signal that a request waits.
# The model loads first and then writes a reply, so the wait is 180 seconds long.
wait_request() {
    local i
    for i in $(seq 1 1800); do
        if grep -qs '"tool":"fs_read"' "$auth_journal"/manifest/*.jsonl; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# The lines of the first run. The pipe stays open, so the run is killed while the request
# still waits for an answer.
feed_auth() {
    printf 'spawn worker\n'
    printf 'task worker read the file one.txt with the fs_read tool and repeat its first line\n'
    wait_killed "$auth_journal"
}

# The line of the restored run. The request is derived again early in the run and it fails
# at its deadline, which is 500 ticks and five seconds of the pace. The answer therefore
# waits for the manifest of the restored run to name the request, and goes at once.
feed_auth_answer() {
    local want="$1" i one id
    for i in $(seq 1 6000); do
        for one in "$auth_journal"/manifest/*.jsonl; do
            case "$one" in
                *"$want".jsonl) continue ;;
            esac
            id=$(sed -n 's/.*"tool":"fs_read".*"request":\([0-9]*\).*/\1/p' "$one" \
                 2>/dev/null | head -1)
            if [ -n "$id" ]; then
                printf 'authorize %s\n' "$id"
                sleep 120
                return 0
            fi
        done
        sleep 0.05
    done
    sleep 5
}

scenario_auth() {
    local id before boot_1 boot_2 held after granted turns replies bad=0
    local carried missing
    rm -rf "$auth_journal" "$auth_root"
    mkdir -p "$auth_journal" "$auth_root"
    printf 'the first line of the file\nthe second line of the file\n' >"$auth_root/one.txt"

    feed_auth | "$build/aotx_boot" --journal "$auth_journal" --models "$models" \
        --root "$auth_root" >"$auth_journal/run-1.log" 2>&1 &
    local boot=$!
    if ! wait_request; then
        kill -9 "$boot" 2>/dev/null
        : >"$auth_journal/killed"
        wait "$boot" 2>/dev/null
        echo "replay_test: auth gave no request that waits for the operator in 180 seconds;" \
             "see $auth_journal/run-1.log"
        return 2
    fi
    id=$(grep -h '"tool":"fs_read"' "$auth_journal"/manifest/*.jsonl \
        | sed -n 's/.*"request":\([0-9]*\).*/\1/p' | head -1)
    held=$(grep -hc "\"request\":$id," "$auth_journal"/manifest/*.jsonl | head -1)
    sleep 1
    kill -9 "$boot"
    : >"$auth_journal/killed"
    wait "$boot" 2>/dev/null
    sleep 1

    before=$("$build/aotx_restore" --journal "$auth_journal" --summary) || {
        echo "replay_test: no restorable journal after the kill" >&2
        return 1
    }
    boot_1=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$before")
    echo "auth before: request $id, $held turn lines, $before"

    feed_auth_answer "$boot_1" | "$build/aotx_boot" --journal "$auth_journal" --restore \
        --ticks 4000 --models "$models" --root "$auth_root" \
        >"$auth_journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $auth_journal/run-2.log" >&2
        return 1
    }
    boot_2=$("$build/aotx_restore" --journal "$auth_journal" --summary \
        | sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p')

    after=$(grep -c "\"request\":$id," "$auth_journal/manifest/$boot_2.jsonl" 2>/dev/null || true)
    granted=$(grep -c "\"request\":$id,.*\"auth\":\"granted\"" \
        "$auth_journal/requests.jsonl" 2>/dev/null || true)
    turns=$(grep -c '"agent":' "$auth_journal/manifest/$boot_2.jsonl" 2>/dev/null || true)
    replies=$(sed -n 's/^feed: requests [0-9]*, replies \([0-9]*\),.*/\1/p' \
        "$auth_journal/run-2.log" | tail -1)

    echo "auth cases: 1 kill, 1 restore, request $id waited $held turn before the kill and" \
         "$after after the restore, $granted granted lines, ${replies:-0} reply parts," \
         "$turns turns in the restored run"
    if [ "${turns:-0}" -eq 0 ]; then
        # The replay rebuilds the sequence of the agent from the token records. The
        # restored run made no turn of its own in its tick count. This form of the case
        # therefore states nothing about the request that waited. The device form of the
        # same case is the authorization arm of tests/agent_test.cu.
        echo "replay_test: auth reached the kill with request $id waiting, and the restored" \
             "run made no turn in 4000 ticks; see $auth_journal/run-2.log"
        return 2
    fi
    [ "$held" -ge 1 ] || { echo "replay_test: FAIL no request waited before the kill" >&2; bad=1; }
    [ "$after" -ge 1 ] || { echo "replay_test: FAIL the request was not presented again with the same number" >&2; bad=1; }
    [ "$granted" -ge 1 ] || { echo "replay_test: FAIL the answer of the operator is not in the requests file" >&2; bad=1; }
    [ "$turns" -ge 2 ] || { echo "replay_test: FAIL the restored run made $turns turns, so no reply reached the agent" >&2; bad=1; }
    [ "${replies:-0}" -ge 1 ] || { echo "replay_test: FAIL the feeder made ${replies:-0} reply parts" >&2; bad=1; }
    [ -n "$boot_2" ] && [ "$boot_2" != "$boot_1" ] || { echo "replay_test: FAIL the restored run has boot $boot_2" >&2; bad=1; }

    # The content of the reply, and not the count of the lines. The reply record carries
    # the bytes of the file into the journal. A device that writes an argument line the
    # feeder cannot split gives the reason of a file that is not there. This arm therefore
    # fails on that regression.
    carried=$(grep -rac 'the first line of the file' "$auth_journal" 2>/dev/null \
              | awk -F: '{ s += $2 } END { print s + 0 }')
    missing=$(grep -rac 'the file is not there' "$auth_journal" 2>/dev/null \
              | awk -F: '{ s += $2 } END { print s + 0 }')
    echo "auth reply: $carried records carry the bytes of the file, $missing say the file" \
         "is not there"
    [ "${carried:-0}" -ge 1 ] || { echo "replay_test: FAIL no reply carried the bytes of the file" >&2; bad=1; }
    [ "${missing:-0}" -eq 0 ] || { echo "replay_test: FAIL $missing replies say the file is not there" >&2; bad=1; }
    compare_turns "$auth_journal" "$boot_1" "$boot_2" "auth" || bad=1
    [ "$bad" -eq 0 ] && echo "replay_test: PASS auth, request $id, $after request lines, $turns turns, $carried replies with the bytes"
    return "$bad"
}

# ---- the fourth scenario: a request granted and answered before the kill ----

answered_journal="${journal}-answered"
answered_root="${journal}-answered-root"

# The lines of the first run. The grant goes in when the request stands in the manifest.
# The reply of the feeder lands and the worker takes its second turn. The kill comes after
# that turn. The restored run must not execute the request a second time.
feed_answered() {
    local id
    printf 'spawn worker\n'
    printf 'task worker read the file one.txt with the fs_read tool and repeat its first line\n'
    id=$(wait_first_request "$answered_journal") || return 0
    printf 'authorize %s\n' "$id"
    wait_killed "$answered_journal"
}

scenario_answered() {
    local before after hash_before hash_after boot_1 boot_2 id tick_1 turns replies granted bad=0
    rm -rf "$answered_journal" "$answered_root"
    mkdir -p "$answered_journal" "$answered_root"
    printf 'the first line of the file\nthe second line of the file\n' >"$answered_root/one.txt"

    feed_answered | "$build/aotx_boot" --journal "$answered_journal" --models "$models" \
        --root "$answered_root" >"$answered_journal/run-1.log" 2>&1 &
    local boot=$!
    wait_turns "$answered_journal" 2 || echo "replay_test: answered made no second turn in 360 seconds"
    sleep 1
    kill -9 "$boot"
    : >"$answered_journal/killed"
    wait "$boot" 2>/dev/null
    sleep 1

    before=$("$build/aotx_restore" --journal "$answered_journal" --summary) || {
        echo "replay_test: no restorable journal after the kill" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    boot_1=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$before")
    tick_1=$(sed -n 's/.*last_tick=\([0-9]*\).*/\1/p' <<<"$before")
    id=$(first_request "$answered_journal")
    echo "answered before: request ${id:-none}, $before"

    "$build/aotx_boot" --journal "$answered_journal" --restore --ticks 300 --models "$models" \
        --root "$answered_root" </dev/null >"$answered_journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $answered_journal/run-2.log" >&2
        return 1
    }
    after=$("$build/aotx_restore" --journal "$answered_journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    boot_2=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$after")
    echo "answered after:  $after"
    "$build/aotx_journal" tokens "$answered_journal" --boot "$boot_1" >"$answered_journal/tokens-1.txt" 2>/dev/null
    "$build/aotx_journal" tokens "$answered_journal" --boot "$boot_2" >"$answered_journal/tokens-2.txt" 2>/dev/null

    turns=$(turn_key "$answered_journal/manifest/$boot_1.jsonl" | wc -l)
    replies=$(sed -n 's/^feed: requests [0-9]*, replies \([0-9]*\),.*/\1/p' \
        "$answered_journal/run-2.log" | tail -1)
    echo "answered cases: 1 kill, 1 restore, request ${id:-none} granted and answered," \
         "$turns turns before the kill, the restored feeder made ${replies:-not stated} reply parts"
    [ -n "$id" ] || { echo "replay_test: FAIL no file read request was made" >&2; bad=1; }
    [ "$turns" -ge 2 ] || { echo "replay_test: FAIL the worker took $turns turns before the kill, so no reply landed" >&2; bad=1; }
    granted=$(grep -c "\"request\":${id:-0},.*\"auth\":\"granted\"" "$answered_journal/requests.jsonl" 2>/dev/null || true)
    # The killed run wrote the one line of the grant. A restored run that derived the grant
    # again would give the feeder the request a second time.
    [ "${granted:-0}" -eq 1 ] || { echo "replay_test: FAIL the requests file holds ${granted:-0} granted lines for request ${id:-0}, and the killed run wrote one" >&2; bad=1; }
    [ "${replies:-1}" -eq 0 ] || { echo "replay_test: FAIL the restored run executed the request again and made ${replies:-?} reply parts" >&2; bad=1; }
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    compare_offsets "$answered_journal" "$answered_journal/tokens-1.txt" \
        "$answered_journal/tokens-2.txt" "$tick_1" "answered" \
        || { echo "replay_test: FAIL the pace of the answered replay merged ticks of the journal" >&2; bad=1; }
    compare_turns "$answered_journal" "$boot_1" "$boot_2" "answered" || bad=1
    [ "$bad" -eq 0 ] && echo "replay_test: PASS answered, request $id, $turns turns, state_hash $hash_before"
    return "$bad"
}

# ---- the fifth scenario: a request granted that no reply reaches ----

late_journal="${journal}-late"

# The run has no root, so its feeder executes no request. The grant starts the deadline,
# the deadline passes, and the device writes the late verdict. The worker then takes its
# second turn with the reason. The kill comes after that turn.
feed_late() {
    local id
    printf 'spawn worker\n'
    printf 'task worker read the file one.txt with the fs_read tool and repeat its first line\n'
    id=$(wait_first_request "$late_journal") || return 0
    printf 'authorize %s\n' "$id"
    wait_killed "$late_journal"
}

scenario_late() {
    local before after hash_before hash_after boot_1 boot_2 id tick_1 turns replies granted bad=0
    rm -rf "$late_journal"
    mkdir -p "$late_journal"

    feed_late | "$build/aotx_boot" --journal "$late_journal" --models "$models" \
        >"$late_journal/run-1.log" 2>&1 &
    local boot=$!
    wait_turns "$late_journal" 2 || echo "replay_test: late made no second turn in 360 seconds"
    sleep 1
    kill -9 "$boot"
    : >"$late_journal/killed"
    wait "$boot" 2>/dev/null
    sleep 1

    before=$("$build/aotx_restore" --journal "$late_journal" --summary) || {
        echo "replay_test: no restorable journal after the kill" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    boot_1=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$before")
    tick_1=$(sed -n 's/.*last_tick=\([0-9]*\).*/\1/p' <<<"$before")
    id=$(first_request "$late_journal")
    echo "late before: request ${id:-none}, $before"

    "$build/aotx_boot" --journal "$late_journal" --restore --ticks 300 --models "$models" \
        </dev/null >"$late_journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $late_journal/run-2.log" >&2
        return 1
    }
    after=$("$build/aotx_restore" --journal "$late_journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    boot_2=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$after")
    echo "late after:  $after"
    "$build/aotx_journal" tokens "$late_journal" --boot "$boot_1" >"$late_journal/tokens-1.txt" 2>/dev/null
    "$build/aotx_journal" tokens "$late_journal" --boot "$boot_2" >"$late_journal/tokens-2.txt" 2>/dev/null

    turns=$(turn_key "$late_journal/manifest/$boot_1.jsonl" | wc -l)
    replies=$(sed -n 's/^feed: requests [0-9]*, replies \([0-9]*\),.*/\1/p' \
        "$late_journal/run-1.log" | tail -1)
    echo "late cases: 1 kill, 1 restore, request ${id:-none} granted with no feeder to answer," \
         "$turns turns before the kill, the feeder of the killed run made ${replies:-0} reply parts"
    [ -n "$id" ] || { echo "replay_test: FAIL no file read request was made" >&2; bad=1; }
    granted=$(grep -c "\"request\":${id:-0},.*\"auth\":\"granted\"" "$late_journal/requests.jsonl" 2>/dev/null || true)
    # The killed run wrote the one line of the grant. A restored run that derived the grant
    # again would give the feeder the request a second time.
    [ "${granted:-0}" -eq 1 ] || { echo "replay_test: FAIL the requests file holds ${granted:-0} granted lines for request ${id:-0}, and the killed run wrote one" >&2; bad=1; }
    [ "$turns" -ge 2 ] || { echo "replay_test: FAIL the worker took $turns turns before the kill, so no late verdict ended its request" >&2; bad=1; }
    [ "${replies:-0}" -eq 0 ] || { echo "replay_test: FAIL the feeder of the killed run made ${replies} reply parts, so the verdict was not late" >&2; bad=1; }
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    compare_offsets "$late_journal" "$late_journal/tokens-1.txt" "$late_journal/tokens-2.txt" \
        "$tick_1" "late" || { echo "replay_test: FAIL the pace of the late replay merged ticks of the journal" >&2; bad=1; }
    compare_turns "$late_journal" "$boot_1" "$boot_2" "late" || bad=1
    [ "$bad" -eq 0 ] && echo "replay_test: PASS late, request $id, $turns turns, state_hash $hash_before"
    return "$bad"
}

# ---- the sixth scenario: sixteen workers at once ----

wide_journal="${journal}-wide"

# Sixteen workers take sixteen tasks. Every prompt holds more records than the apply takes
# in one tick, so the pace of the replay is exercised at width. The kill comes when six
# turns ended and the rest still run.
feed_wide() {
    local i
    printf 'spawn worker 8\n'
    printf 'spawn worker 8\n'
    for i in $(seq 1 16); do
        printf 'task worker write a story of two hundred words about a clock that runs ahead of its town\n'
    done
    wait_killed "$wide_journal"
}

scenario_wide() {
    local before after hash_before hash_after boot_1 boot_2 tick_1 turns refused bad=0
    rm -rf "$wide_journal"
    mkdir -p "$wide_journal"

    feed_wide | "$build/aotx_boot" --journal "$wide_journal" --models "$models" \
        >"$wide_journal/run-1.log" 2>&1 &
    local boot=$!
    wait_turns "$wide_journal" 6 || echo "replay_test: wide made no six turns in 360 seconds"
    sleep 1
    kill -9 "$boot"
    : >"$wide_journal/killed"
    wait "$boot" 2>/dev/null
    sleep 1

    before=$("$build/aotx_restore" --journal "$wide_journal" --summary) || {
        echo "replay_test: no restorable journal after the kill" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    boot_1=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$before")
    tick_1=$(sed -n 's/.*last_tick=\([0-9]*\).*/\1/p' <<<"$before")
    echo "wide before: $before"

    "$build/aotx_boot" --journal "$wide_journal" --restore --ticks 300 --models "$models" \
        </dev/null >"$wide_journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $wide_journal/run-2.log" >&2
        return 1
    }
    after=$("$build/aotx_restore" --journal "$wide_journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    boot_2=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$after")
    echo "wide after:  $after"
    "$build/aotx_journal" tokens "$wide_journal" --boot "$boot_1" >"$wide_journal/tokens-1.txt" 2>/dev/null
    "$build/aotx_journal" tokens "$wide_journal" --boot "$boot_2" >"$wide_journal/tokens-2.txt" 2>/dev/null

    turns=$(turn_key "$wide_journal/manifest/$boot_1.jsonl" | wc -l)
    refused=$(sed -n 's/^restore: applied [0-9]* hash [0-9a-f]* refused \([0-9]*\).*/\1/p' \
        "$wide_journal/run-2.log" | head -1)
    echo "wide cases: 1 kill, 1 restore, 16 workers, $turns turns before the kill," \
         "refused ${refused:-not stated}"
    [ "$turns" -ge 6 ] || { echo "replay_test: FAIL only $turns turns ended before the kill" >&2; bad=1; }
    [ "${refused:-1}" -eq 0 ] || { echo "replay_test: FAIL the restored run refused ${refused:-?} sequence opens" >&2; bad=1; }
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    compare_offsets "$wide_journal" "$wide_journal/tokens-1.txt" "$wide_journal/tokens-2.txt" \
        "$tick_1" "wide" || { echo "replay_test: FAIL the pace of the wide replay merged ticks of the journal" >&2; bad=1; }
    compare_turns "$wide_journal" "$boot_1" "$boot_2" "wide" || bad=1
    [ "$bad" -eq 0 ] && echo "replay_test: PASS wide, $turns turns, state_hash $hash_before"
    return "$bad"
}

source "$(dirname "$0")/replay_session.sh"
# ---- the settings scenario: a set line and a settings file across a kill ----

# The settings file names one key and the console changes another. Both are class A
# records. The restored run must hold both before its first operator line and its state
# hash must equal the killed run's. The pace arm reads the wall time of 300 restored ticks.
# At 40 ms a tick they take at least 12 s, against 3 s at the default.

feed_settings() {
    echo "set tick.period_ms 40"
    local i
    for i in $(seq 1 16); do
        echo "note settings line $i"
    done
    wait_killed "$journal"
}

scenario_settings() {
    local before after hash_before hash_after records start_ns end_ns took_ms bad=0
    rm -rf "$journal"
    mkdir -p "$journal"
    printf 'decode.reply_limit = 100\nsample.top_k = 7\n' >"$journal/aotx.settings"

    feed_settings | "$build/aotx_boot" --journal "$journal" --settings "$journal/aotx.settings" \
        >"$journal/run-1.log" 2>&1 &
    local boot=$!
    sleep 4
    kill -9 "$boot"
    # The producer holds the pipe until this file exists, and the wait below holds until the
    # producer ends, so the file comes first.
    touch "$journal/killed"
    wait "$boot" 2>/dev/null
    sleep 1

    before=$("$build/aotx_restore" --journal "$journal" --summary) || {
        echo "replay_test: no restorable journal after the kill" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    echo "settings before: $before"

    start_ns=$(date +%s%N)
    "$build/aotx_boot" --journal "$journal" --restore --ticks 300 </dev/null \
        >"$journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $journal/run-2.log" >&2
        return 1
    }
    end_ns=$(date +%s%N)
    took_ms=$(( (end_ns - start_ns) / 1000000 ))

    after=$("$build/aotx_restore" --journal "$journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    echo "settings after:  $after"

    # The restored boot must hold the three records again, every one replayed. The set
    # line's record stands after the file's two, which is the order the journal holds.
    records=$("$build/aotx_journal" settings "$journal" 2>/dev/null | grep -c 'replayed=1' || true)
    echo "settings cases: 1 kill, 1 restore, $records replayed setting records," \
         "300 restored ticks in $took_ms ms"
    [ "$records" -eq 3 ] || { echo "replay_test: FAIL $records replayed setting records, 3 expected" >&2; bad=1; }
    [ "$took_ms" -ge 9000 ] || { echo "replay_test: FAIL 300 ticks took $took_ms ms; the restored run does not hold the 40 ms period" >&2; bad=1; }
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    [ "$bad" -eq 0 ] && echo "replay_test: PASS settings, state_hash $hash_before"
    return "$bad"
}

# ---- the seventh scenario: a device tool module over a kill ----

# The run imports one device tool module, the graph takes a node for it, and the kill falls
# after that. The restore replays the import from the journal and the host glue reads the
# module file again and checks its digest. The scenario then changes that file and restores
# again, and the entry is refused with the reason of the digest. The module directory comes
# from the build, which the module setup filled. The scenario reports a skip when it is not
# there.
scenario_module() {
    local before after hash_before hash_after installed captured refused bad=0
    local tools="$journal/tools"
    if [ ! -f "$build/modules/word_count/word_count.ptx" ]; then
        return 2
    fi
    rm -rf "$journal" "$tools"
    mkdir -p "$journal" "$tools"
    cp -r "$build/modules/word_count" "$tools/word_count"
    chmod -R u+w "$tools/word_count"

    sleep 20 | "$build/aotx_boot" --journal "$journal" --modules "$tools" \
        >"$journal/run-1.log" 2>&1 &
    local boot=$!
    sleep 5
    kill -9 "$boot"
    wait "$boot" 2>/dev/null
    # The disk-side programs die with their parent and finish the published blocks first.
    sleep 1

    installed=$(cat "$journal"/*/console.log 2>/dev/null \
                | grep -c 'import: word_count tool installed' || true)
    captured=$(cat "$journal"/*/console.log 2>/dev/null \
               | grep -c 'the tick graph was captured again' || true)
    before=$("$build/aotx_restore" --journal "$journal" --summary) || {
        echo "replay_test: no restorable journal after the kill" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    echo "module before: $before"

    "$build/aotx_boot" --journal "$journal" --restore --modules "$tools" --ticks 40 \
        </dev/null >"$journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $journal/run-2.log" >&2
        return 1
    }
    after=$("$build/aotx_restore" --journal "$journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    echo "module after:  $after"

    # The module file changes on disk, and a third run refuses it by its digest.
    printf '\n' >> "$tools/word_count/word_count.ptx"
    "$build/aotx_boot" --journal "$journal" --restore --modules "$tools" --ticks 40 \
        </dev/null >"$journal/run-3.log" 2>&1 || true
    refused=$(grep -c 'is refused' "$journal/run-3.log" || true)

    echo "module cases: 1 kill, 2 restores, $installed imports, $captured captures," \
         "$refused refusals of a changed module file"
    [ "$installed" -ge 1 ] || { echo "replay_test: FAIL the module did not install before the kill" >&2; bad=1; }
    [ "$captured" -ge 1 ] || { echo "replay_test: FAIL the graph was not captured again" >&2; bad=1; }
    [ "$refused" -ge 1 ] || { echo "replay_test: FAIL a changed module file was not refused" >&2; bad=1; }
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    [ "$bad" -eq 0 ] && echo "replay_test: PASS module, state_hash $hash_before"
    return "$bad"
}

# ---- the scenarios ----

scenario_lines || fail=1
scenario_settings || fail=1

applied=2
skipcount=0
if [ ! -f "$models/manifest.jsonl" ]; then
    skipped="say, auth, answered, late and wide (no $models/manifest.jsonl)"
    skipcount=5
elif ! grep -q '"name":"language"' "$models/manifest.jsonl"; then
    skipped="say, auth, answered, late and wide (the manifest holds no language role)"
    skipcount=5
else
    scenario_say || fail=1
    applied=$((applied + 1))
    scenario_auth
    case "$?" in
        0) applied=$((applied + 1)) ;;
        2) skipped="auth (the restored run made no turn)"; skipcount=1 ;;
        *) fail=1; applied=$((applied + 1)) ;;
    esac
    scenario_answered || fail=1
    applied=$((applied + 1))
    scenario_late || fail=1
    applied=$((applied + 1))
    scenario_wide || fail=1
    applied=$((applied + 1))
    scenario_session || fail=1
    applied=$((applied + 1))
fi

scenario_module
case "$?" in
    0) applied=$((applied + 1)) ;;
    2) skipped="${skipped:+$skipped, }module (the build holds no module file)"
       skipcount=$((skipcount + 1)) ;;
    *) fail=1; applied=$((applied + 1)) ;;
esac

if [ "$skipcount" -gt 0 ]; then
    echo "replay_test: scenarios applied $applied, skipped $skipcount: $skipped"
else
    echo "replay_test: scenarios applied $applied, skipped 0"
fi
[ "$fail" -eq 0 ] || exit 1
exit 0
