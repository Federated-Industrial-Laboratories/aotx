#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# replay_affect.sh: define the affect state arm of the replay check.
# Inputs: replay_test.sh functions and variables. Outputs: scenario_affect. Not run alone.

# The arm runs two forms: one agent, and every slot of the build. In each form the conductor
# reads a file that is not there with the fs_read tool, so a tool error event reaches the
# state. The workers of the slot form take a short task with no tool. The page pool holds
# about 24 tool prompts at once, and every slot must take a turn. The kill lands inside a
# later turn with a non-zero state. The restore must give the same state hash and the same
# state records field by field, and a next turn along the update law.

affect_journal="${journal}-affect"
affect_tools="${journal}-affect-tools"
affect_root="${journal}-affect-root"

# The slot count of the build, from its version line.
affect_slots() {
    "$build/aotx_boot" --version 2>/dev/null | sed -n 's/.* slots \([0-9]*\).*/\1/p' | head -1
}

# Waits until the affect stream holds a state line of the given count of agents.
wait_states() {
    local want="$1" i got
    for i in $(seq 1 3600); do
        got=$(cat "$affect_journal"/*/affect.jsonl 2>/dev/null | grep '"kind":"state"' \
              | sed -n 's/.*"agent":\([0-9]*\),.*/\1/p' | sort -u | wc -l)
        if [ "${got:-0}" -ge "$want" ]; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# Waits until agent 0 holds the given count of state lines.
wait_agent_states() {
    local want="$1" i got
    for i in $(seq 1 3600); do
        got=$(cat "$affect_journal"/*/affect.jsonl 2>/dev/null \
              | grep -c '"agent":0,"turn":[0-9]*,"kind":"state"' || true)
        if [ "${got:-0}" -ge "$want" ]; then
            return 0
        fi
        sleep 0.1
    done
    return 1
}

# The sampled tokens of slot 0 in the journal.
affect_sampled() {
    "$build/aotx_journal" tokens "$affect_journal" 2>/dev/null \
        | grep -c '^slot=0 .* sampled=1 ' || true
}

# Waits until slot 0 sampled tokens past the given count, so a turn is in flight.
wait_in_flight() {
    local base="$1" i got
    for i in $(seq 1 720); do
        got=$(affect_sampled)
        if [ "${got:-0}" -ge $((base + 4)) ]; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# The lines the feeder reads. The conductor reads the file with the tool, and the tool
# result is an error. Every worker takes one short task. The conductor then takes a long
# third turn, which the kill cuts short.
feed_affect() {
    local agents="$1" left batch i base
    printf 'say read the file one.txt with the fs_read tool and repeat its first line\n'
    if [ "$agents" -gt 1 ]; then
        left=$((agents - 1))
        while [ "$left" -gt 0 ]; do
            batch=$(( left > 8 ? 8 : left ))
            printf 'spawn worker %u\n' "$batch"
            left=$((left - batch))
        done
        for i in $(seq 1 $((agents - 1))); do
            printf 'task worker reply with the word OK and nothing else\n'
        done
    fi
    wait_states "$agents" || return 0
    wait_agent_states 2 || return 0
    base=$(affect_sampled)
    printf 'say count from one to one hundred, one number for each line\n'
    wait_in_flight "${base:-0}" || return 0
    : >"$affect_journal/in-flight"
    wait_killed "$affect_journal"
}

# Prints one line for each state record of a boot: the tick, the agent, the replayed mark
# and the body bytes. The agent comes from the body, because a restore stamps its own
# writer on a record it applies again. A record past the tick limit is left out.
affect_records() {
    local boot="$1" limit="$2"
    "$build/aotx_journal" records "$affect_journal" --boot "$boot" 2>/dev/null \
        | awk -v limit="$limit" '
            / type=31 / {
                tick = ""; writer = ""; flags = ""; body = "";
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^tick=/) { tick = substr($i, 6); }
                    if ($i ~ /^writer=/) { writer = substr($i, 8); }
                    if ($i ~ /^flags=/) { flags = substr($i, 7); }
                    if ($i ~ /^body=/) { body = substr($i, 6); }
                }
                agent = 0;
                for (b = 4; b >= 1; b--) {
                    agent = agent * 256 + index("0123456789abcdef", substr(body, 2 * b - 1, 1)) * 16 \
                          + index("0123456789abcdef", substr(body, 2 * b, 1)) - 17;
                }
                if (tick + 0 <= limit + 0) {
                    print tick, agent, (flags % 2), body;
                }
            }'
}

# Returns one when slot 0 has one more open record than it has manifests. The unmatched
# last open is the turn in flight at the kill. The marker controls the pipe only.
affect_open_at_kill() {
    local boot="$1" limit="$2" records opens turns
    records=$("$build/aotx_journal" records "$affect_journal" --boot "$boot" 2>/dev/null)
    opens=$(awk -v limit="$limit" '
        / type=15 / {
            tick = 0; body = "";
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^tick=/) tick = substr($i, 6);
                if ($i ~ /^body=/) body = substr($i, 6);
            }
            slot = substr(body, 7, 2) substr(body, 5, 2) substr(body, 3, 2) substr(body, 1, 2);
            event = substr(body, 15, 2) substr(body, 13, 2) substr(body, 11, 2) substr(body, 9, 2);
            if (tick + 0 <= limit + 0 && slot == "00000000" && event == "00000001") n++;
        }
        END { print n + 0 }' <<<"$records")
    turns=$(grep -c '"agent":0,' "$affect_journal/manifest/$boot.jsonl" 2>/dev/null || true)
    [ "$opens" -eq $((turns + 1)) ] && echo 1 || echo 0
}

# Prints the fields of one state record body: agent, turn, the two fast parts, the two
# slow parts and the event mask, as decimal numbers.
affect_fields() {
    local hex="$1"
    u32() { printf '%d' "0x${hex:$((($1 + 3) * 2)):2}${hex:$((($1 + 2) * 2)):2}${hex:$((($1 + 1) * 2)):2}${hex:$(($1 * 2)):2}"; }
    s16() {
        local v
        v=$(printf '%d' "0x${hex:$((($1 + 1) * 2)):2}${hex:$(($1 * 2)):2}")
        if [ "$v" -ge 32768 ]; then v=$((v - 65536)); fi
        printf '%d' "$v"
    }
    echo "$(u32 0) $(u32 4) $(s16 8) $(s16 10) $(s16 16) $(s16 18) $(u32 28)"
}

# The update law in awk at the default settings, from the event weights of the header. The
# arguments are the four parts before the turn, the event mask of the turn and the four
# parts after it. Two Q1.15 steps of slack cover the float against double difference.
affect_law_check() {
    awk -v f0="$1" -v f1="$2" -v s0="$3" -v s1="$4" -v mask="$5" \
        -v g0="$6" -v g1="$7" -v h0="$8" -v h1="$9" '
        function tanh(x) { return (exp(2 * x) - 1) / (exp(2 * x) + 1); }
        function absv(x) { return (x < 0) ? -x : x; }
        function q15(v) {
            v = v * 32768;
            v = (v >= 0) ? int(v + 0.5) : -int(-v + 0.5);
            if (v > 32767) { v = 32767; }
            if (v < -32768) { v = -32768; }
            return v;
        }
        BEGIN {
            split("0.10 -0.25 -0.50 -0.50 0.50 -0.50 -0.30 -0.50 0.50 -0.50 -0.25 -0.10 0.00 -0.10 -0.25", wv, " ");
            split("0.00 0.25 0.50 0.25 0.00 0.25 0.00 0.50 0.00 0.25 0.25 0.00 0.25 0.25 0.00", wa, " ");
            e0 = 0; e1 = 0;
            for (b = 0; b < 15; b++) {
                if (int(mask / (2 ^ b)) % 2 == 1) { e0 += wv[b + 1]; e1 += wa[b + 1]; }
            }
            wf0 = q15(tanh(0.5 * f0 / 32768 + 0.5 * e0));
            wf1 = q15(tanh(0.5 * f1 / 32768 + 0.5 * e1));
            ws0 = q15(tanh(0.9 * s0 / 32768 + 0.1 * e0));
            ws1 = q15(tanh(0.9 * s1 / 32768 + 0.1 * e1));
            bad = (absv(wf0 - g0) > 2) + (absv(wf1 - g1) > 2) + (absv(ws0 - h0) > 2) + (absv(ws1 - h1) > 2);
            printf "mask %d: want fast %d %d slow %d %d, got fast %d %d slow %d %d\n",
                   mask, wf0, wf1, ws0, ws1, g0, g1, h0, h1;
            exit (bad ? 1 : 0);
        }'
}

# Checks every agent of the restored boot that wrote a state record after its replayed
# ones. The turn must continue, and the state must follow the law from the restored state.
# Prints the count of the agents checked and returns 1 on any that does not hold.
affect_continued() {
    local file="$1" bad=0 checked=0 line hex fields agent last_hex
    declare -A last
    while read -r line; do
        set -- $line
        agent="$2"
        hex="$4"
        if [ "$3" = "1" ]; then
            last["$agent"]="$hex"
            continue
        fi
        last_hex="${last[$agent]:-}"
        if [ -z "$last_hex" ]; then
            continue
        fi
        fields=$(affect_fields "$last_hex")
        set -- $fields
        local turn0="$2" f0="$3" f1="$4" s0="$5" s1="$6"
        fields=$(affect_fields "$hex")
        set -- $fields
        local turn1="$2" g0="$3" g1="$4" h0="$5" h1="$6" mask="$7"
        checked=$((checked + 1))
        if [ "$turn1" -ne $((turn0 + 1)) ]; then
            echo "replay_test: FAIL agent $agent continued at turn $turn1 after turn $turn0" >&2
            bad=1
        fi
        if ! affect_law_check "$f0" "$f1" "$s0" "$s1" "$mask" "$g0" "$g1" "$h0" "$h1" >"$affect_journal/law.txt"; then
            echo "replay_test: FAIL agent $agent did not continue along the law: $(cat "$affect_journal/law.txt")" >&2
            bad=1
        fi
        unset "last[$agent]"
    done <"$file"
    echo "$checked"
    return "$bad"
}

# One form of the arm at a count of agents.
affect_form() {
    local agents="$1" name="affect $1"
    local before after hash_before hash_after boot_1 boot_2 tick_1 refused bad=0
    local records replayed nonzero lines checked flight
    rm -rf "$affect_journal" "$affect_tools" "$affect_root"
    mkdir -p "$affect_journal" "$affect_tools" "$affect_root"
    cp -r "$(dirname "$0")/../modules/roles/." "$affect_tools"
    chmod -R u+w "$affect_tools"
    # The role copy needs no operator for the file tool, so the error lands on its own. The
    # worker copy holds no tool and a short body, so its prompt takes few pages.
    sed -i 's/^authorise: .*/authorise:/' "$affect_tools/conductor/module.manifest"
    sed -i 's/^tools: .*/tools:/; s/^authorise: .*/authorise:/' \
        "$affect_tools/worker/module.manifest"
    printf 'You are a worker. Give the result of the task in a short answer.\n' \
        >"$affect_tools/worker/overlay.txt"
    # A sequence takes pages for its prompt and its reply limit at the open. The limit
    # keeps every slot of the build inside the page pool.
    printf 'affect.on = 1\nsample.temperature = 0\nsample.seed = 7\ndecode.reply_limit = 64\n' \
        >"$affect_journal/aotx.settings"

    feed_affect "$agents" | "$build/aotx_boot" --settings "$affect_journal/aotx.settings" \
        --journal "$affect_journal" --models "$models" --modules "$affect_tools" \
        --root "$affect_root" >"$affect_journal/run-1.log" 2>&1 &
    local boot=$!
    wait_states "$agents" || echo "replay_test: $name gave no state line of every agent in 360 seconds"
    wait_agent_states 2 || echo "replay_test: $name gave no second turn of agent 0 in 360 seconds"
    local i
    for i in $(seq 1 720); do
        if [ -f "$affect_journal/in-flight" ]; then break; fi
        sleep 0.5
    done
    flight=$([ -f "$affect_journal/in-flight" ] && echo 1 || echo 0)
    sleep 1
    kill -9 "$boot"
    : >"$affect_journal/killed"
    wait "$boot" 2>/dev/null
    sleep 1

    before=$("$build/aotx_restore" --journal "$affect_journal" --summary) || {
        echo "replay_test: no restorable journal after the kill; see $affect_journal/run-1.log" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    boot_1=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$before")
    tick_1=$(sed -n 's/.*last_tick=\([0-9]*\).*/\1/p' <<<"$before")
    flight=$(affect_open_at_kill "$boot_1" "$tick_1")
    echo "$name before: $before"
    affect_records "$boot_1" "$tick_1" >"$affect_journal/states-1.txt"
    awk '{ print $1, $2, $4 }' "$affect_journal/states-1.txt" >"$affect_journal/key-1.txt"
    records=$(wc -l <"$affect_journal/key-1.txt")
    # The last record of agent 0 before the kill holds a non-zero state.
    nonzero=$(awk '$2 == 0 { last = $4 } END {
        state = substr(last, 17, 32); print (state ~ /[1-9a-f]/) ? 1 : 0 }' \
        "$affect_journal/states-1.txt")

    "$build/aotx_boot" --settings "$affect_journal/aotx.settings" --journal "$affect_journal" \
        --restore --ticks 900 --models "$models" --modules "$affect_tools" --root "$affect_root" \
        </dev/null >"$affect_journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $affect_journal/run-2.log" >&2
        return 1
    }
    after=$("$build/aotx_restore" --journal "$affect_journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    boot_2=$(sed -n 's/^restore boot=\([0-9a-f]*\).*/\1/p' <<<"$after")
    echo "$name after:  $after"
    affect_records "$boot_2" 999999999 >"$affect_journal/states-2.txt"
    awk '$3 == 1 { print $1, $2, $4 }' "$affect_journal/states-2.txt" >"$affect_journal/key-2.txt"
    replayed=$(wc -l <"$affect_journal/key-2.txt")
    lines=$(cat "$affect_journal/$boot_2/affect.jsonl" 2>/dev/null | grep -c '"kind":"state".*"replayed":1' || true)
    refused=$(sed -n 's/^restore: applied [0-9]* hash [0-9a-f]* decode_refused \([0-9]*\).*/\1/p' \
        "$affect_journal/run-2.log" | head -1)
    local rejected
    rejected=$(sed -n 's/^restore: .* rejected \([0-9]*\)$/\1/p' \
        "${affect_journal}/run-2.log" | head -1)
    [ "${rejected:-1}" -eq 0 ] || { echo "replay_test: FAIL rejected records ${rejected:-not stated}" >&2; bad=1; }
    "$build/aotx_journal" tokens "$affect_journal" --boot "$boot_1" >"$affect_journal/tokens-1.txt" 2>/dev/null
    "$build/aotx_journal" tokens "$affect_journal" --boot "$boot_2" >"$affect_journal/tokens-2.txt" 2>/dev/null
    checked=$(affect_continued "$affect_journal/states-2.txt") || bad=1

    echo "$name cases: 1 kill, 1 restore, $agents agents, $records state records before the kill," \
         "$replayed applied again, $lines replayed state lines, $checked agents continued," \
         "in flight $flight, refused ${refused:-not stated}"
    [ "$flight" -eq 1 ] || { echo "replay_test: FAIL the journal has no last open turn without a manifest" >&2; bad=1; }
    [ "$records" -ge "$agents" ] || { echo "replay_test: FAIL $records state records before the kill, $agents agents" >&2; bad=1; }
    [ "$nonzero" = "1" ] || { echo "replay_test: FAIL the state of agent 0 was zero at the kill" >&2; bad=1; }
    if ! diff -u "$affect_journal/key-1.txt" "$affect_journal/key-2.txt" >"$affect_journal/key.diff"; then
        echo "replay_test: FAIL the state records differ; see $affect_journal/key.diff" >&2
        head -20 "$affect_journal/key.diff" >&2
        bad=1
    fi
    [ "$lines" -eq "$replayed" ] || { echo "replay_test: FAIL $lines replayed state lines, $replayed records applied again" >&2; bad=1; }
    [ "${checked:-0}" -ge 1 ] || { echo "replay_test: FAIL no agent wrote a state record after the restore" >&2; bad=1; }
    [ "${refused:-1}" -eq 0 ] || { echo "replay_test: FAIL the restored run refused ${refused:-?} sequence opens or token records" >&2; bad=1; }
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    compare_offsets "$affect_journal" "$affect_journal/tokens-1.txt" "$affect_journal/tokens-2.txt" \
        "$tick_1" "$name" || { echo "replay_test: FAIL the pace of the $name replay merged ticks of the journal" >&2; bad=1; }
    compare_turns "$affect_journal" "$boot_1" "$boot_2" "$name" || bad=1
    [ "$bad" -eq 0 ] && echo "replay_test: PASS $name, $records state records, state_hash $hash_before"
    return "$bad"
}

scenario_affect() {
    local slots bad=0
    local setting=()
    mapfile -t setting < <(sed -n 's/^AOTX_AFFECT:BOOL=//p' "$build/CMakeCache.txt" 2>/dev/null)
    if [ "${#setting[@]}" -ne 1 ]; then
        echo "replay_test: the build does not state affect support" >&2
        return 1
    fi
    case "${setting[0]^^}" in
        ''|0|OFF|NO|FALSE|N|IGNORE|NOTFOUND|*-NOTFOUND) return 2 ;;
    esac
    slots=$(affect_slots)
    if [ -z "$slots" ]; then
        echo "replay_test: FAIL the build states no slot count" >&2
        return 1
    fi
    affect_form 1 || bad=1
    affect_form "$slots" || bad=1
    return "$bad"
}
