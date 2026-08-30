#!/usr/bin/env bash
# replay_model.sh: define the run-time model arm of the replay check.
# Inputs: replay_test.sh functions and variables. Outputs: scenario_model. Not run alone.

model_journal="${journal}-model"

feed_model() {
    printf 'model load language language-q4\n'
    local i
    for i in $(seq 1 2400); do
        if grep -q 'placement is complete' "$model_journal"/*/console.log 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    sleep 1
    printf 'models\n'
    printf 'say name one colour and nothing else\n'
    wait_turns "$model_journal" 1 || return 0
    wait_killed "$model_journal"
}

feed_model_restore() {
    local i
    for i in $(seq 1 2400); do
        if grep -q '^restore:' "$model_journal/run-2.log" 2>/dev/null; then
            printf 'models\n'
            sleep 2
            return 0
        fi
        sleep 0.1
    done
}

scenario_model() {
    local before after hash_before hash_after first second placed turns bad=0
    rm -rf "$model_journal"
    mkdir -p "$model_journal"

    feed_model | "$build/aotx_boot" --journal "$model_journal" --models "$models" \
        >"$model_journal/run-1.log" 2>&1 &
    local boot=$!
    wait_turns "$model_journal" 1 \
        || echo "replay_test: model made no turn after the placement in 360 seconds"
    sleep 1
    kill -9 "$boot"
    : >"$model_journal/killed"
    wait "$boot" 2>/dev/null
    sleep 1
    first=$(grep -h 'Qwen3-4B-Q4_0.gguf' "$model_journal"/*/console.log \
            2>/dev/null | wc -l)

    before=$("$build/aotx_restore" --journal "$model_journal" --summary) || {
        echo "replay_test: no restorable model journal after the kill" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    echo "model before: $before"

    feed_model_restore | "$build/aotx_boot" --journal "$model_journal" --restore \
        --ticks 300 --models "$models" >"$model_journal/run-2.log" 2>&1 || {
        echo "replay_test: the model restore failed; see $model_journal/run-2.log" >&2
        return 1
    }
    after=$("$build/aotx_restore" --journal "$model_journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    echo "model after:  $after"

    second=$(grep -h 'Qwen3-4B-Q4_0.gguf' "$model_journal"/*/console.log \
             2>/dev/null | wc -l)
    placed=$(grep -h -c 'placement is complete' "$model_journal"/*/console.log \
             2>/dev/null | awk '{ total += $1 } END { print total + 0 }')
    turns=$(turn_key "$model_journal"/manifest/*.jsonl | wc -l)
    echo "model cases: 1 kill, 1 restore, $turns turns after the placement," \
         "$first first-run and $second restored lines name the Q4_0 file"
    [ "$turns" -ge 1 ] \
        || { echo "replay_test: FAIL no say ran after the model placement" >&2; bad=1; }
    [ "$placed" -ge 2 ] \
        || { echo "replay_test: FAIL the file was not placed in both runs" >&2; bad=1; }
    [ "$first" -ge 1 ] \
        || { echo "replay_test: FAIL the first run did not list the Q4_0 file" >&2; bad=1; }
    [ "$second" -gt "$first" ] \
        || { echo "replay_test: FAIL the restored run did not list the Q4_0 file" >&2; bad=1; }
    if [ -z "$hash_before" ] || [ "$hash_before" != "$hash_after" ]; then
        echo "replay_test: FAIL model state_hash before=$hash_before restore_hash after=$hash_after" >&2
        bad=1
    fi
    [ "$bad" -eq 0 ] \
        && echo "replay_test: PASS model, state_hash $hash_before, Q4_0 restored"
    return "$bad"
}
