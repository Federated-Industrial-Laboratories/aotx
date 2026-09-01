# replay_request.sh: the late, wide and module cases of the replay check.
# The caller provides the journal paths, helper functions and program paths.

# ---- a request granted that no reply reaches ----

late_journal="${journal}-late"
late_tools="${journal}-late-tools"
late_root="${journal}-late-root"

# The host tool runs longer than its device deadline. The grant starts the tool, the
# deadline passes, and the device writes the late verdict before the feeder has a reply.
# The worker then takes its second turn with the reason. The kill comes after that turn.
late_request() {
    grep -h '"kind":"call","tool":"module-' \
        "$late_journal"/*/transcript/*.jsonl 2>/dev/null \
        | sed -n 's/.*"request":\([0-9]*\).*/\1/p' | head -1
}

wait_late_request() {
    local i id
    for i in $(seq 1 1800); do
        id=$(late_request)
        if [ -n "$id" ]; then
            echo "$id"
            return 0
        fi
        sleep 0.1
    done
    return 1
}

feed_late() {
    local id
    printf 'spawn worker\nagent 1 pages 64\n'
    printf 'task worker call hang_tool with text=one and state its result\n'
    id=$(wait_late_request) || return 0
    printf 'authorize %s\n' "$id"
    wait_killed "$late_journal"
}

scenario_late() {
    local before after hash_before hash_after boot_1 boot_2 id tick_1 turns replies granted bad=0
    if [ ! -f "$build/modules/hang_tool/hang_tool.sh" ]; then
        return 2
    fi
    rm -rf "$late_journal" "$late_tools" "$late_root"
    mkdir -p "$late_journal" "$late_tools" "$late_root"
    cp -r "$(dirname "$0")/../modules/roles/." "$late_tools"
    cp -r "$build/modules/hang_tool" "$late_tools/hang_tool"
    chmod -R u+w "$late_tools/hang_tool"
    sed -i '/^tools:/ s/$/,hang_tool/' "$late_tools/worker/module.manifest"
    sed -i 's/^authorise: .*/authorise: always/; s/^timeout: .*/timeout: 300/' \
        "$late_tools/hang_tool/module.manifest"
    printf 'deadline: 20\n' >>"$late_tools/hang_tool/module.manifest"

    feed_late | "$build/aotx_boot" --settings "$empty_settings" --journal "$late_journal" --models "$models" \
        --modules "$late_tools" --root "$late_root" >"$late_journal/run-1.log" 2>&1 &
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
    id=$(late_request)
    echo "late before: request ${id:-none}, $before"

    "$build/aotx_boot" --settings "$empty_settings" --journal "$late_journal" --restore --ticks 300 --models "$models" \
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
    echo "late cases: 1 kill, 1 restore, request ${id:-none} granted with no reply," \
         "$turns turns before the kill, the feeder of the killed run made ${replies:-0} reply parts"
    [ -n "$id" ] || { echo "replay_test: FAIL no hanging tool request was made" >&2; bad=1; }
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

# ---- sixteen workers at once ----

wide_journal="${journal}-wide"

# Sixteen workers take sixteen tasks. Every prompt holds more records than the apply takes
# in one tick, so the pace of the replay is exercised at width. The kill comes when six
# turns ended and the rest still run.
feed_wide() {
    local i
    printf 'spawn worker 8\n'
    printf 'spawn worker 8\n'
    for i in $(seq 1 16); do
        printf 'agent %u pages 64\n' "$i"
        printf 'task worker write a story of two hundred words about a clock that runs ahead of its town\n'
    done
    wait_killed "$wide_journal"
}
scenario_wide() {
    local before after hash_before hash_after boot_1 boot_2 tick_1 turns refused bad=0
    rm -rf "$wide_journal"
    mkdir -p "$wide_journal"

    feed_wide | "$build/aotx_boot" --settings "$empty_settings" --journal "$wide_journal" --models "$models" \
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

    "$build/aotx_boot" --settings "$empty_settings" --journal "$wide_journal" --restore --ticks 300 --models "$models" \
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

# ---- a device tool module over a kill ----

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

    sleep 20 | "$build/aotx_boot" --settings "$empty_settings" --journal "$journal" --modules "$tools" \
        >"$journal/run-1.log" 2>&1 &
    local boot=$!
    sleep 5
    kill -9 "$boot"
    wait "$boot" 2>/dev/null
    # The disk-side programs die with their parent and finish the published blocks first.
    sleep 1

    installed=$(cat "$journal"/*/console.log 2>/dev/null \
                | grep -c 'import: the tool word_count is installed' || true)
    captured=$(cat "$journal"/*/console.log 2>/dev/null \
               | grep -c 'the tick graph was captured again' || true)
    before=$("$build/aotx_restore" --journal "$journal" --summary) || {
        echo "replay_test: no restorable journal after the kill" >&2
        return 1
    }
    hash_before=$(field state_hash "$before")
    echo "module before: $before"

    "$build/aotx_boot" --settings "$empty_settings" --journal "$journal" --restore --modules "$tools" --ticks 40 \
        </dev/null >"$journal/run-2.log" 2>&1 || {
        echo "replay_test: the restore run failed; see $journal/run-2.log" >&2
        return 1
    }
    after=$("$build/aotx_restore" --journal "$journal" --summary) || return 1
    hash_after=$(field restore_hash "$after")
    echo "module after:  $after"

    # The module file changes on disk, and a third run refuses it by its digest.
    printf '\n' >> "$tools/word_count/word_count.ptx"
    "$build/aotx_boot" --settings "$empty_settings" --journal "$journal" --restore --modules "$tools" --ticks 40 \
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
