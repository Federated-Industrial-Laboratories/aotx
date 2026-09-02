#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the fixture conversation set twice, with the affect substrate off and then on. Build
# the pairs file of the quality score tool from the transcript stream of each run. Each user
# line goes to the console agent as one say line. One outcome line goes before it when the
# turn scripts a tool result. The next line waits until the reply is complete. A side runs
# in boots of a few conversations each, so the transcript of the agent stays short.
#   Inputs: a build directory, a model store, the conversation file and an output directory.
#   Outputs: <out>/off and <out>/on with one directory for each boot, which holds the
#   journal, the streams and the run log; <out>/pairs.jsonl; <out>/tier1.txt with the tier
#   1 means; <out>/events.txt with the scripted and the observed tool events of the run
#   with the substrate on.
#   Exit codes: 0 pass, 1 a run or a check failed, 2 usage or environment error.
set -u
set -o pipefail

# Conversations one boot takes at the most.
chunk_size=6

if [ "$#" -ne 4 ]; then
    echo "usage: quality_pair.sh <build> <models> <conversations> <out>" >&2
    exit 2
fi
build=$(realpath "$1")
models=$(realpath "$2")
conversations=$(realpath "$3")
out="$4"
root=$(cd "$(dirname "$0")/.." && pwd)
if [ ! -x "$build/aotx_boot" ]; then
    echo "quality_pair: the build program is not executable: $build/aotx_boot" >&2
    exit 2
fi
if [ ! -r "$models/manifest.jsonl" ] || [ ! -r "$conversations" ]; then
    echo "quality_pair: the store or the conversation file does not read" >&2
    exit 2
fi
mkdir -p "$out" || exit 2
out=$(realpath "$out")

# The language role of the store is the role its calibration line names. A store with no
# calibration line runs the eight bit role.
role=$(sed -n 's/.*"role":"\([a-z0-9-]*\)".*/\1/p' "$models/affect/calibration.jsonl" 2>/dev/null | tail -1)
role=${role:-language}

# The turns of the set, one line each: the conversation, the tool outcome or a dash, and
# the user text.
python3 - "$conversations" >"$out/turns.tsv" <<'PY' || exit 2
import json, sys
for number, line in enumerate(open(sys.argv[1], encoding="utf-8")):
    if not line.strip():
        continue
    item = json.loads(line)
    for turn in item["turns"]:
        text = turn["user"]
        if "\t" in text or "\n" in text:
            sys.exit("quality_pair: a user line holds a tab or a line feed: %s" % item["name"])
        print("%d\t%s\t%s" % (number, turn.get("tool", "-"), text))
PY
turns=$(wc -l <"$out/turns.tsv")
count=$(cut -f1 "$out/turns.tsv" | sort -u | wc -l)
echo "quality_pair: role $role, $count conversations, $turns user lines, boots of $chunk_size, out $out"

boot=
fifo=
finish()
{
    local status=$?
    trap - EXIT HUP INT TERM
    if [ -n "$boot" ] && kill -0 "$boot" 2>/dev/null; then
        kill -TERM "$boot"
        wait "$boot"
    fi
    [ -n "$fifo" ] && rm -f "$fifo"
    exit "$status"
}
trap finish EXIT HUP INT TERM

# The manifests of the console agent whose finish is not a tool call. Each user line ends
# with one of them, so their count is the count of the complete replies.
final_replies()
{
    cat "$1"/manifest/*.jsonl 2>/dev/null | grep '"agent":0,' | grep -vc '"finish":"tool"'
}

# One boot of one side over the conversations first to last. The user lines of the boot
# go in order; the streams of the boot go beside its journal.
run_boot()
{
    local side="$1" dir="$2" first="$3" last="$4" journal target ready step total quality k lag=0
    mkdir -p "$dir" || return 1
    journal="$dir/journal"
    mkdir "$journal" || return 1
    fifo="$dir/feed"
    mkfifo "$fifo" || return 1
    "$build/aotx_boot" --settings "$out/$side.settings" --journal "$journal" \
        --models "$models" --modules "$root/modules/roles" --ticks 100000000 \
        <"$fifo" >"$dir/run.log" 2>&1 &
    boot=$!
    exec 9>"$fifo"
    for ((step = 0; step < 600; step++)); do
        if grep -q 'holds slot 0' "$dir/run.log" 2>/dev/null; then break; fi
        if ! kill -0 "$boot" 2>/dev/null; then break; fi
        sleep 0.1
    done
    k=0
    while IFS=$'\t' read -r conversation tool text; do
        if [ "$conversation" -lt "$first" ] || [ "$conversation" -gt "$last" ]; then continue; fi
        k=$((k + 1))
        if [ "$tool" != "-" ]; then
            printf 'outcome %s\n' "$tool" >&9
        fi
        printf 'say %s\n' "$text" >&9
        # The reply is complete at its final manifest. The quality line of the turn follows
        # it. A line that does not come inside the patience is named. The run then goes on
        # with the lag it saw. The next turn end writes the line late, and every later line
        # comes one turn late after that.
        ready=0
        patience=300
        for ((step = 0; step < 9000; step++)); do
            target=$(final_replies "$journal")
            total=$(cat "$journal"/manifest/*.jsonl 2>/dev/null | grep -c '"agent":0,')
            quality=$(cat "$journal"/*/quality.jsonl 2>/dev/null | grep -c '"agent":0,')
            if [ "${target:-0}" -ge "$k" ]; then
                if [ $((${quality:-0} + lag)) -ge "${total:-0}" ]; then
                    ready=1
                    break
                fi
                patience=$((patience - 1))
                if [ "$patience" -le 0 ]; then
                    lag=$((${total:-0} - ${quality:-0}))
                    echo "$side: conversation $conversation: the quality line of turn ${total:-0} did not come in 30 seconds; the boot goes on with a lag of $lag"
                    ready=1
                    break
                fi
            fi
            if ! kill -0 "$boot" 2>/dev/null; then break; fi
            sleep 0.1
        done
        if [ "$ready" -eq 0 ]; then
            echo "quality_pair: $side: the reply of a line of conversation $conversation did not complete" >&2
            return 1
        fi
        echo "$side: conversation $conversation line $k complete ($(date +%H:%M:%S)): $tool: ${text:0:60}"
    done <"$out/turns.tsv"
    printf '%s\n' quit >&9
    exec 9>&-
    wait "$boot"
    step=$?
    boot=
    rm -f "$fifo"
    fifo=
    if [ "$step" -ne 0 ]; then
        echo "quality_pair: $side: the boot ended with status $step" >&2
        return 1
    fi
    local boot_dir
    boot_dir=$(find "$journal" -mindepth 1 -maxdepth 1 -type d -name '[0-9a-f][0-9a-f]*' | sort | tail -1)
    if [ -z "$boot_dir" ] || [ ! -f "$boot_dir/transcript/0.jsonl" ] \
        || [ ! -f "$boot_dir/quality.jsonl" ]; then
        echo "quality_pair: $side: the transcript or the quality stream is absent" >&2
        return 1
    fi
    cp "$boot_dir/transcript/0.jsonl" "$dir/transcript.jsonl"
    cp "$boot_dir/quality.jsonl" "$dir/quality.jsonl"
    cp "$boot_dir/affect.jsonl" "$dir/affect.jsonl" 2>/dev/null
    cat "$journal"/manifest/*.jsonl >"$dir/manifest.jsonl"
    return 0
}

# One side: the settings, then one boot for each run of conversations.
run_side()
{
    local side="$1" on="$2" first=0 number=0
    rm -rf "$out/$side"
    mkdir -p "$out/$side" || return 1
    printf '%s\n' "affect.on = $on" 'quality.on = 1' 'affect.steer_gain = 0.25' \
        "models.roles = $role,embedding" 'sample.temperature = 0.6' 'sample.seed = 7' \
        'decode.reply_limit = 256' 'derive.list = tokens,pages,affect,quality,transcript' \
        >"$out/$side.settings"
    while [ "$first" -lt "$count" ]; do
        run_boot "$side" "$out/$side/boot-$number" "$first" $((first + chunk_size - 1)) || return 1
        first=$((first + chunk_size))
        number=$((number + 1))
    done
    return 0
}

run_side off 0 || exit 1
run_side on 1 || exit 1

# The pairs file: side a is the run with the substrate off, side b the run with it on. The
# reply of a user line is the last reply of the console agent before the next user line.
# The same read of the streams gives the tier 1 means of both sides. It also gives the
# tool events of the run with the substrate on against the script, per armed user line.
python3 - "$conversations" "$out" <<'PY'
import glob, json, sys
conversations = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
out = sys.argv[2]

def boots(side):
    return sorted(glob.glob("%s/%s/boot-*" % (out, side)), key=lambda path: int(path.rsplit("-", 1)[1]))

def rows(path, kinds=None):
    for line in open(path, encoding="utf-8"):
        item = json.loads(line)
        if kinds is None or item.get("kind") in kinds:
            yield item

def replies(side):
    turns = []
    for boot in boots(side):
        for item in rows(boot + "/transcript.jsonl", ("line", "reply")):
            if item["kind"] == "line":
                turns.append({"user": item["text"], "reply": ""})
            elif turns:
                turns[-1]["reply"] = item["text"]
    return turns

sides = {side: replies(side) for side in ("off", "on")}
expected = [turn["user"] for item in conversations for turn in item["turns"]]
for side, turns in sides.items():
    got = [turn["user"] for turn in turns]
    if got != expected:
        sys.exit("quality_pair: the %s transcripts hold %d user lines, the set %d, or their texts differ"
                 % (side, len(got), len(expected)))
    if any(turn["reply"] == "" for turn in turns):
        sys.exit("quality_pair: the %s transcripts have a user line with no reply" % side)

with open("%s/pairs.jsonl" % out, "w", encoding="utf-8") as pairs:
    at = 0
    for item in conversations:
        turns = []
        for turn in item["turns"]:
            turns.append({"user": turn["user"], "a": sides["off"][at]["reply"], "b": sides["on"][at]["reply"]})
            at += 1
        pairs.write(json.dumps({"name": item["name"], "turns": turns}, ensure_ascii=False) + "\n")

with open("%s/tier1.txt" % out, "w", encoding="utf-8") as tier:
    for side in ("off", "on"):
        lines = [row for boot in boots(side) for row in rows(boot + "/quality.jsonl") if row["agent"] == 0]
        coherence = [row["coherence_prompt"] for row in lines if row["coherence_prompt"] is not None]
        repetition = [row["repetition"] for row in lines]
        line = "tier1 %s: %d boots, %d turns, coherence_prompt mean %.6f over %d, repetition mean %.6f over %d" % (
            side, len(boots(side)), len(lines), sum(coherence) / len(coherence) if coherence else float("nan"),
            len(coherence), sum(repetition) / len(repetition) if repetition else float("nan"), len(repetition))
        print(line)
        tier.write(line + "\n")

# The turn range of each user line, boot by boot: from the turn after the previous final
# manifest to the final manifest of the line. A manifest whose finish is a tool call
# belongs to the line. The traces of the boot give the tool events of those turns.
ranges, traces = [], {}
for number, boot in enumerate(boots("on")):
    low = 0
    for row in rows(boot + "/manifest.jsonl"):
        if row["agent"] == 0 and row["finish"] != "tool":
            ranges.append((number, low + 1, row["turn"]))
            low = row["turn"]
    for row in rows(boot + "/affect.jsonl"):
        if row["agent"] == 0 and row["kind"] == "trace":
            traces[(number, row["turn"])] = [word for word in row["reason"] if word in ("tool_ok", "tool_error", "tool_refused")]
bad, at = 0, 0
with open("%s/events.txt" % out, "w", encoding="utf-8") as events:
    for item in conversations:
        for turn in item["turns"]:
            number, first, last = ranges[at] if at < len(ranges) else (-1, 0, -1)
            seen = [word for t in range(first, last + 1) for word in traces.get((number, t), [])]
            scripted = turn.get("tool")
            want = ["tool_" + scripted] if scripted else []
            state = "match" if (seen == want if scripted else True) else "MISMATCH"
            if scripted and seen != want:
                bad += 1
            events.write("%s line %d boot %d turns %d-%d: scripted %s, seen %s: %s\n"
                         % (item["name"], at + 1, number, first, last, scripted or "-", ",".join(seen) or "-", state))
            at += 1
    events.write("armed lines that do not match: %d\n" % bad)
print("events: %d armed lines checked against the affect stream, %d do not match"
      % (sum(1 for item in conversations for turn in item["turns"] if turn.get("tool")), bad))
sys.exit(1 if bad else 0)
PY
status=$?
if [ "$status" -ne 0 ]; then
    echo "quality_pair: FAIL"
    exit 1
fi
echo "quality_pair: PASS, pairs in $out/pairs.jsonl"
exit 0
