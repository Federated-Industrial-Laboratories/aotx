#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Check the two modes of the quality score tool on a private copy of a store. A ten-item
# task set with known answers runs at three doses. The plain score is high, the large dose
# changes a letter, and each score is computed again from the item lines. A one-item and a
# 64-item run assert the mean. A four-pair fixture runs in the pair mode with the blinded
# output. The malformed inputs of both modes are refused.
#   Inputs: a build directory and a model store.
#   Outputs: the tool lines and one line per check.
#   Exit codes: 0 pass, 1 a check failed, 2 usage or environment error.
set -u
set -o pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: quality_score.sh <build> <models>" >&2
    exit 2
fi
build="$1"
models="$2"
for item in "$build/aotx_quality_score" "$build/aotx_steer_derive"; do
    if [ ! -x "$item" ]; then
        echo "quality_score: the build program is not executable: $item" >&2
        exit 2
    fi
done
if [ ! -r "$models/manifest.jsonl" ]; then
    echo "quality_score: the model store has no manifest: $models" >&2
    exit 2
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/aotx-quality-score-XXXXXX") || exit 2
bad=0

# A pass removes the work directory. A failure keeps the store and the logs and names the
# directory, so that the difference can be examined.
finish()
{
    local status=$?
    trap - EXIT
    if [ "$status" -eq 0 ]; then
        rm -rf "$work"
    else
        echo "quality_score: the work directory is kept at $work" >&2
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 1' HUP INT TERM

check()
{
    if [ "$1" -eq 0 ]; then
        echo "quality_score: ok   $2"
    else
        echo "quality_score: BAD  $2"
        bad=1
    fi
}

# The private store holds the manifest of the given store and one link for each of its
# files. The steer tool writes the vector the score tool steers with beside the links.
store="$work/store"
mkdir -p "$store" || exit 2
cp "$models/manifest.jsonl" "$store/" || exit 2
grep -o '"path":"[^"]*"' "$models/manifest.jsonl" | cut -d'"' -f4 | while read -r path; do
    if [ "${path#/}" = "$path" ]; then
        ln -s "$(realpath "$models/$path")" "$store/$path" || exit 2
    fi
done

# The vector: the mean difference of two pairs at one layer, under the name of an axis.
printf '%s\t%s\n' \
    'The garden was full of light and the children laughed all day.' \
    'The garden was gray and the children sat in silence.' \
    'She opened the letter and smiled at the good news inside.' \
    'She opened the letter and wept at the bad news inside.' >"$work/pairs2.tsv"
echo "quality_score: the trait mode writes the vector the score tool steers with"
"$build/aotx_steer_derive" --models "$store" --trait valence --pairs "$work/pairs2.tsv" --layers 12 \
    >"$work/trait.log" 2>&1
check $? "the trait mode on two pairs ends with status 0"
grep -E '^(set|trait) ' "$work/trait.log"

# Ten items with known answers, the answers spread over the four letters. Two items, q01
# and q05, state an answer the model does not choose. The plain score is then 0.8 exactly,
# and a kernel that trusts the answer field scores 1 and fails.
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    q01 'Which of these is a color?' 'seven' 'red' 'table' 'run' A \
    q02 'How many legs does a dog have?' 'two' 'three' 'four' 'six' C \
    q03 'Which animal says meow?' 'a dog' 'a cow' 'a cat' 'a horse' C \
    q04 'What is two plus two?' 'three' 'four' 'five' 'six' B \
    q05 'Which of these is a fruit?' 'an apple' 'a chair' 'a stone' 'a cloud' C \
    q06 'Which month comes after March?' 'January' 'April' 'June' 'October' B \
    q07 'Which of these is a day of the week?' 'Monday' 'August' 'summer' 'noon' A \
    q08 'What do bees make?' 'milk' 'honey' 'bread' 'wool' B \
    q09 'How many days are in one week?' 'five' 'six' 'seven' 'ten' C \
    q10 'Which of these is the largest animal?' 'a mouse' 'a cat' 'a horse' 'a whale' D \
    >"$work/tasks10.tsv"

echo "quality_score: the task mode on ten items at doses 0, 0.25 and 1"
"$build/aotx_quality_score" --models "$store" --tasks "$work/tasks10.tsv" --axis valence \
    --doses 0,0.25,1 --out "$work/out" --print-items >"$work/score.log" 2>&1
check $? "the task mode on ten items ends with status 0"
grep -E '^(set|tasks|letter|\{) ' "$work/score.log" | grep -v '^item '
grep -E '^\{' "$work/score.log"
grep -c '^letter [ABCD]: token [0-9]*$' "$work/score.log" | grep -qx 4
check $? "the four letters are one token each"
grep -q '^set .*tasks10.tsv: 10 texts, .* 1 passes$' "$work/score.log"
check $? "the ten items run in one pass"

# The capability file holds one line per dose. The score of each line equals the share of
# the item lines of that dose marked right, computed here from the printed lines. The
# largest letter differs between items, so the argmax reads the logits. The plain score is
# 0.8 exactly: the eight true items right and the two false keys wrong. The dose changes
# the logit of the largest letter of at least one item, else it did not reach the model.
# The changed letters are counted beside it.
python3 - "$work/out/capability.jsonl" "$work/score.log" <<'EOF'
import json, re, sys
lines = open(sys.argv[1]).read().splitlines()
if len(lines) != 3:
    print("quality_score: BAD  the capability file holds %d lines, not 3" % len(lines)); sys.exit(1)
items = {}
for line in open(sys.argv[2]).read().splitlines():
    m = re.match(r"item (\S+) at dose (\S+): answer ([A-D]), largest ([A-D]), (right|wrong), logit (\S+)$", line)
    if m:
        items.setdefault(float(m.group(2)), []).append((m.group(1), m.group(3), m.group(4), m.group(5), float(m.group(6))))
bad = 0
for line, dose in zip(lines, (0.0, 0.25, 1.0)):
    row = json.loads(line)
    marks = items.get(dose, [])
    share = sum(1 for item in marks if item[3] == "right") / len(marks) if marks else -1.0
    same = (row["axis"] == "valence" and row["dose"] == dose and row["items"] == 10 and len(marks) == 10
            and all((a == l) == (mark == "right") for _, a, l, mark, _ in marks)
            and len(set(l for _, _, l, _, _ in marks)) > 1
            and 0.0 <= row["score"] <= 1.0 and abs(row["score"] - share) < 1e-9)
    print("quality_score: %s  the line at dose %g holds the score %.9g, the share of the ten item lines marked right (%.9g)"
          % ("ok " if same else "BAD", dose, row["score"], share))
    bad |= not same
plain = json.loads(lines[0])["score"]
wrong = sorted(name for name, _, _, mark, _ in items.get(0.0, []) if mark == "wrong")
same = abs(plain - 0.8) < 1e-9 and wrong == ["q01", "q05"]
print("quality_score: %s  the ten items score %.9g at dose 0, 0.8 exactly, the false keys q01 and q05 wrong (%s)" % ("ok " if same else "BAD", plain, ",".join(wrong)))
bad |= not same
moved = sum(1 for one, two in zip(items.get(0.0, []), items.get(1.0, [])) if abs(one[4] - two[4]) > 1e-6)
changed = sum(1 for one, two in zip(items.get(0.0, []), items.get(1.0, [])) if one[2] != two[2])
same = moved >= 1 and len(items.get(1.0, [])) == 10
print("quality_score: %s  the dose 1 changes the logit of the largest letter of %d of the ten items against dose 0, and the letter of %d"
      % ("ok " if same else "BAD", moved, changed))
bad |= not same
print("quality_score: the score at dose 0 is %.9g, at dose 0.25 %.9g and at dose 1 %.9g" % tuple(json.loads(l)["score"] for l in lines))
sys.exit(bad)
EOF
check $? "the three capability lines hold the scores of the item lines, the plain score is 0.8 exactly, the dose changes a letter logit"

# A one-item run and a 64-item run, the first 64 items of the task set. The mean of each is
# the share of its item lines marked right.
sed -n '2p' "$work/tasks10.tsv" >"$work/tasks1.tsv"
head -64 "$(dirname "$0")/fixtures/quality/tasks.tsv" >"$work/tasks64.tsv"
for count in 1 64; do
    echo "quality_score: the task mode on $count items at dose 0"
    "$build/aotx_quality_score" --models "$store" --tasks "$work/tasks$count.tsv" --axis valence \
        --doses 0 --out "$work/out$count" --print-items >"$work/score$count.log" 2>&1
    check $? "the task mode on $count items ends with status 0"
    python3 - "$work/out$count/capability.jsonl" "$work/score$count.log" "$count" <<'EOF'
import json, re, sys
lines = open(sys.argv[1]).read().splitlines()
marks = [m.group(1) for m in (re.match(r"item \S+ at dose 0: answer [A-D], largest [A-D], (right|wrong), logit \S+$", line)
                              for line in open(sys.argv[2]).read().splitlines()) if m]
count = int(sys.argv[3])
row = json.loads(lines[0]) if len(lines) == 1 else {}
share = marks.count("right") / len(marks) if marks else -1.0
same = len(lines) == 1 and row.get("items") == count and len(marks) == count and abs(row.get("score", -1.0) - share) < 1e-9
print("quality_score: %s  the %d-item line holds the score %.9g, the share of its %d item lines marked right (%.9g)"
      % ("ok " if same else "BAD", count, row.get("score", -1.0), len(marks), share))
sys.exit(not same)
EOF
    check $? "the $count-item run asserts its mean"
done

# The pair mode on a four-pair fixture: two pairs where b is plainly better, one where a is,
# and one tie of two near-identical replies. The rubric holds four items.
printf '%s\t%s\n' \
    on_topic 'Does the reply stay on the topic of the conversation?' \
    answers 'Does the reply answer what the user asked in the last turn?' \
    concise 'Is the reply concise, with no padding or filler?' \
    not_repetitive 'Does the reply avoid repeating itself or earlier replies?' >"$work/rubric4.tsv"
python3 - "$work/pairs4.jsonl" <<'EOF'
import json, sys
pairs = [
    {"name": "better-b-1", "turns": [
        {"user": "What is the capital of France?", "a": "Cats are nice. Cats are nice. Cats are nice. I do not know what you mean.", "b": "The capital of France is Paris."},
        {"user": "And of Italy?", "a": "Cats are nice. Cats are nice. Cats are nice.", "b": "The capital of Italy is Rome."}]},
    {"name": "better-b-2", "turns": [
        {"user": "Give me two tips for sleeping well.", "a": "No.", "b": "Keep a fixed bedtime, and keep screens out of the bedroom."}]},
    {"name": "better-a", "turns": [
        {"user": "How many days are in a week?", "a": "A week has seven days.", "b": "Bananas bananas bananas bananas. I will not talk about weeks. Bananas."}]},
    {"name": "tie", "turns": [
        {"user": "Name one color.", "a": "Red is one color.", "b": "Red is one color!"}]},
]
with open(sys.argv[1], "w") as out:
    for pair in pairs:
        out.write(json.dumps(pair) + "\n")
EOF
echo "quality_score: the pair mode on four pairs with the blinded output"
"$build/aotx_quality_score" --models "$store" --pairs "$work/pairs4.jsonl" --rubric "$work/rubric4.tsv" \
    --out "$work/pairs-out" --blind 1 >"$work/pairs.log" 2>&1
check $? "the pair mode on four pairs ends with status 0"
grep -E '^(pairs|answer|pair|\{)' "$work/pairs.log"
grep -cE '^answer (yes|no): token [0-9]+$' "$work/pairs.log" | grep -qx 2
check $? "the words yes and no are one token each"
# The summary line: two wins, one loss and one tie over the four pairs and the win rate
# 0.625. The scores are in nats of log-odds, with the margin 0.5 stated. The Wilson interval is
# computed here again at the tool's z within 1e-6. Every item of the rubric has a rate. The
# pair of two near-identical replies is a tie. The blinded transcripts, with the key applied
# back, give the input file.
python3 - "$work/pairs-out" "$work/pairs4.jsonl" <<'EOF'
import json, math, re, sys
out, source = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(out + "/pairs.jsonl").read().splitlines()]
pairs, summary = [row for row in rows if "summary" not in row], [row for row in rows if "summary" in row]
items = ["on_topic", "answers", "concise", "not_repetitive"]
bad = 0
def show(ok, text):
    global bad
    print("quality_score: %s  %s" % ("ok " if ok else "BAD", text)); bad |= not ok
show(len(pairs) == 4 and len(summary) == 1 and rows[-1] is summary[0], "the pairs file holds four pair lines and then the summary line")
s = summary[0]
results = {p["name"]: p["result"] for p in pairs}
show(s["pairs"] == 4 and s["wins"] == 2 and s["ties"] == 1 and abs(s["win_rate"] - 0.625) < 1e-9
     and results == {"better-b-1": "win", "better-b-2": "win", "better-a": "loss", "tie": "tie"},
     "the summary holds 2 wins and 1 tie over 4 pairs with the win rate 0.625, and the pair lines are win, win, loss, tie")
margin = s.get("margin", 0.0)
show(s.get("unit") == "nats" and abs(margin - 0.5) < 1e-6,
     "the summary states the unit nats and the margin %.9g" % margin)
z, n, w = 1.644853627, 4.0, 0.625
center, half = (w + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(w * (1 - w) / n + z * z / (4 * n * n)) / (1 + z * z / n)
show(abs(s["wilson_low"] - (center - half)) < 1e-6 and abs(s["wilson_high"] - (center + half)) < 1e-6,
     "the interval [%.9g, %.9g] is the Wilson interval at z 1.644853627, computed here as [%.9g, %.9g]" % (s["wilson_low"], s["wilson_high"], center - half, center + half))
show(list(s["items"].keys()) == items and all(0.0 <= v <= 1.0 for v in s["items"].values()),
     "the summary holds a rate for each of the four items: %s" % ", ".join("%s %.9g" % kv for kv in s["items"].items()))
# A score is the log-odds of yes in nats: finite, of either sign. The side score is the
# mean of the item scores within the print precision.
show(all(list(p["a"].keys()) == items and list(p["b"].keys()) == items
         and all(math.isfinite(v) for v in list(p["a"].values()) + list(p["b"].values()))
         and abs(sum(p["a"].values()) / len(items) - p["a_score"]) < 1e-5
         and abs(sum(p["b"].values()) / len(items) - p["b_score"]) < 1e-5 for p in pairs),
     "every pair line holds finite log-odds of both sides, and each side score is their mean")
gaps = {p["name"]: p["b_score"] - p["a_score"] for p in pairs}
show(all((gaps[name] > margin) == (results[name] == "win") and (gaps[name] < -margin) == (results[name] == "loss") for name in gaps),
     "every result follows the gap of the side scores against the margin: %s" % ", ".join("%s %+.4g" % kv for kv in gaps.items()))
tie = [p for p in pairs if p["name"] == "tie"][0]
show(tie["result"] == "tie" and abs(tie["a_score"] - tie["b_score"]) <= margin,
     "the two near-identical replies are a tie inside the margin (a %.9g, b %.9g nats)" % (tie["a_score"], tie["b_score"]))
key = {}
for line in open(out + "/pairs-key.tsv").read().splitlines()[1:]:
    number, name, x, y = line.split("\t")
    key[int(number)] = (name, x, y)
restored = []
for block in re.split(r"^## pair ", open(out + "/pairs-blind.md").read(), flags=re.M)[1:]:
    head, body = block.split("\n", 1)
    number, name = head.split(": ", 1)
    turns = re.findall(r"\*\*user:\*\* (.*?)\n\n\*\*X:\*\* (.*?)\n\n\*\*Y:\*\* (.*?)\n\n", body, flags=re.S)
    _, x, y = key[int(number)]
    restored.append({"name": name, "turns": [{"user": u, x: xt, y: yt} for u, xt, yt in turns]})
original = [json.loads(line) for line in open(source).read().splitlines()]
show(restored == original and {v[1] for v in key.values()} <= {"a", "b"} and all(v[1] != v[2] for v in key.values()),
     "the key applied to the blinded transcripts restores the input file (X is %s)" % ",".join(key[n][1] for n in sorted(key)))
sys.exit(bad)
EOF
check $? "the summary line, the item rates, the tie and the blind key hold"

# Two replies that are both good: the chance of yes saturates near one on both sides, so
# a score in chances gave a tie. The log-odds differ by more than the margin, and the
# better reply wins. The one-pair run asserts the saturation and the win from its own
# figures.
printf '%s\n' '{"name":"both-good","turns":[{"user":"What is the capital of France?","a":"Paris","b":"The capital of France is Paris."}]}' >"$work/pairs-good.jsonl"
echo "quality_score: the pair mode on one pair of two good replies"
"$build/aotx_quality_score" --models "$store" --pairs "$work/pairs-good.jsonl" --rubric "$work/rubric4.tsv" \
    --out "$work/pairs-good-out" >"$work/pairs-good.log" 2>&1
check $? "the pair mode on one pair ends with status 0"
grep -E '^(pair|\{)' "$work/pairs-good.log"
python3 - "$work/pairs-good-out/pairs.jsonl" <<'EOF'
import json, math, sys
rows = [json.loads(line) for line in open(sys.argv[1]).read().splitlines()]
pair, summary = rows[0], rows[-1]
chance = lambda nats: 1.0 / (1.0 + math.exp(-nats))
a, b = chance(pair["a_score"]), chance(pair["b_score"])
gap = pair["b_score"] - pair["a_score"]
ok = a > 0.9 and b > 0.9 and abs(a - b) < 0.1 and gap > summary["margin"] and pair["result"] == "win" and summary["wins"] == 1
print("quality_score: %s  both chances of yes saturate (a %.4f, b %.4f, apart by %.4f) and the log-odds gap %.4g nats over the margin gives a win, not a tie (%s)"
      % ("ok " if ok else "BAD", a, b, abs(a - b), gap, pair["result"]))
sys.exit(not ok)
EOF
check $? "two good replies whose chances saturate near one are a win on the log-odds"

# A rubric of seventeen items and a pairs line without side b are refused by their number,
# before any model is placed.
python3 -c 'for i in range(17): print("item%d\tIs the reply good?" % i)' >"$work/rubric17.tsv"
"$build/aotx_quality_score" --models "$store" --pairs "$work/pairs4.jsonl" --rubric "$work/rubric17.tsv" \
    --out "$work/pairs-bad" >"$work/pairs-bad.log" 2>&1
status=$?
check "$(test "$status" -eq 2 && grep -q 'more than 16 items' "$work/pairs-bad.log"; echo $?)" \
      "the rubric of seventeen items ends the run with status 2"
head -1 "$work/pairs4.jsonl" >"$work/pairs-bad.jsonl"
printf '%s\n' '{"name":"no-b","turns":[{"user":"Hello.","a":"Hello."}]}' >>"$work/pairs-bad.jsonl"
"$build/aotx_quality_score" --models "$store" --pairs "$work/pairs-bad.jsonl" --rubric "$work/rubric4.tsv" \
    --out "$work/pairs-bad" >"$work/pairs-bad2.log" 2>&1
status=$?
check "$(test "$status" -eq 2 && grep -q 'line 2 is malformed' "$work/pairs-bad2.log"; echo $?)" \
      "the pairs line without side b ends the run with status 2 and is named by its number"
grep -h 'malformed\|more than' "$work/pairs-bad.log" "$work/pairs-bad2.log"
test ! -e "$work/pairs-bad/pairs.jsonl"
check $? "a refused pair run writes no pairs file"

# A task line with six fields is refused by its number, before any model is placed, and no
# capability line is written. An answer that is not one letter A to D is refused the same way.
echo "quality_score: a malformed task line is refused"
head -2 "$work/tasks10.tsv" >"$work/tasks-bad.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' q03 'Which animal says meow?' 'a dog' 'a cow' 'a cat' 'a horse' >>"$work/tasks-bad.tsv"
"$build/aotx_quality_score" --models "$store" --tasks "$work/tasks-bad.tsv" --axis valence \
    --doses 0 --out "$work/out-bad" >"$work/bad.log" 2>&1
status=$?
check "$(test "$status" -eq 2 && grep -q 'line 3 is malformed' "$work/bad.log"; echo $?)" \
      "the six-field line ends the run with status 2 and is named by its number"
grep 'malformed' "$work/bad.log"
head -2 "$work/tasks10.tsv" >"$work/tasks-bad2.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' q03 'Which animal says meow?' 'a dog' 'a cow' 'a cat' 'a horse' E >>"$work/tasks-bad2.tsv"
"$build/aotx_quality_score" --models "$store" --tasks "$work/tasks-bad2.tsv" --axis valence \
    --doses 0 --out "$work/out-bad" >"$work/bad2.log" 2>&1
status=$?
check "$(test "$status" -eq 2 && grep -q 'line 3 is malformed' "$work/bad2.log"; echo $?)" \
      "the answer E ends the run with status 2 and is named by its number"
test ! -e "$work/out-bad/capability.jsonl"
check $? "a refused run writes no capability line"

if [ "$bad" -ne 0 ]; then
    echo "quality_score: FAIL"
    exit 1
fi
echo "quality_score: PASS"
exit 0
