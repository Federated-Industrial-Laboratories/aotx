#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Check the task mode of the quality score tool on a private copy of a store. A ten-item
# task set with known answers runs at two doses. The score is computed again from the item
# lines, and a malformed task line is refused.
# Inputs: a build directory and a model store. Outputs: the tool lines and one line per check.
# Exit codes: 0 pass, 1 a check failed, 2 usage or environment error.
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

# Ten items with known answers, the answers spread over the four letters.
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    q01 'Which of these is a color?' 'seven' 'red' 'table' 'run' B \
    q02 'How many legs does a dog have?' 'two' 'three' 'four' 'six' C \
    q03 'Which animal says meow?' 'a dog' 'a cow' 'a cat' 'a horse' C \
    q04 'What is two plus two?' 'three' 'four' 'five' 'six' B \
    q05 'Which of these is a fruit?' 'an apple' 'a chair' 'a stone' 'a cloud' A \
    q06 'Which month comes after March?' 'January' 'April' 'June' 'October' B \
    q07 'Which of these is a day of the week?' 'Monday' 'August' 'summer' 'noon' A \
    q08 'What do bees make?' 'milk' 'honey' 'bread' 'wool' B \
    q09 'How many days are in one week?' 'five' 'six' 'seven' 'ten' C \
    q10 'Which of these is the largest animal?' 'a mouse' 'a cat' 'a horse' 'a whale' D \
    >"$work/tasks10.tsv"

echo "quality_score: the task mode on ten items at doses 0 and 0.25"
"$build/aotx_quality_score" --models "$store" --tasks "$work/tasks10.tsv" --axis valence \
    --doses 0,0.25 --out "$work/out" --print-items >"$work/score.log" 2>&1
check $? "the task mode on ten items ends with status 0"
grep -E '^(set|tasks|letter|\{) ' "$work/score.log" | grep -v '^item '
grep -E '^\{' "$work/score.log"
grep -c '^letter [ABCD]: token [0-9]*$' "$work/score.log" | grep -qx 4
check $? "the four letters are one token each"
grep -q '^set .*tasks10.tsv: 10 texts, .* 1 passes$' "$work/score.log"
check $? "the ten items run in one pass"

# The capability file holds one line per dose. The score of each line equals the share of
# the item lines of that dose marked right, computed here from the printed lines. The
# largest letter differs between items, so the argmax reads the logits.
python3 - "$work/out/capability.jsonl" "$work/score.log" <<'EOF'
import json, re, sys
lines = open(sys.argv[1]).read().splitlines()
if len(lines) != 2:
    print("quality_score: BAD  the capability file holds %d lines, not 2" % len(lines)); sys.exit(1)
items = {}
for line in open(sys.argv[2]).read().splitlines():
    m = re.match(r"item (\S+) at dose (\S+): answer ([A-D]), largest ([A-D]), (right|wrong)$", line)
    if m:
        items.setdefault(float(m.group(2)), []).append((m.group(1), m.group(3), m.group(4), m.group(5)))
bad = 0
for line, dose in zip(lines, (0.0, 0.25)):
    row = json.loads(line)
    marks = items.get(dose, [])
    share = sum(1 for item in marks if item[3] == "right") / len(marks) if marks else -1.0
    same = (row["axis"] == "valence" and row["dose"] == dose and row["items"] == 10 and len(marks) == 10
            and all((a == l) == (mark == "right") for _, a, l, mark in marks)
            and len(set(l for _, _, l, _ in marks)) > 1
            and 0.0 <= row["score"] <= 1.0 and abs(row["score"] - share) < 1e-9)
    print("quality_score: %s  the line at dose %g holds the score %.9g, the share of the ten item lines marked right (%.9g)"
          % ("ok " if same else "BAD", dose, row["score"], share))
    bad |= not same
print("quality_score: the score at dose 0 is %.9g and at dose 0.25 is %.9g" % (json.loads(lines[0])["score"], json.loads(lines[1])["score"]))
sys.exit(bad)
EOF
check $? "the two capability lines hold the scores of the item lines"

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
