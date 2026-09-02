#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Check the axis mode and the calibrate mode of the steer tool on a private copy of a store.
# The axis mode runs with a named probe layer, a standardization set of its own, its
# readouts printed and a pass of 64 sequences. The calibrate mode runs on two axes at one
# layer, so the composite it measures differs from the raw vectors.
# Inputs: a build directory and a model store. Outputs: the tool lines and one line per check.
# Exit codes: 0 pass, 1 a check failed, 2 usage or environment error.
set -u
set -o pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: affect_axis.sh <build> <models>" >&2
    exit 2
fi
build="$1"
models="$2"
here="$(cd "$(dirname "$0")" && pwd)"
fixtures="$here/fixtures/affect"
for item in "$build/aotx_steer_derive" "$build/aotx_boot"; do
    if [ ! -x "$item" ]; then
        echo "affect_axis: the build program is not executable: $item" >&2
        exit 2
    fi
done
if [ ! -r "$models/manifest.jsonl" ]; then
    echo "affect_axis: the model store has no manifest: $models" >&2
    exit 2
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/aotx-affect-axis-XXXXXX") || exit 2
bad=0

# A pass removes the work directory. A failure keeps the store, the logs and the journal
# and names the directory, so that the difference can be examined.
finish()
{
    local status=$?
    trap - EXIT
    if [ "$status" -eq 0 ]; then
        rm -rf "$work"
    else
        echo "affect_axis: the work directory is kept at $work" >&2
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 1' HUP INT TERM

check()
{
    if [ "$1" -eq 0 ]; then
        echo "affect_axis: ok   $2"
    else
        echo "affect_axis: BAD  $2"
        bad=1
    fi
}

# The private store holds the manifest of the given store and one link for each of its
# files. The tool writes its vectors, probes and catalog lines beside the links, never into
# the given store.
store="$work/store"
mkdir -p "$store" || exit 2
cp "$models/manifest.jsonl" "$store/" || exit 2
grep -o '"path":"[^"]*"' "$models/manifest.jsonl" | cut -d'"' -f4 | while read -r path; do
    if [ "${path#/}" = "$path" ]; then
        ln -s "$(realpath "$models/$path")" "$store/$path" || exit 2
    fi
done

# The small fixtures: two pairs, two neutral texts and two held-out pairs, one pass each.
# One neutral text gives a pass of one sequence.
printf '%s\t%s\n' \
    'The pond lay still under a pale evening sky.' \
    'The siren blared and the whole street ran for cover.' \
    'The cat dozed on the warm stone step all afternoon.' \
    'The pot boiled over and everyone rushed to the stove.' >"$work/pairs2.tsv"
printf '%s\n' 'The shop opens at eight on weekdays.' 'A week has seven days.' >"$work/neutral2.txt"
printf '%s\n' 'The bridge has four lanes.' >"$work/neutral1.txt"
printf '%s\t%s\n' \
    'User: How is the lake? Assistant: Still and quiet in the evening light.' \
    'User: How is the lake? Assistant: A boat capsized and everyone is shouting for help!' \
    'User: How is the office? Assistant: Calm, with the printers humming softly.' \
    'User: How is the office? Assistant: The alarm went off and everyone is racing to the exits!' \
    >"$work/heldout2.tsv"

# The first run names the probe layer 8, not the last of the list. The second run leaves
# it to its default, the last of the list, 12.
echo "affect_axis: the axis mode on two pairs (one pass of a few sequences)"
"$build/aotx_steer_derive" --models "$store" --axis arousal --pairs "$work/pairs2.tsv" \
    --neutral "$work/neutral2.txt" --heldout "$work/heldout2.tsv" --layers 8,12 --probe-layer 8 \
    --print-readouts --print-direction >"$work/axis-small.log" 2>&1
check $? "the axis mode on the two-pair fixture ends with status 0"
grep -E '^(set|axis|readout) ' "$work/axis-small.log"
check "$(test -f "$store/arousal.aotxvec" && test -f "$store/affect/arousal.aotxprb"; echo $?)" \
      "the vector file and the probe file of arousal are in the store"
grep -q '^axis arousal: probe layer 8, ' "$work/axis-small.log"
check $? "the run names the probe layer 8"
chosen=$(sed -n 's/^axis arousal: layer \([0-9]*\) chosen,.*/\1/p' "$work/axis-small.log")
check "$(test "$chosen" = 8 || test "$chosen" = 12; echo $?)" "the vector of arousal holds the layer $chosen"

# The standardization set is the dialogue set, so the mean and the scale of the probe
# come from the form of the held-out set. The vector of valence goes to the layer the
# vector of arousal holds. The composite of the two is then not a copy of the raw vectors.
echo "affect_axis: the axis mode on the valence fixture (passes at the text and row bounds)"
"$build/aotx_steer_derive" --models "$store" --axis valence --pairs "$fixtures/valence.tsv" \
    --neutral "$fixtures/neutral.txt" --heldout "$fixtures/heldout-valence.tsv" --layers "$chosen" \
    --standardise "$fixtures/neutral-dialogue.txt" --print-readouts --print-direction >"$work/axis-full.log" 2>&1
check $? "the axis mode on the valence fixture ends with status 0"
grep -E '^(set|axis) ' "$work/axis-full.log"
check "$(grep -c '^set .* 64 texts,' "$work/axis-full.log" | grep -qx 3; echo $?)" \
      "the three sets hold 64 texts each"
check "$(grep '^set .*valence.tsv: 64 texts,' "$work/axis-full.log" \
           | sed -n 's/^set .*: 64 texts, \([0-9]*\) tokens, .* \([0-9]*\) passes$/\1 \2/p' \
           | awk '{ exit !($1 > 512 && $2 >= 2) }'; echo $?)" \
      "the pair set runs in more than one pass at the row bound"
grep -q '^set .*neutral-dialogue.txt: 64 texts,' "$work/axis-full.log"
check $? "the standardization set is the dialogue set"
grep -q "^axis valence: probe layer $chosen, .*standardized on .*neutral-dialogue.txt" "$work/axis-full.log"
check $? "the run names the probe layer $chosen and the standardization set"

# The catalog line of each probe holds the accuracy the head of its file holds. The layer
# of the line is the probe layer of its run. The head holds the mean and the scale the run
# printed for the probe layer. The two are computed here again from the readouts the run
# printed for the standardization texts. They are the mean and the standard deviation,
# with one degree of freedom taken. The direction of the file has unit length and equals
# the printed device row of the probe layer in its first eight values.
python3 - "$store" "$chosen" "$work/axis-small.log" "$work/axis-full.log" <<'EOF'
import json, math, re, struct, sys
store = sys.argv[1]
bad = 0
lines = open(store + "/probes.jsonl").read().splitlines()
if len(lines) != 2:
    print("affect_axis: BAD  the probe catalog holds %d lines, not 2" % len(lines)); sys.exit(1)
expected = {"arousal": 8, "valence": int(sys.argv[2])}
single = lambda text: struct.unpack("<f", struct.pack("<f", float(text)))[0]
printed, readouts, directions = {}, {}, {}
for path in sys.argv[3:]:
    for line in open(path).read().splitlines():
        m = re.match(r"axis (\w+): probe layer (\d+), accuracy (\S+), agreement (\S+), mean (\S+), scale (\S+),", line)
        if m:
            printed[m.group(1)] = (int(m.group(2)), single(m.group(5)), single(m.group(6)))
        m = re.match(r"readout (\w+): text (\d+), layer (\d+), (\S+)$", line)
        if m:
            readouts.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), single(m.group(4))))
        m = re.match(r"direction (\w+): layer (\d+), norm (\S+), (.*)$", line)
        if m:
            directions[m.group(1)] = (int(m.group(2)), float(m.group(3)), [single(v) for v in m.group(4).split()])
for line in lines:
    row = json.loads(line)
    body = open(store + "/" + row["file"], "rb").read()
    magic, hidden, layer, axis, accuracy, agreement, mean, scale, reserved = struct.unpack("<8sIIIffffI", body[:40])
    direction = struct.unpack("<%df" % hidden, body[40:40 + 4 * hidden])
    catalog = single(row["accuracy"])
    same = magic == b"AOTXPRB1" and axis == row["axis"] and layer == row["layer"] and catalog == accuracy and reserved == 0
    same = same and layer == expected[row["name"]] and printed.get(row["name"]) == (layer, mean, scale)
    print("affect_axis: %s  the catalog line of %s equals its head at the probe layer (accuracy %.9g, agreement %.9g, layer %d, mean %.9g, scale %.9g)"
          % ("ok " if same else "BAD", row["name"], accuracy, agreement, layer, mean, scale))
    bad |= not same
    values = [value for number, at, value in readouts.get(row["name"], []) if at == layer]
    n = len(values)
    again_mean = sum(values) / n if n else float("nan")
    again_scale = math.sqrt(sum((v - again_mean) ** 2 for v in values) / (n - 1)) if n > 1 else float("nan")
    close = lambda a, b: abs(a - b) <= 1e-5 * max(abs(a), abs(b), 1.0)
    same = n == len(readouts.get(row["name"], [])) and n >= 2 and close(again_mean, mean) and close(again_scale, scale)
    print("affect_axis: %s  the mean and the scale of %s are the mean and the standard deviation of its %d printed readouts at layer %d (%.9g, %.9g)"
          % ("ok " if same else "BAD", row["name"], n, layer, again_mean, again_scale))
    bad |= not same
    length = math.sqrt(sum(v * v for v in direction))
    same = len(body) == 40 + 4 * hidden and abs(length - 1.0) < 1e-4
    print("affect_axis: %s  the direction of %s has unit length (%.9g over %d values)" % ("ok " if same else "BAD", row["name"], length, hidden))
    bad |= not same
    at, norm, first = directions.get(row["name"], (-1, 0.0, []))
    same = at == layer and abs(norm - 1.0) < 1e-4 and len(first) == 8 and list(direction[:8]) == first
    print("affect_axis: %s  the first eight values of the file of %s are the printed values of the device row at layer %d (norm %.9g)"
          % ("ok " if same else "BAD", row["name"], at, norm))
    bad |= not same
sys.exit(bad)
EOF
check $? "every catalog line equals its file head at the probe layer, the mean and the scale come from the printed readouts, the direction has unit length and is the printed device row"

echo "affect_axis: a second run refuses the existing vector by name"
"$build/aotx_steer_derive" --models "$store" --axis valence --pairs "$fixtures/valence.tsv" \
    --neutral "$fixtures/neutral.txt" --heldout "$fixtures/heldout-valence.tsv" --layers 8,12 \
    >"$work/axis-again.log" 2>&1
status=$?
check "$(test "$status" -eq 1 && grep -q '^the steer vector valence is already in the model store' "$work/axis-again.log"; echo $?)" \
      "the second run ends with status 1 and names the vector"
grep -c '' "$store/probes.jsonl" | grep -qx 2
check $? "the second run added no catalog line"

# A boot on the store loads the two rows and states the count. One say line under
# affect.on gives a trace whose flags mark the loaded rows.
echo "affect_axis: a boot on the store loads the probe rows"
journal="$work/journal"
mkdir -p "$journal" || exit 2
mkfifo "$work/feed" || exit 2
printf '%s\n' 'affect.on = 1' 'sample.temperature = 0' 'sample.seed = 7' \
    'derive.list = tokens,pages,affect' >"$work/boot.settings"
"$build/aotx_boot" --settings "$work/boot.settings" --journal "$journal" --models "$store" \
    --ticks 32 <"$work/feed" >"$work/boot.log" 2>&1 &
boot=$!
exec 9>"$work/feed"
printf '%s\n' 'say name one color and nothing else' >&9
exec 9>&-
wait "$boot"
check $? "the boot ends with status 0"
grep -q '^probes: 2 rows$' "$work/boot.log"
check $? "the boot states the two loaded rows"
grep '^probes:' "$work/boot.log"
boot_dir=$(find "$journal" -mindepth 1 -maxdepth 1 -type d -name '[0-9a-f][0-9a-f]*' | sort | tail -1)
test -n "$boot_dir" && test -f "$boot_dir/affect.jsonl" && grep -q '"flags":1' "$boot_dir/affect.jsonl"
check $? "the trace of the turn marks the loaded rows"
if [ -n "$boot_dir" ] && [ -f "$boot_dir/affect.jsonl" ]; then
    grep '"kind":"trace"' "$boot_dir/affect.jsonl" | head -2
fi

# The calibrate mode on one neutral text (passes of one sequence) and on the neutral
# fixture (passes at the bounds). Each run appends one line whose every figure is finite.
echo "affect_axis: the calibrate mode on one text and on the neutral fixture"
"$build/aotx_steer_derive" --models "$store" --calibrate --axes arousal \
    --neutral "$work/neutral1.txt" --dose 0.5 >"$work/calibrate-small.log" 2>&1
check $? "the calibrate mode on one text ends with status 0"
grep -E '^(set|calibrate|probe|M|K|dose|perplexity|composite|calibration) ' "$work/calibrate-small.log"
"$build/aotx_steer_derive" --models "$store" --calibrate --axes valence,arousal \
    --neutral "$fixtures/neutral.txt" --dose 0.5 >"$work/calibrate-full.log" 2>&1
check $? "the calibrate mode on the neutral fixture ends with status 0"
grep -E '^(set|calibrate|probe|M|K|dose|perplexity|composite|calibration) ' "$work/calibrate-full.log"

# A set of 64 short texts fills one pass to the text bound: 64 sequences in one pass.
for i in $(seq 1 64); do echo "Item $i."; done >"$work/neutral64.txt"
"$build/aotx_steer_derive" --models "$store" --calibrate --axes valence,arousal \
    --neutral "$work/neutral64.txt" --dose 0.5 >"$work/calibrate-64.log" 2>&1
check $? "the calibrate mode on 64 short texts ends with status 0"
grep -E '^set ' "$work/calibrate-64.log"
check "$(grep -q '^set .*neutral64.txt: 64 texts, .* 1 passes' "$work/calibrate-64.log"; echo $?)" \
      "the 64 texts fill one pass at the text bound"
# Each line holds K of the composites and K_raw of the raw vectors. The head of each
# composite file holds the composite K of its axis over two, from the last line.
python3 - "$store" <<'EOF'
import json, math, struct, sys
store = sys.argv[1]
lines = open(store + "/affect/calibration.jsonl").read().splitlines()
if len(lines) != 3:
    print("affect_axis: BAD  the calibration file holds %d lines, not 3" % len(lines)); sys.exit(1)
def numbers(value):
    if isinstance(value, bool): return []
    if isinstance(value, (int, float)): return [value]
    if isinstance(value, list): return [n for v in value for n in numbers(v)]
    if isinstance(value, dict): return [n for v in value.values() for n in numbers(v)]
    return []
bad = 0
for line in lines:
    row = json.loads(line)
    axes = len(row["axes"])
    rows = len(row["rows"])
    shape = (len(row["M"]) == rows and all(len(m) == axes for m in row["M"])
             and len(row["K"]) == axes and all(len(k) == axes for k in row["K"])
             and len(row["K_raw"]) == axes and all(len(k) == axes for k in row["K_raw"])
             and len(row["ratio"]) == axes and len(row["perplexity"]) == axes
             and len(row["perplexity_twice"]) == axes and len(row["layers"]) == axes
             and len(row["probe_layers"]) == rows and len(row["composite"]) == axes
             and row["delta"] > 0 and row["dominant"] in (0, 1) and row["orthogonal"] in (0, 1))
    figures = numbers(row)
    finite = all(math.isfinite(n) for n in figures)
    print("affect_axis: %s  the calibration line of %s holds M %dx%d, K %dx%d, K_raw %dx%d and %d finite figures (dominant %d, orthogonal %d)"
          % ("ok " if (finite and shape) else "BAD", ",".join(row["axes"]), rows, axes, axes, axes, axes, axes, len(figures),
             row["dominant"], row["orthogonal"]))
    bad |= not (finite and shape)
    # One axis alone has no other axis to orthogonalize against, so its composite is the
    # raw copy and K equals K_raw. Two axes at one layer are orthogonalized, so K differs.
    apart = row["K"] != row["K_raw"]
    same = apart == (axes == 2)
    print("affect_axis: %s  the line of %s holds K %s K_raw (%d axes)" % ("ok " if same else "BAD", ",".join(row["axes"]), "apart from" if apart else "equal to", axes))
    bad |= not same
last = json.loads(lines[-1])
def vector(path):
    body = open(path, "rb").read()
    magic, hidden, layers, potency, reserved = struct.unpack("<8sIIfI", body[:24])
    held = struct.unpack("<%dI" % layers, body[24:24 + 4 * layers])
    return magic, hidden, layers, potency, reserved, held, struct.unpack("<%df" % (layers * hidden), body[24 + 4 * layers:])
for j, name in enumerate(last["axes"]):
    magic, hidden, layers, potency, reserved, held, values = vector(store + "/" + last["composite"][j])
    want = struct.unpack("<f", struct.pack("<f", 0.5 * last["K"][j][j]))[0]
    same = magic == b"AOTXSTV1" and layers == len(last["layers"][j]) and potency == want and reserved == 0
    print("affect_axis: %s  the composite file of %s holds %d layers and the potency %.9g (K %.9g over two)"
          % ("ok " if same else "BAD", name, layers, potency, last["K"][j][j]))
    bad |= not same
    raw = vector(store + "/" + name + ".aotxvec")
    gap = max(abs(a - b) for a, b in zip(values, raw[6]))
    top = max(abs(a) for a in raw[6])
    same = held == raw[5] and len(values) == len(raw[6]) and gap > 1e-4 * top
    print("affect_axis: %s  the composite of %s differs from its raw vector at the layer %s (largest gap %.9g against the largest value %.9g)"
          % ("ok " if same else "BAD", name, ",".join(str(l) for l in held), gap, top))
    bad |= not same
sys.exit(bad)
EOF
check $? "the three calibration lines hold K, K_raw and every figure in its shape, finite, K apart from K_raw on two axes, the composite heads hold K over two and the composites differ from the raw vectors"
for name in valence arousal; do
    test -f "$store/affect/composite-$name.aotxvec"
    check $? "the composite file of $name is in the store"
done

# A set of 32 short pairs fills one pass of the axis mode to the text bound. The capture
# and the potency passes take its 64 sequences in one pass.
echo "affect_axis: the axis mode on 64 short texts (one pass at the text bound)"
for i in $(seq 1 32); do printf 'Item %s.\tThing %s.\n' "$i" "$i"; done >"$work/pairs64.tsv"
"$build/aotx_steer_derive" --models "$store" --axis sycophancy --pairs "$work/pairs64.tsv" \
    --neutral "$work/neutral2.txt" --heldout "$work/heldout2.tsv" --layers 12 >"$work/axis-64.log" 2>&1
check $? "the axis mode on 64 short texts ends with status 0"
grep -E '^(set|axis) ' "$work/axis-64.log"
check "$(grep -q '^set .*pairs64.tsv: 64 texts, .* 1 passes$' "$work/axis-64.log"; echo $?)" \
      "the 64 texts fill one pass at the text bound"

if [ "$bad" -ne 0 ]; then
    echo "affect_axis: FAIL"
    exit 1
fi
echo "affect_axis: PASS"
exit 0
