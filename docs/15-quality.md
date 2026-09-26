<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Quality measurement

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Quality instrumentation provides per-turn observations, paired-reply scoring and task-set measurements.
It uses the local model and recorded inputs without a network scoring service.
The figures describe their input set; they do not establish general conversational reliability.

The runtime quality stream requires `AOTX_AFFECT=ON` and `quality.on=1`.
Both quality and affect runtime settings default to zero.
Standalone scoring tools have their own input and asset requirements.

| Instrument | Result |
| --- | --- |
| Quality stream | Coherence, repetition, refusal proxy and available guard observations for each turn. |
| Pair mode | Model-scored rubric comparison plus optional blind transcripts. |
| Task mode | Answer accuracy at each declared control dose. |

Fitted assets still require exact bindings.
A measurement tool can inspect a candidate without establishing a qualified runtime control.
See [control qualification](37-control-bindings.md#qualification-component).

<details>
<summary>On this page</summary>

- [The quality stream](#the-quality-stream)
- [The score tool](#the-score-tool)
- [The pair script](#the-pair-script)
- [The fixtures](#the-fixtures)
- [What the instrument is for](#what-the-instrument-is-for)
- [Interpretation limits](#interpretation-limits)

</details>

## The quality stream

`quality.jsonl` holds one line for each turn whose sequence opened while `quality.on` was 1.
Three figures of the line describe the reply. `coherence_prompt` is the cosine of the reply and
the message of the same turn. `repetition` is one less the distinct share of the token trigrams
of the reply. `refusal` is 1 when the reply holds a phrase of the refusal list; it is a weak
proxy. The section "The quality stream" of `docs/14-affect.md` states every field of the line,
its flags and the bodies the stream refuses.

The pair script prints the mean `coherence_prompt` and the mean `repetition` of each side. A
comparison of two runs reads those means beside the win rate.

## The score tool

`aotx_quality_score` opens the language model of a store and nothing else: no state, no journal
and no affect kernel. It has two modes. The pair mode scores the two sides of each pair on a
rubric. The task mode measures what a steer vector costs the model on a task set.

```text
usage: aotx_quality_score --models DIR --tasks FILE --axis NAME --doses LIST --out DIR
                          [--role NAME] [--print-items]
       aotx_quality_score --models DIR --pairs FILE --rubric FILE --out DIR [--blind 1]
                          [--role NAME]
```

Both modes read the same store wrap table as the device prompt path.
A generation prompt closes an empty thinking block when the table defines one.
A stored assistant turn uses only its assistant head and tail, with no generation span.
One prefill gives the logits of the last row of each query. The score reads named answer tokens.
The tool refuses a wrap that did not pass the model load check.

The exit status is 2 for a refused input, 1 for a run that failed, and 0 otherwise. The
program finds the token of each answer word once and prints it. It refuses to run when an
answer word is more than one token.

### The pair mode

```text
aotx_quality_score --models models --pairs build/pairs/pairs.jsonl \
  --rubric tests/fixtures/quality/rubric.tsv --out build/pairs/score --blind 1
```

| option | effect |
| --- | --- |
| `--models <dir>` | the model store the program reads |
| `--pairs <file>` | the pairs file, one conversation on each line |
| `--rubric <file>` | the rubric, one item on each line |
| `--out <dir>` | the directory that takes the output files |
| `--blind 1` | write the blind transcripts and their key as well |
| `--role <name>` | the model role to open; the default is `language` |

For each pair, each side and each item the program builds one query. The query is the
conversation with the replies of that side, then the question of the item and the line
`Answer yes or no.`. The score of a side on an item is the log-odds of yes at the last row, in
nats. That is the logit of `yes` less the logit of `no`. The program applies no sigmoid, so two
answers that are both near certain still differ. Each side is scored alone, so the order of the
two sides changes no score.

The side score is the mean of the log-odds over the items. Side b wins a pair when its side
score is above the side score of side a by more than the margin. It loses when its side score
is under the other by more than the margin. Every other pair is a tie, counted one half. The
margin is 0.5 nats, a constant of the program, and the summary line states it with the unit.

The win rate is the wins plus one half of the ties, over the pairs.
The program prints Wilson bounds with the normal quantile for a nominal 90 percent interval.
It prints the win rate of each
item under the same rule as well. A gain on one item beside a loss on another is then visible.
Kernels compute the scores, the means, the results, the tally, the interval and the blind
order.

The bounds use the [Wilson formula](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).
Ties count as one half. Related prompts and ties do not establish independent binary observations.
These bounds describe the supplied set. They do not guarantee 90 percent coverage for other conversations.

A conversation over 400 tokens is cut. The cut keeps the first user line and the last turns
that fit under 400 tokens with it. When the first user line and the last turn do not fit
together, the reply of the last turn stands alone, cut from its front. The line of a cut pair
holds `cut` 1, and the summary line counts the cut pairs.

The program writes `<out>/pairs.jsonl`: one line for each pair, then one summary line. The
fields of a pair line are these:

| field | meaning |
| --- | --- |
| `name` | the name of the conversation |
| `cut` | 1 when the conversation was cut to the token bound |
| `a`, `b` | the log-odds of each item for that side, by item identity |
| `a_score`, `b_score` | the side scores |
| `result` | `win`, `loss` or `tie`, for side b |

The fields of the summary line are these:

| field | meaning |
| --- | --- |
| `summary` | 1; the mark of the summary line |
| `pairs`, `wins`, `ties` | the counts |
| `win_rate` | the win rate |
| `wilson_low`, `wilson_high` | Wilson bounds with a nominal 90 percent normal quantile |
| `unit`, `margin` | `nats` and the margin |
| `items` | the win rate of each item |
| `cut` | the count of the cut pairs |

With `--blind 1` the program writes `<out>/pairs-blind.md` and `<out>/pairs-key.tsv`. The
first holds the transcripts of every pair with the two sides in a shuffled order, named X and
Y. The second holds the shuffle: the pair number, the name, the side of X and the side of Y.
The order comes from a hash of the pair number under a fixed seed, so the two files of one run
agree. A person reads the first file and marks each pair, and opens the key after that.

The pairs file holds one conversation on each line, in this form:

```text
{"name":"garden-watering-plan","turns":[{"user":"...","a":"...","b":"..."}]}
```

The program refuses these inputs:

- a pairs file that does not open, holds no pair, or holds more than 256 pairs;
- a pair line without a name and 1 to 16 turns, each with the keys `user`, `a` and `b`;
- a pairs line of more than 65,536 bytes;
- a rubric that does not open, holds no item, or holds more than 16 items;
- an item identity that is not letters, digits, underscore and dash under 32 bytes;
- a question of 512 bytes or more, or a second tab on an item line;
- a turn that does not fit one block of 8,192 bytes, or a query that does not fit 8,192 bytes.

### The task mode

```text
aotx_quality_score --models models --tasks tests/fixtures/quality/tasks.tsv \
  --axis valence --doses 0,0.25,0.5 --out build/capability
```

| option | effect |
| --- | --- |
| `--models <dir>` | the model store the program reads |
| `--tasks <file>` | the task set |
| `--axis <name>` | the name of the steer vector each dose applies |
| `--doses <list>` | one to eight strengths, comma separated, each at or above zero |
| `--out <dir>` | the directory that takes `capability.jsonl` |
| `--role <name>` | the model role to open; the default is `language` |
| `--print-items` | print one line for each item at each dose |

The program builds one query for each item. The query is the question, the four choices with
the letters A to D, and the line `Answer with one letter.`. The last row of the query holds
the logits of the first letter of the answer. The score of an item is 1 when the largest of the four letter
logits is the letter of the answer, and 0 otherwise. A tie keeps the earlier letter. The score
of a dose is the mean over the items, in double.

Dose 0 runs the passes plain. Each other dose puts the vector in the first steer slot at that
strength. The program prints one line for each dose and adds the same line to
`<out>/capability.jsonl`, so the file keeps the lines of every run:

```text
{"axis":"valence","dose":0,"score":0.8,"items":200}
```

`--print-items` prints one line for each item at each dose. The line gives the answer, the
largest letter, the result and the logit of the largest letter. A dose that reaches the model
is therefore visible before it changes a letter.

The program refuses these inputs:

- a dose list that is empty, or that holds more than eight values;
- a dose that is under zero or is not a figure;
- a task file that does not open, or that holds no item;
- a line that does not hold seven fields, or that holds an empty field;
- an identity of 32 bytes or more, or an answer that is not one letter of A to D;
- a line of more than 8,192 bytes, or a query that does not fit 8,448 bytes;
- a task file of more than 256 items.

## The pair script

`tools/quality_pair.sh` runs the fixture conversation set twice and builds the pairs file:

```text
tools/quality_pair.sh <build> <models> <conversations> <out>
```

The first run has `affect.on` 0 and the second has it 1. Both runs have `quality.on` 1,
`affect.steer_gain` 0.25, `sample.temperature` 0.6, `sample.seed` 7 and `decode.reply_limit`
256. Each conversation runs in a boot of its own, so no conversation sees another. The language
role is the role the last calibration line of the store names, or `language` without one. The
embedding role loads beside it, for the quality stream.

Each user line goes to the conductor agent as one `say` line after one `outcome` line. The
outcome line arms the scripted tool result of the line, so a call the model makes runs no tool.
A line with no tool field arms `call`. A call the model makes then completes as `ok`, and a turn
with no call carries no event. The script waits for the reply and its quality line before the
next line. It ends the run when the quality line of a reply does not come in 30 seconds.

The script writes these files under `<out>`:

| file | content |
| --- | --- |
| `off/`, `on/` | one directory for each boot of the side: the journal, the streams and the run log |
| `pairs.jsonl` | the pairs file; side a is the run with the substrate off, side b the run with it on |
| `tier1.txt` | the mean `coherence_prompt` and the mean `repetition` of each side |
| `events.txt` | the scripted tool event of each line against the affect stream of the run with the substrate on |
| `boots.txt` | the seconds each boot took to its console agent |

The exit status is 0 when both runs are complete and every scripted event matched. It is 1 when
a run or the check failed, and 2 for a usage or environment error. A line with no tool field is
not counted in the check, because the call of the model decides whether an event comes.

## The fixtures

| file | form | size |
| --- | --- | --- |
| `tests/fixtures/quality/conversations.jsonl` | one conversation on each line: a name and its turns; each turn holds `user` and, on a scripted line, `tool` as `ok`, `error` or `refused` | 32 conversations of 4 to 7 turns, 152 user lines |
| `tests/fixtures/quality/rubric.tsv` | one item on each line: the identity, one tab and the question | 8 items |
| `tests/fixtures/quality/tasks.tsv` | one item on each line: the identity, the question, four choices and the answer letter, with a tab between the fields | 200 items, 50 for each letter |

A conversation line has this form:

```text
{"name":"garden-watering-plan","turns":[{"user":"...","tool":"ok"},{"user":"..."}]}
```

`docs/14-affect.md` states the refusal phrase list of the quality stream.

## What the instrument is for

Bind each measurement to its exact model file, input set, build and settings.
Report counts, per-item effects and lost previously correct answers beside aggregate scores.
`docs/08-measured.md` reports measurements from the named earlier versions. A measurement is not a check of the build.
`tests/quality_score.sh` checks the tool on small known inputs, and no check list runs a
measurement.

The historical paired-substrate protocol uses the conditions below with `quality.on=1` on both sides.
It is separate from complete exact-model control qualification:

- the win rate above 0.5, with the low end of the Wilson interval above 0.5, at 32 pairs or
  more;
- the mean `coherence_prompt` of the run with the substrate on not under the other run by more
  than 0.02;
- the mean `repetition` of that run not above the other run by more than 0.02;
- the task set score at the working dose within 0.05 of the score at dose 0;
- the read of the blind transcripts recorded, whatever it shows.

A result that meets the first condition and fails one of the next three is a style gain at a
cost, and not a gain. A result with the win rate at or under 0.5 is no measured gain.

## Interpretation limits

A model scoring its own replies can share their errors or preferences.
Blind transcript review and independent task outcomes remain separate evidence.
A score increase cannot offset a lost required behavior unless the declared acceptance contract permits that tradeoff.

The current control package uses its own fixed intended-effect, capability-cost, control, scope and recovery requirements.
Do not replace those requirements with the historical aggregate thresholds above.
The [support guide](support.md) identifies enabled packages and unresolved accuracy limits.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
