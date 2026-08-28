# The bus schema

The bus is the message layer over records. An agent puts a message on the bus with one append,
and the append writes one bus record of class B. The drain turns that record into one line of
`bus/<date>-aotx.jsonl`. This document gives the message on the device and the line on the disk.
`04-journal-format.md` gives the byte layout of the record.

## The seven kinds

| kind | number | what the message carries |
| --- | --- | --- |
| finding | 1 | a claim and a provenance value |
| rank | 2 | a reference to a message, a score from 0 to 1, and a basis |
| question | 3 | a text |
| answer | 4 | a reference to a message, and a text |
| handoff | 5 | a path, and a status of draft, ready or blocked |
| cost | 6 | what was consumed, and what was produced |
| note | 7 | a text |

The record body carries one text of 160 bytes at most. A kind with two text fields takes the
part before the first line feed and the part after it. A handoff takes its status from the
second part, and a second part that names no status gives the status draft. A cost with no
second part gives the produced field the value `not stated`.

## The envelope

Each line carries the same envelope: the version `v`, the run, the agent, the sequence `seq`,
the time `ts`, the type, and the body. The version is 1 and the run is `aotx`. The type is the
name of the kind. The body holds the fields of that kind.

The drain adds three fields to the end of every body. The field `tick` is the tick of the
record. The field `boot` is the boot identity in 16 hexadecimal digits. The field `lag_ms` is
the time from the newest tick-start record to the write, in milliseconds.

## From the device form to the line

The device has no wall clock. A record carries the tick, the record sequence and a device clock
sample, and those three give the time of a message on the device. The drain makes the `ts` field
from its own clock when it writes the line. The form is ISO 8601 with milliseconds and the
offset of the local time zone. The lag beside it is the distance between the two clocks.

The feeder publishes a tick-start record ten times a second, and each one carries the wall clock
of the feeder. The drain holds the newest of those values. The field `lag_ms` is that value
taken from the write time. It is `null` until the first tick-start record arrives.

## The writer identity

The append stamps the writer from its argument. It never takes the writer from the body of the
message. The writer identities below 1024 are system writers: 0 system, 1 feeder, 2 restore, 3
console. The value 1024 is the first agent. The drain gives the name `agent-N` to the identity
1024 plus N, for N below 256.

The append refuses a writer identity of 1088 or above, because the sequence table holds no entry
for it. It refuses an identity from 4 to 1023, because the disk side gives that range no name.
A refusal writes no record and raises the refusal count.

Provenance names where the content of a finding comes from: 1 computed, 2 fetched, 3 recalled, 4
testimony. The append refuses a finding whose provenance is outside that range. It refuses any
other kind that carries a provenance value.

## Sequence numbers

Each writer has its own count. The append takes the next value of that writer from a table of
1088 entries. The first message of a writer therefore carries the writer sequence 1. Two
writers never give one message number to two messages.

One line file spans the boots of a day, so the drain must not give a writer the same number
twice. It reads the file back when it opens it, and it holds the next free number of each
writer. The envelope `seq` is the writer sequence of the record when the file does not already
hold that number, and the next free number otherwise.

## Corrections

A correction is an append that names the record it replaces. There is no separate file and no
change to the line that the correction replaces. The drain holds a map of the last 65,536 record
sequences, so it can give the message identity of a reference.

When the map holds the record, the envelope takes `"req":["msg-relations"]`. The body takes
`"corrects"` with that message identity, and `"reason"` with the text of the correction. When
the map does not hold it, the body takes an `"unresolved"` field that names the record sequence.

A rank names a finding or a handoff. A rank whose reference the map does not hold takes the
type note, and states the score and the gap. A rank that names another kind takes the type note
in the same way. An answer whose reference the map does not hold also takes the type note.

## The derived events

Three other record types make message lines, under the same `bus` name in `--derive`. A task in
the state done makes a handoff with the path `task <id>`, the status ready, and the result as
the note. Every other task state makes a note from the writer of the record. Every agent event
makes a note from the writer of the record.

A console record and a note record make a note line as well, under the names `console` and
`note`. The text of the record is the text of the line, and the writer of the record names the
agent. A note of an agent therefore does not read as a note of the console.

The drain makes one more note from the writer `system` for each sequence record whose event is
done or stopped. A sequence event belongs to the run and not to one agent. An opened event and a
released event make no line, because neither ends a reply. A token record makes no line, and the
text of a reply reaches the console log from the console records.

## The refusals

The drain writes no line, and raises the refused count, in six cases.

- The writer identity has no name.
- The writer sequence of the body is zero.
- The body is shorter than the 32 fixed bytes.
- The text is empty, or it holds white space only.
- A finding carries a provenance outside 1 to 4.
- A rank carries a score outside 0 to 1.

## One line for each kind

```
{"v":1,"run":"aotx","agent":"agent-0","seq":1,"ts":"2026-08-28T14:41:50.626+01:00","type":"finding","body":{"id":"agent-0-1","claim":"claim 0 of the run","provenance":"computed","tick":7,"boot":"38ccfbd6eccc7df1","lag_ms":12.500}}
{"v":1,"run":"aotx","agent":"agent-0","seq":2,"ts":"2026-08-28T14:41:50.630+01:00","type":"rank","body":{"re":"agent-0-1","score":0.750000,"basis":"basis 0 of the run","tick":8,"boot":"38ccfbd6eccc7df1","lag_ms":16.000}}
{"v":1,"run":"aotx","agent":"agent-1","seq":1,"ts":"2026-08-28T14:41:50.634+01:00","type":"question","body":{"text":"question 0 of the run","tick":9,"boot":"38ccfbd6eccc7df1","lag_ms":20.000}}
{"v":1,"run":"aotx","agent":"agent-1","seq":2,"ts":"2026-08-28T14:41:50.638+01:00","type":"answer","body":{"re":"agent-0-1","text":"answer 0 of the run","tick":10,"boot":"38ccfbd6eccc7df1","lag_ms":24.000}}
{"v":1,"run":"aotx","agent":"agent-0","seq":3,"ts":"2026-08-28T14:41:50.642+01:00","type":"handoff","body":{"path":"path/of/0","status":"ready","tick":11,"boot":"38ccfbd6eccc7df1","lag_ms":28.000}}
{"v":1,"run":"aotx","agent":"agent-0","seq":4,"ts":"2026-08-28T14:41:50.646+01:00","type":"cost","body":{"consumed":"consumed 0","produced":"produced 0","tick":12,"boot":"38ccfbd6eccc7df1","lag_ms":32.000}}
{"v":1,"run":"aotx","agent":"system","seq":1,"ts":"2026-08-28T14:41:15.875+01:00","type":"note","body":{"text":"sequence done slot 0 role 2 prompt 353 sampled 256 ticks 195","tick":196,"boot":"0772fbccf1e3666e","lag_ms":72.560}}
```

A correction of the first line above reads as follows.

```
{"v":1,"run":"aotx","agent":"agent-0","seq":5,"ts":"2026-08-28T14:41:50.650+01:00","req":["msg-relations"],"type":"finding","body":{"id":"agent-0-5","claim":"the claim of the run","provenance":"computed","corrects":["agent-0-1"],"reason":"the claim of the run","tick":13,"boot":"38ccfbd6eccc7df1","lag_ms":36.000}}
```

A task event and an agent event read as follows. The handoff of a done task carries a note
beside the path and the status.

```
{"v":1,"run":"aotx","agent":"agent-1","seq":8,"ts":"2026-08-28T14:46:54.611+01:00","type":"handoff","body":{"path":"task 0","status":"ready","note":"the first line of the file is \"the first line of the file\".","tick":52,"boot":"3be0fc1d0ac3af61","lag_ms":37.242}}
{"v":1,"run":"aotx","agent":"agent-1","seq":1,"ts":"2026-08-28T14:41:50.638+01:00","type":"note","body":{"text":"agent 1 spawned role 1 parent 0 state 1 turn 0 ticks 0","tick":2,"boot":"38ccfbd6eccc7df1","lag_ms":null}}
```
