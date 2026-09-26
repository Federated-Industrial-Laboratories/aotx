<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Message bus schema

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

The bus carries typed messages between device agents.
A device append creates a class B BUS record; the drain renders it as a JSON line in `bus/<date>-aotx.jsonl`.
The [journal format](04-journal-format.md) defines the enclosing record bytes.

<details>
<summary>On this page</summary>

- [The seven kinds](#the-seven-kinds)
- [The envelope](#the-envelope)
- [The writer identity](#the-writer-identity)
- [Sequence numbers](#sequence-numbers)
- [Corrections and references](#corrections-and-references)
- [The derived events](#the-derived-events)
- [Refused output](#refused-output)
- [One line for each kind](#one-line-for-each-kind)

</details>

## The seven kinds

| Kind | ID | Contents |
| --- | ---: | --- |
| `finding` | 1 | A claim and its provenance. |
| `rank` | 2 | A message reference, score from 0 to 1 and basis. |
| `question` | 3 | Question text. |
| `answer` | 4 | A message reference and answer text. |
| `handoff` | 5 | A path and status: `draft`, `ready` or `blocked`. |
| `cost` | 6 | Consumed and produced values. |
| `note` | 7 | Note text. |

The body holds at most 160 text bytes.
Kinds with two text fields split at the first line feed.
A handoff with an unknown second field uses `draft`.
A cost without a second field uses `not stated` for its produced value.

## The envelope

| Field | Meaning |
| --- | --- |
| `v` | Schema version, 1. |
| `run` | Runtime family, `aotx`. |
| `agent` | Named system writer or `agent-N`. |
| `seq` | Per-writer sequence in the derived line file. |
| `ts` | Disk write time in ISO 8601 with milliseconds and local offset. |
| `type` | Message kind name. |
| `body` | Fields defined by that kind, plus runtime observations. |

Every body also contains `tick`, hexadecimal `boot` and `lag_ms`.
The lag compares the latest feeder wall-clock sample with the disk write time.
It is null until the drain receives a sample.
It is not a direct subtraction of device timer and host wall-clock values.

## The writer identity

`aotx_bus_append` stamps the caller's writer identity; message text cannot supply it.
System IDs are 0 for system, 1 for feeder, 2 for restore and 3 for console.
Agent slot N uses writer ID `1024 + N`.

The device writer table has `1024 + AOTX_SLOTS` entries.
It refuses IDs outside that configured range and unused system IDs from 4 through 1023.
The derived disk name supports agent slots below 256.
This is a profile-dependent bound, not a fixed 64-agent limit.

A finding requires provenance `computed`, `fetched`, `recalled` or `testimony`, encoded as 1 through 4.
Other message kinds require zero provenance.
Invalid writer or provenance values raise refusal counters and produce no accepted message.

## Sequence numbers

Each writer starts its device message count at one.
The derived daily file can span multiple boots, so the drain reads existing counters before appending.
It preserves a record's writer sequence when unused, or assigns the next free sequence otherwise.
Message identity therefore remains unique within that file's writer namespace.

## Corrections and references

A correction appends a new message that names the earlier record.
It never edits the earlier line.
The drain keeps a map of the latest 65536 record sequences for relation lookup.

A resolved correction adds `req: ["msg-relations"]`, the corrected message identity and a reason.
An unavailable target produces an `unresolved` record reference.
An unresolved rank or answer becomes a note that states the missing relation.
A rank can target a finding or handoff; another target kind also produces a note.

## The derived events

Task, agent and sequence events can produce bus lines even when they are not BUS records.
A completed task produces a ready handoff with its result note.
Other task states and agent events produce notes.
A completed or stopped sequence produces a system note.

Console and NOTE records use the `console` and `note` derive selections.
The `bus` selection includes messages and applicable task, agent and sequence events.
Individual token records do not produce bus lines.

## Refused output

The disk renderer refuses a line when its writer is unknown, writer sequence is zero or fixed body is shorter than 32 bytes.
It also refuses empty text, whitespace-only text, invalid finding provenance and a rank score outside 0 through 1.
See `aotx_derive_message` in `disk/drain/derive_bus.c` for these checks.

## One line for each kind

```text
{"v":1,"run":"aotx","agent":"agent-0","seq":1,"ts":"2000-01-01T00:00:00.000+00:00","type":"finding","body":{"id":"agent-0-1","claim":"claim 0 of the run","provenance":"computed","tick":7,"boot":"0000000000000000","lag_ms":12.500}}
{"v":1,"run":"aotx","agent":"agent-0","seq":2,"ts":"2000-01-01T00:00:00.001+00:00","type":"rank","body":{"re":"agent-0-1","score":0.750000,"basis":"basis 0 of the run","tick":8,"boot":"0000000000000000","lag_ms":16.000}}
{"v":1,"run":"aotx","agent":"agent-1","seq":1,"ts":"2000-01-01T00:00:00.002+00:00","type":"question","body":{"text":"question 0 of the run","tick":9,"boot":"0000000000000000","lag_ms":20.000}}
{"v":1,"run":"aotx","agent":"agent-1","seq":2,"ts":"2000-01-01T00:00:00.003+00:00","type":"answer","body":{"re":"agent-0-1","text":"answer 0 of the run","tick":10,"boot":"0000000000000000","lag_ms":24.000}}
{"v":1,"run":"aotx","agent":"agent-0","seq":3,"ts":"2000-01-01T00:00:00.004+00:00","type":"handoff","body":{"path":"path/of/0","status":"ready","tick":11,"boot":"0000000000000000","lag_ms":28.000}}
{"v":1,"run":"aotx","agent":"agent-0","seq":4,"ts":"2000-01-01T00:00:00.005+00:00","type":"cost","body":{"consumed":"consumed 0","produced":"produced 0","tick":12,"boot":"0000000000000000","lag_ms":32.000}}
{"v":1,"run":"aotx","agent":"system","seq":1,"ts":"2000-01-01T00:00:00.006+00:00","type":"note","body":{"text":"sequence done slot 0 role 2 prompt 353 sampled 256 ticks 195","tick":196,"boot":"0000000000000000","lag_ms":72.560}}
```

A correction of the first line above reads as follows.

```text
{"v":1,"run":"aotx","agent":"agent-0","seq":5,"ts":"2000-01-01T00:00:00.007+00:00","req":["msg-relations"],"type":"finding","body":{"id":"agent-0-5","claim":"the claim of the run","provenance":"computed","corrects":["agent-0-1"],"reason":"the claim of the run","tick":13,"boot":"0000000000000000","lag_ms":36.000}}
```

A task event and an agent event read as follows. The handoff of a done task carries a note
beside the path and the status.

```text
{"v":1,"run":"aotx","agent":"agent-1","seq":8,"ts":"2000-01-01T00:00:00.008+00:00","type":"handoff","body":{"path":"task 0","status":"ready","note":"the first line of the file is \"the first line of the file\".","tick":52,"boot":"0000000000000000","lag_ms":37.242}}
{"v":1,"run":"aotx","agent":"agent-1","seq":1,"ts":"2000-01-01T00:00:00.003+00:00","type":"note","body":{"text":"agent 1 spawned role 1 parent 0 state 1 turn 0 ticks 0","tick":2,"boot":"0000000000000000","lag_ms":null}}
```

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
