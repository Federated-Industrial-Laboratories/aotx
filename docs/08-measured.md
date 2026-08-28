# Measured

The figures below come from one machine at one commit. They are the output of the checks and
programs of this repository, run one at a time with no other process on the card. A figure
from another machine, another driver or another commit is a different figure.

## The machine

| item | value |
| --- | --- |
| card | NVIDIA GeForce RTX 3060, 12,288 MiB, compute capability 8.6 |
| driver and toolkit | 595.84, CUDA 13.2 (nvcc 13.2.86) |
| host | Xeon E5-2680 v4, 62 GiB, PCIe 3 x16 |
| disk | SATA SSD, about 500 MB a second |
| commit | 50df09b, 2026-08-28 |
| models | Qwen3-4B Q8_0 (`language`) and Q4_0 (`language-q4`), Qwen3-Embedding-0.6B Q8_0, Qwen3-Reranker-0.6B Q8_0 |

## The matrix kernels

From `aotx_matrix_device_test models`. The peak of the card is 51.0 TFLOPS for half products
with single precision sums, and its memory rate is 360 GB a second.

| kernel | shape | rate |
| --- | --- | --- |
| GEMM Q8_0, tensor cores | m 256, n 2560, k 2560 | 17.7 TFLOPS, 34.7 percent of peak, 190 microseconds a launch |
| GEMM Q8_0, tensor cores | m 256, n 9728, k 2560 | 18.4 TFLOPS, 36.0 percent of peak, 694 microseconds a launch |
| GEMV Q8_0, compiled | m 1, n 2560, k 2560 | 218 GB a second, 60.7 percent of the memory rate |
| GEMV Q8_0, compiled | m 1, n 9728, k 2560 | 239 GB a second, 66.2 percent |
| GEMV Q8_0, compiled | m 8, n 9728, k 2560 | 125 GB a second, 34.8 percent |
| GEMV Q8_0, PTX module | m 1, n 2560, k 2560 | 292 GB a second, 24 microseconds a launch, 81 percent |

The module and the compiled kernel give the same sums against the reference: 0 of 2,560 over
1e-02, worst 7.93e-04.

## The model

From `aotx_model_gate_device_test`. The `language` model at Q8_0 agrees with the single
precision reference at 493 of 505 argmax positions, mean distance 0.0247. The Q4_0 file agrees
at 502 of 505 against its own reference. The embedding model gives a cosine of 0.999 or better
on 69 of 70 lines, least 0.9989. The reranker gives 0.9988 for the first pair and 0.000002 for
the last.

## Decode

From `aotx_decode_device_test models tests/fixtures/tokenizer`. The rate is reply tokens a
second over 16 ticks with every sequence in the batch. The tick is the tick of the decode graph
in milliseconds.

| sequences | Q8_0 tokens a second | Q8_0 tick mean, worst | Q4_0 tokens a second | Q4_0 tick mean, worst |
| --- | --- | --- | --- | --- |
| 1 | 56.6 | 16.5, 17.7 | 49.6 | 18.9, 20.2 |
| 8 | 207.2 | 36.1, 38.6 | 227.6 | 32.9, 35.2 |
| 16 | 203.8 | 73.5, 78.4 | 241.6 | 62.0, 67.4 |
| 64 | 734.3 | 78.6, 84.5 | 753.7 | 76.5, 82.4 |

A prompt of 858 tokens goes in over 3 ticks at 512 prompt tokens a tick, while 8 sequences
join the same batch.

## The tick and the seam

From `aotx_sched_device_test` and `aotx_seam_device_test`. The tick graph with no decode takes
354 microseconds mean over 500 ticks. Its 99th percentile is 382 microseconds and its worst tick
392. The whole graph with its launch takes 374, 403 and 410 microseconds. The record ring takes
1.20 million records a second with one producer block and with 64 producer blocks (6.0 million
records in 5.0 s).

## Agents

From `aotx_agent_device_test models`. Sixteen tasks on the `language` model end in 65 ticks and
12.3 s, 32 turns, each task with one `memory_write` call. One task ends in 49 ticks and 1.7 s.

Sixteen workers take sixteen story tasks at once. The run is `aotx_boot --models models
--roles language` with the lines on its standard input. Each worker reads a prompt of 363
tokens and writes a reply of 256 tokens. Every reply ends between tick 256 and tick 267 of its
sequence. The sixteen replies end 29.8 s after the first task line. That is 137 reply tokens
a second over the whole span, with the prompts and the agent steps inside it.

The drain is 135 ms behind the device at the median, 181 ms at the 90th percentile and 183 ms
at most.

The same sixteen tasks ran again with the window on the display (`--window`), the raster at
60 frames a second. The replies end 32.4 s after the first task line, 126 reply tokens a
second. Every reply takes the same 256 to 267 ticks. The window costs the tick about 8 percent
over the span. It costs the frame nothing: 17,571 frames over 26,306 ticks, mean 16.67 ms,
worst 19.77 ms, 0 over 20 ms, 0 keys dropped.

## Boot and restore

From `aotx_boot`. The three default model files, 5,286 MB, are placed in 1.53 s from a warm
page cache, 3,457 MB a second. A cold read from the disk was not measured. A journal of
10,000 ticks with no model and no agent is restored in 1.42 s. The restore applies 999 records
over 9,997 ticks, and 8,982 of those ticks have no record to take.

## The disk behind the device

From `aotx_boot --workload 12000 --blocks 64 --ticks 3000`, the disk settled before each run.
The tick load writes 12,000 records a tick. The `held` column is the ticks the scheduler held
because the host ring had no room for the worst case of the tick.

| derive list | records | blocks drained | ticks held of 3,000 | on disk |
| --- | --- | --- | --- | --- |
| default, a line for each record | 1.85 million | 176 | 2,846 | 1.3 GB |
| `console,bus` | 33.6 million | 3,071 | 197 | 8.1 GB |
| `none` | 12.7 million | 1,158 | 1,945 | 3.1 GB |

With the default list the drain is 1.3 s behind at the median and 6.6 s at the 99th
percentile. The `none` list holds more ticks than `console,bus` in two passes; the cause is
not measured.
