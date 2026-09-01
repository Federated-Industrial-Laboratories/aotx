# Measured

This document uses these project terms.

| term | standard name by function |
| --- | --- |
| seam | the host-device memory boundary: pinned host memory mapped for the GPU, crossed only by ring buffers |
| ring | a single-producer, single-consumer ring buffer in pinned host memory |
| tick | one iteration of the device scheduling graph, at a fixed period |
| journal | an append-only log of authoritative records; the recovery source after a process stop |
| mirror | a shared-memory snapshot of the display grid, published for the terminal (a frame copy) |
| profile | a build-time table-size configuration for one class of card |

These figures come from one NVIDIA GeForce RTX 3060 card with 12,288 MiB and compute
capability 8.6. The host uses driver 595.84 and CUDA 13.2 with nvcc 13.2.86. Each command ran
alone on the card. The v0.1.0 columns preserve the released measurements of that version.
The v0.2.0 columns come from new Release builds for the 12g and 8g profiles.

Each v0.2.0 figure is the median of three trials. The brackets give the minimum and maximum.

## Decode

The decode check reports reply tokens a second over 16 ticks. Each cell gives the median rate
and its spread. The values after the semicolon give the median mean and worst tick time in
microseconds and their spreads. The 12g profile uses the Q8_0 language model. The 8g profile
uses the Q4_0 language model.

| live sequences | v0.1.0 12g Q8_0 | v0.1.0 Q4_0 | v0.2.0 12g Q8_0 | v0.2.0 8g Q4_0 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 56.6; 16,500 / 17,700 | 49.6; 18,900 / 20,200 | 55.35 [55.34-55.36]; 16,815 [16,788-16,824] / 18,298 [18,264-18,362] | 48.77 [48.75-48.85]; 19,090 [19,051-19,113] / 20,757 [20,737-20,786] |
| 8 | 207.2; 36,100 / 38,600 | 227.6; 32,900 / 35,200 | 202.96 [202.65-203.24]; 36,751 [36,725-36,844] / 39,387 [39,339-39,828] | 225.10 [224.86-225.20]; 33,162 [33,137-33,178] / 35,593 [35,575-35,673] |
| 16 | 203.8; 73,500 / 78,400 | 241.6; 62,000 / 67,400 | 200.18 [200.05-201.17]; 74,729 [74,338-74,785] / 79,780 [79,420-80,039] | 239.71 [238.23-239.74]; 62,365 [62,357-62,750] / 67,976 [67,871-72,532] |
| 32 | not measured | not measured | not measured | 403.36 [400.30-403.55]; 74,103 [74,054-74,668] / 79,992 [79,409-85,139] |
| 64 | 734.3; 78,600 / 84,500 | 753.7; 76,500 / 82,400 | 719.91 [719.88-720.31]; 80,046 [79,958-80,048] / 86,293 [86,110-86,421] | not supported; the profile has 32 slots |

### Version 0.2.5 interim, 12g Q8_0

These medians come from six tally runs on the 12g profile. This version adds the token
statistics and the page-map measurement to decode. The second column shows the rate of a
measurement build with the page map compiled out; the released build has no switch for it.
The page map alone costs 0.9 to 2.3 percent. The other new controls cause the remaining
difference against the v0.2.0 column.

| live sequences | tokens a second | with the page map off |
| ---: | ---: | ---: |
| 1 | 52.64 | 53.25 |
| 8 | 199.11 | 200.93 |
| 16 | 199.12 | 201.03 |
| 64 | 704.87 | 721.20 |

Commands:

```text
build/aotx_decode_device_test models tests/fixtures/tokenizer
build-8g/aotx_decode_device_test models tests/fixtures/tokenizer
```

## Host and device paths

The seam rate is the output of a five-second rate case. The worker figures are the mean cost
of one synchronized tick over 300 paced ticks. The attached run sends one key on every tick
and reads the mirror at 30 Hz in a terminal. Each attached trial received 300 of 300 keys and
found no torn frame. The model rate is a warm page-cache placement rate. The key figure is
the socket-to-inbound-ring round trip at one key.

| measure | v0.1.0 | v0.2.0 12g | v0.2.0 8g |
| --- | ---: | ---: | ---: |
| seam, one producer block, records a second | 1.20 million | 1,200,009 [1,200,007-1,200,013] | 1,200,022 [1,200,018-1,200,025] |
| seam, profile-width producer blocks, records a second | 1.20 million at 64 | 1,200,265 [1,200,260-1,200,267] at 64 | 1,200,261 [1,200,257-1,200,266] at 32 |
| 16 workers with no terminal, microseconds a tick | not measured | 62.6 [60.5-62.6] | 62.7 [60.3-63.2] |
| 16 workers with a terminal at 30 Hz, microseconds a tick | not measured | 57.7 [55.7-59.4] | 57.3 [44.6-60.5] |
| model placement, MB a second | 3,457 | 3,205 [3,110-3,230] | 3,274 [3,234-3,287] |
| key round trip, microseconds | not measured | 22 [18-39] | 22 [21-29] |
| window on the display | 126 reply tokens a second |  |  |

Commands:

```text
build/aotx_seam_device_test --seconds 5
build-8g/aotx_seam_device_test --seconds 5
script -q -e -c 'build/aotx_mirror_test' mirror-12g.log
script -q -e -c 'build-8g/aotx_mirror_test' mirror-8g.log
build/aotx_boot --journal build/measure-12g --models models --roles embedding,reranker,language --ticks 1 --solo
build-8g/aotx_boot --journal build-8g/measure-8g --models models --roles embedding,reranker,language-q4 --ticks 1 --solo
build/aotx_attach_test
build-8g/aotx_attach_test
```

The display rows stay empty until a display run supplies them. This is the exact 12g command
for that run:

```text
cmake -S . -B build-window -G Ninja -DCMAKE_BUILD_TYPE=Release -DAOTX_ARCH=86 -DAOTX_DISPLAY_TESTS=ON && cmake --build build-window -j2 && ctest --test-dir build-window -R '^window$' --output-on-failure
```

The display check writes `build-window/window.ppm`. It opens a window and must run only in a
display session. The 8g form adds `-DAOTX_PROFILE=8g` to the configure command.
