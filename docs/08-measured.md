# Measured

These figures come from one NVIDIA GeForce RTX 3060 card with 12,288 MiB and compute
capability 8.6. The host uses driver 595.84 and CUDA 13.2 with nvcc 13.2.86. Each command ran
alone on the card. The v0.1.0 columns preserve the released measurements of that version.
The v0.2.0 columns come from new Release builds for the 12g and 8g profiles.

## Decode

The decode check reports reply tokens a second over 16 ticks. Each cell gives the rate,
followed by the mean and worst tick time in microseconds. The 12g profile uses the Q8_0
language model. The 8g profile uses the Q4_0 language model.

| live sequences | v0.1.0 12g Q8_0 | v0.1.0 Q4_0 | v0.2.0 12g Q8_0 | v0.2.0 8g Q4_0 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 56.6; 16,500 / 17,700 | 49.6; 18,900 / 20,200 | 54.91; 16,945 / 19,533 | 48.57; 19,168 / 21,410 |
| 8 | 207.2; 36,100 / 38,600 | 227.6; 32,900 / 35,200 | 202.26; 36,888 / 40,886 | 222.57; 33,534 / 38,138 |
| 16 | 203.8; 73,500 / 78,400 | 241.6; 62,000 / 67,400 | 199.21; 75,092 / 81,976 | 237.82; 62,814 / 68,695 |
| 32 | not measured | not measured | not measured | 399.90; 74,740 / 80,998 |
| 64 | 734.3; 78,600 / 84,500 | 753.7; 76,500 / 82,400 | 715.55; 80,540 / 87,923 | not supported; the profile has 32 slots |

Commands:

```text
build/aotx_decode_device_test models tests/fixtures/tokenizer
build-8g/aotx_decode_device_test models tests/fixtures/tokenizer
```

## Host and device paths

The seam rate is the output of a five-second rate case. The worker figures are the mean cost
of one synchronized tick over 300 paced ticks. The attached run sends one key on every tick
and reads the mirror at 30 Hz in a terminal. Both attached runs received 300 of 300 keys and
found no torn frame. The model rate is a warm page-cache placement rate. The key figure is
the socket-to-inbound-ring round trip at one key.

| measure | v0.1.0 | v0.2.0 12g | v0.2.0 8g |
| --- | ---: | ---: | ---: |
| seam, one producer block, records a second | 1.20 million | 1,200,011 | 1,200,013 |
| seam, profile-width producer blocks, records a second | 1.20 million at 64 | 1,200,274 at 64 | 1,200,281 at 32 |
| 16 workers with no terminal, microseconds a tick | not measured | 59.6 | 60.6 |
| 16 workers with a terminal at 30 Hz, microseconds a tick | not measured | 58.4 | 55.1 |
| model placement, MB a second | 3,457 | 3,103 | 3,117 |
| key round trip, microseconds | not measured | 33 | 35 |
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
