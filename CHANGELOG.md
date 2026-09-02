# Changelog

Each entry names the commits that supply its change. The entries come from the commit record
after the preceding version tag.

## 0.2.5

### Release

- Set version 0.2.5 (the release commit).

### Decode controls and telemetry

- Add agent decode controls (`f8f32ef`). Class A records set the sampler row of each agent. The token statistics stream, the think budget and the stop command for one agent come with them.
- Add conduct controls and measurement tools (`b4fd0c4`). The steer vector derivation gives a potency figure. The steer add follows each layer residual. The bias profiles and the page map stream come with them.
- Publish the device ring use (`b0fc9f8`) and show the ring occupancy and record the rates (`a59f934`).
- Split the matrix rate fixture (`b570fc7`).

### Control client

- Make first-run startup and voice reliable (`a6fa503`).
- Improve conversation and sync controls (`1feb216`).
- Add model and persona controls (`28e4737`).
- Repair the control program and document it (`8904192`).
- Wrap the reply text at the window edge (`9b8db35`).

### Checks and repairs

- Fix the findings of the review of the quality pass (`5fa9c7c`).
- Harden the steer tools, the conduct loader and the checks (`7097b56`).
- Keep one manifest line for each role (`685c7bb`).
- Name an empty settings file in the replay gate (`96ca1dc`).

### Repository page

- Add the divider graphic to the README (`673d4e9`) and update it (`056546a`).

## 0.2.0

### Release

- Set version 0.2.0 and add the release documents and measured figures (`6086667`).
- Add the repository page (`8306cd8`).
- Add the terms table to the repository page (`608de31`).
- Replace the mark on the repository page (`30b3bb4`).
- Show the banner at full width on the repository page (`5495311`).
- Fix the release findings and update the verified figures (``4b56f51``).

### Profiles and settings

- Add the settings records, the card record and the common key list (`1c97e00`).
- Add the four build profiles and move device table limits into them (`e382048`).
- Add settings files and their disk-side readers (`82b9117`).
- Write a changed device setting at the tick commit (`15515be`).

### Modules and tools

- Add the import and remove record layouts (`d63e39b`).
- Add the device catalog and the import of modules (`9c3a356`, `47fbecb`, `4c7bc3e`).
- Add device and host tool modules, checks, examples and the tool SDK (`5afe360`, `abd1722`, `fa07d81`).
- Join tool nodes to the tick graph and make the tool contracts exact (`c87564f`, `7ec445a`).

### Terminal and conversation

- Add the display mirror and the terminal program (`bf61047`, `ddd5ee7`, `f2016e9`).
- Add long input lines, transcripts and conversation memory (`cd47376`, `4e4b645`, `316a637`).

### Model store

- Add the model record, the store program, run-time model load and store checks (`30ec4ca`, `c905dbd`).
- Make replacement, fetch and catalog rules exact (`d0a5b88`).

## 0.1.0

- 088c239 Add the repository layout, the gates and the clock module
- 4bba818 Add the seam
- 973c096 Add the surface
- 80a57ba Add the model file and the tokenizer
- 582948d Add the model kernels
- 5d730d1 Add decode through the tick
- 2a26563 Add agents and tools
- 7e70542 Add the documentation set
- b7d1257 Fold the block scale once in the PTX product
- a52bf6f Record the late verdict and pace the replay
- 9ecab9a State the late verdict and the pace of a replay
- 6297929 Hold the replay clock until a later tick is seen
- 50df09b Refuse a second granted line after a restore
- ff6ced1 Add the measured figures of one machine
