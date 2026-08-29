# The tool SDK

A tool is a module an operator installs. This document is for the author of a tool. It
gives the contract of a device tool and the contract of a host tool. It also gives the check
program, the build script and the two examples that stand in the repository.

Read `docs/09-modules.md` first. It gives the module directory, the manifest keys and the
import commands. This document gives what the two kinds of tool must do.

## The two kinds

| kind | `side` | runs | holds |
| --- | --- | --- | --- |
| device tool | `device` | as one node of the tick graph | a module file the driver loads |
| host tool | `host` | as a program the feeder starts | an executable file |

A device tool computes over its arguments and its scratch. It reads no file and it opens no
connection. A host tool is a program in any language. It reaches what the operator lets it
reach.

## The device contract

### The header

A device tool includes one header and nothing else of the system: `sdk/aotx_tool.h`. The
header is plain C. It holds the layouts and the constants of the contract, and no CUDA
symbol. The constant `AOTX_TOOL_ABI` is the version of the contract. A change to any
structure raises it.

### The entry

The module holds one kernel with two parameters:

```
extern "C" __global__ void aotx_tool_<name>(aotx_tool_batch *batch, aotx_tool_output *out)
```

The name comes from the manifest key `entry`. A manifest that gives no `entry` takes
`aotx_tool_` and the name of the module. The name must not change under a C++ compiler, so
the kernel carries `extern "C"`.

The system launches the node with one block for each request row of the profile and 256
threads in a block (`AOTX_TOOL_MODULE_THREADS`, `cuda/tool/module.cuh`). The row of a block
is `blockIdx.x`. A kernel that takes no row exits at once, which costs about 1.4
microseconds.

### The batch rule

`batch->rows` is the request rows of the build. Row `i` of `batch->row` carries:

| field | meaning |
| --- | --- |
| `take` | 1 when the row is a request for this tool in this tick, else 0 |
| `request` | the number of the request |
| `agent` | the agent that made the call |
| `arguments` | the argument keys that carry a value |
| `seed` | a lane seed of the row and the tick |
| `argument[k]` | the key, the value and the length of argument `k` |

**The rule: a module reads the rows whose `take` is 1.** It writes the output rows of
those rows and no other row. A row of another tool stands beside it in the same tables. A
module that writes an untaken row writes the answer of another tool. The check program
refuses such a module.

The keys stand in the order the manifest key `arguments` gives. Key `k` of the manifest is
`argument[k]` of the row. A key that the call did not carry has a length of zero. A key
holds 32 bytes at the most (`AOTX_TOOL_KEY_BYTES`) and a value holds 1,024 bytes at the most
(`AOTX_TOOL_VALUE_BYTES`). A tool takes 4 keys at the most (`AOTX_TOOL_ARGS_MAX`).

### The scratch

`batch->scratch` is `batch->rows` times `batch->scratch_bytes` bytes of device memory. Row
`i` owns the bytes from `i` times `batch->scratch_bytes` for the tick. The bytes hold no
value from the tick before. A module that needs more than the scratch is a module the
profile cannot hold.

### The done store

The output of row `i` is `out->head[i]`. Its text starts at `out->text` plus `i` times
`batch->out_bytes`.

1. Write the text.
2. Write `length`, which must be `batch->out_bytes` or less.
3. Write `status`: `AOTX_TOOL_STATUS_OK`, or `AOTX_TOOL_STATUS_ERROR` to make the text a
   reason.
4. Put a fence between the text and the last store.
5. Store `done` as 1, last.

The tool step of the tick reads `done`. A row that holds no `done` waits for the tick after
it, and for the deadline of the request after that. The deadline is the manifest key
`deadline`, or the setting `tool.deadline_ticks`.

### What a device tool does not see

The layout gives the arguments, the scratch and the output. It gives no ring, no agent
table, no model and no row of another tool. A module reaches nothing else. A later version
of the contract adds a field; it never gives a module an address of the state of the system.

### The architecture

The module file holds the text of a module. A module built for `sm_86` loads on a newer card
through the forward compatibility of the driver. A module whose target is above the card is
refused at the load with the reason. The check program states the target it found.

## The host contract

The feeder starts the program with the module directory as its working directory. The
program takes:

| what | where |
| --- | --- |
| the call | one requests line, as JSON, on the standard input |
| the answer | the standard output, under the cap of 4,096 bytes |
| the reason | the first 256 bytes of the standard error |
| the verdict | the exit status; zero is an answer, and every other value is a reason |

The requests line carries the fields `agent`, `request`, `tool` and `arg`. The `arg`
field holds the arguments of the call as `key=value` pairs. The keys stand in the order the
manifest gives them, and the unit separator byte parts two pairs. That byte is 31. The drain
writes it as a JSON escape, so the line stays one JSON line.

The feeder ends a program that runs past the seconds of the manifest key `timeout`, and it
answers with that reason. The default is 30 seconds. The device deadline stands beside the
timeout, and it is a count of ticks.

## The trust boundary

The install is the trust act. A tool program runs with the rights of the operator, and the
feeder puts no sandbox around it. Two rules stand between a model and a program:

- The manifest key `authorise` with the value `always` puts the operator before every call.
- The role key `authorise` adds the same wait for a tool that says `never`. A role cannot
  take the wait off a tool that says `always`.

The commands `grant` and `refuse` answer a request that waits (`docs/07-operation.md`).

The built-in tools of the file group stand in the catalog, and a role names the ones it may
call. The three that write carry `authorise: always`, so a call by a role that names them
waits for the operator. A role that no operator watches must therefore not name them. This
is a role manifest that names them:

```
kind: role
name: editor
version: 1
description: Reads and writes files below the allowed root.
model: language
tools: memory_recall,memory_write,fs_read,fs_list,fs_write,fs_update,skill_use
authorise: fs_read
budget: 0
skills:
body: overlay.txt
```

The roles the repository carries name the tools that need no operator. A run with nobody at
the console therefore does not stop on a request that waits.

## The check program

`aotx_module_check <directory> [rows]` proves a module before an operator installs it. With
no row count it runs the module at 1 row and at the row count of the profile. It prints one
line for each check with the figure, and it gives the exit status 1 when a check fails.

| line | what it means |
| --- | --- |
| the manifest of the module | the reader of the device took the manifest; a refusal gives the reason |
| bytes of the example line | the manifest holds the key `example` |
| modules the driver holds | the module file opened, its digest agreed and the driver loaded it |
| the program starts under the timeout | a host tool: the feeder started the program |
| the exit status of the program | a host tool: the status the program gave |
| the program wrote bytes | a host tool: the bytes of the standard output |
| the answer is under the cap | a host tool: the bytes stand under 4,096 |
| bytes of local memory | the kernel holds no local memory; the spill rule refuses any |
| registers the kernel keeps | the register count of the kernel |
| threads of a block the kernel takes | the kernel takes the 256 threads of a node |
| the architecture of the module | the architecture the driver made the module for |
| the version of the module text | the version of the module text, times ten |
| at N rows the module answered rows | every taken row holds `done` |
| at N rows a status the contract refuses | every row holds `ok` or `error` |
| at N rows a length over the bound | no length is above `out_bytes` |
| at N rows an untaken row that was written | no untaken row changed |
| at N rows the longest result | the longest result of the run |
| at N rows the launch of 10,000 microseconds took | the launch stands inside the tick budget |

The check fills every row with the value of the manifest key `example`. It then makes the
content of each row distinct. The row number stands in the value of the second key, when the
tool has a second key. It stands at the end of the value of the first key, when the tool has
one key alone. The check writes a pattern in every untaken row, and it reads that pattern
back.

For a host tool the check starts the program over the example line, under the timeout. It
uses the starter of the feeder, so the program gets the working directory, the line and the
environment a run gives it. The check then states the exit status, the bytes the program
wrote, and whether those bytes stand under the cap.

The check goes no further than the framing. The poll of the feeder publishes the reply into
the inbound ring, and the check program holds no ring. The check of the file tools on the
disk side exercises that path.

## The build script

`tools/module-build.sh <directory> [--arch <number>]` builds the module file of a device
tool.

1. It reads the architecture from `aotx_boot --version` beside the script, or from
   `--arch`.
2. It runs `tools/seam-gate.py` over every `.cu` file of the directory. A device tool is a
   device file, so a host call in it refuses the build.
3. It compiles with `nvcc -ptx -arch=sm_<number> -Isdk` to `<directory>/<name>.ptx`.
4. It writes the digest of that file into the manifest key `sha256`.

The feeder computes the digest again at the import, and the head of the import carries
it. The host glue reads the module file at the import and again after a restore. It refuses
a file whose digest is not the digest of the import (`cuda/tool/module_host.cu`).

The loader keeps a module whose digest and kernel name it holds already, so a capture
inside a run pays no load. A run therefore keeps the code the import named. A file that
changes on disk under a run does not change what the device runs. The next run reads every
module file again, and it refuses a file that changed.

The head of an import carries 63 bytes of the path of the module directory. A longer path
does not open. The loader then looks for the module below the directory of the module
directories of the run, by the name of the module. The boot names that directory with
`--modules`.

## The examples

`sdk/examples/word_count/` is a device tool. It counts the words of its `text` argument and
gives the figure. The kernel counts the word starts of the row with the whole block and adds
them in shared memory, so it holds no local memory.

`sdk/examples/echo_upper/` is a host tool. The program is a shell script. It takes the
requests line on its standard input, reads the value of the `text` argument, and writes that
value in capital letters.

The built-in tools are two more examples. `cuda/tool/device_tools.cu` holds the memory tools
and `skill_use`, which run inside the tool step. `disk/feed/fs_tool.c` holds the file tools,
which run on the disk side.

## Where each part stands

| part | path |
| --- | --- |
| the layout header | `sdk/aotx_tool.h` |
| the batch of the tick | `cuda/tool/module.cuh`, `aotx_tool_modules` |
| the fill of the rows | `cuda/tool/module.cu`, `aotx_tool_module_fill` |
| the load and the digest | `cuda/tool/module_host.cu`, `aotx_tool_module_open` |
| the node of the graph | `cuda/tool/node_host.cu`, `aotx_tool_module_capture` |
| the answer in the tool step | `cuda/tool/module.cu`, `aotx_tool_module_reap` |
| the check program | `cuda/catalog/check_host.cu`, `cuda/catalog/check.cu` |
| the build script | `tools/module-build.sh` |
