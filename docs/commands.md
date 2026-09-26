<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Console commands

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

The GPU console controls local agents, models and tools.
Use `help` to inspect its main command list. Type commands in the window, terminal or an attached input stream.
These commands do not invoke a shell.

<details>
<summary>On this page</summary>

- [Command reference](#command-reference)
- [Conversation controls](#conversation-controls)
- [Tool selection and authorization](#tool-selection-and-authorization)
- [Model and module changes](#model-and-module-changes)
- [Memory, media and policy commands](#memory-media-and-policy-commands)
- [Input and output bounds](#input-and-output-bounds)

</details>

## Command reference

| Command | Effect |
| --- | --- |
| `help` | show the command lines |
| `bus [kind]` | show the last bus messages of a kind |
| `note <text>` | put a note on the bus |
| `finding <source> <text>` | put a finding on the bus |
| `say <text>` | send a message to the conductor agent |
| `stop` | end the reply that runs |
| `continue` | resume a reply that ended at its reply limit |
| `outcome <ok\|error\|refused\|none\|call>` | arm one tool result for the next turn of the conductor agent; that turn runs no tool |
| `spawn <role> [n]` | make n agents of a role; n is 1 to 8 |
| `task <agent\|role> <text> [verify]` | open a task for an agent or for a role |
| `authorize <id>` | let a tool request of that number run |
| `refuse <id>` | stop a tool request of that number |
| `mem` | show the memory regions and the budget |
| `memory` | show the page pool and the limit of each live agent |
| `agents` | show the agents |
| `agent <id>` | show the transcript counts and the summary sequence of one agent |
| `agent <id> pages <n\|auto>` | change the hot memory bound at the next turn |
| `agent <id> compact` | start a compaction turn when the agent is idle |
| `agent <id> stop` | stop the reply of one agent at its next token |
| `agent <id> continue` | resume the reply of one agent that ended at its reply limit |
| `agent <id> decode.<key> <value>` | change one sampling value at the next turn |
| `stats` | show the counts of the last tick |
| `settings` | show the settings and when each takes effect |
| `set <key> <value>` | change a setting; the change is a class A record |
| `model load <role> <name>` | place a model file between two ticks |
| `model fetch <name>` | ask the feeder to fetch one model into the store |
| `models` | show each resident model, its file, digest and placement tick |
| `modules [kind]` | show all catalog modules, or only skill, role or tool modules |
| `module <name>` | show one module in full |
| `skills` | show skill modules |
| `roles` | show role modules |
| `tools` | show tool modules |
| `tool <name\|all> <on\|off>` | set instance tool defaults for subsequent turns |
| `agent <id> tools` | show the tool selection of one conversation |
| `agent <id> tools <name\|all> <on\|off\|inherit>` | set conversation tool choices for subsequent turns |
| `import <path>` | import a module directory through the feeder |
| `remove <name>` | remove an imported module |
| `quit` | stop the run |

Agent IDs run from zero through one less than the profile slot count.
Agent zero is the conductor. The supplied roles are `conductor`, `worker` and `verifier`.
Bus kinds are `finding`, `rank`, `question`, `answer`, `handoff`, `cost` and `note`.
Finding provenance is `computed`, `fetched`, `recalled` or `testimony`.

The `outcome` command supplies a scripted tool result for measurement.
It does not execute that tool. See [affect](14-affect.md) before using it.

## Conversation controls

A `say` starts one conductor turn. A second `say` while that agent is busy is refused.
`stop` ends its reply at a token boundary. The transcript records the stopped result.
`continue` resumes a reply that reached its limit and starts a new turn budget.

Per-agent decode keys include `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty` and `repeat_window`.
They also include `presence_penalty`, `frequency_penalty`, `seed` and `think_limit`.
Use `absent` to remove a thinking limit.

```text
agent 0 decode.temperature 0.6
agent 0 decode.think_limit 0
agent 0 pages 160
```

A thinking limit of zero prevents a new thinking span.
A positive limit permits that many tokens inside it, then permits only its closing token.
Sampling and page changes apply to subsequent turns.

Qualified vector selection uses `agent 0 decode.steer0 NAME:DOSE`.
Use `agent 0 decode.steer0 absent` to clear it.
A selector cannot enable an unavailable control; see [control bindings](37-control-bindings.md).

## Tool selection and authorization

Instance defaults and per-conversation overrides select the effective tool set.
Role grants and required operator authorization still apply.
Turning a tool on does not bypass either boundary.

```text
tool fs_read on
agent 0 tools
agent 0 tools fs_read off
agent 0 tools fs_read inherit
```

`authorize ID` and `refuse ID` answer the exact pending request.
An authorization wait has no tool deadline. The configured deadline starts when the operator grants the request.
A completed or expired request cannot gain another execution through replay.

The built-in tool names and override rules are in [modules](09-modules.md#tool-selection).
The [tool SDK](10-tool-sdk.md) describes trusted host executables and CUDA module contracts.

## Model and module changes

`model fetch NAME` downloads into the store only. It does not replace a resident model.
`model load ROLE NAME` requires the exact active manifest entry and verified file digest.
A live sequence using that role prevents replacement; stop it first.

The current model roles include `language`, `language-q4`, `embedding`, `reranker` and `language-audio`.
Role support still depends on the file's checked tensors and wrapper.
The [model guide](16-model-files.md) describes activation and compatible replacements.

`import PATH` reads a module directory through the feeder.
Accepted module bytes enter the device catalog and journal.
Restore rebuilds that catalog from recorded imports rather than rereading changed module text.

## Memory, media and policy commands

Specialized commands use bounded files or explicit control values:

| Function | Guide |
| --- | --- |
| Load state, bind a conversation or submit a prepared query | [Live memory](20-live-memory.md) |
| Prepare a text query | [Text memory](21-text-memory.md) |
| Retain accepted source input | [Input retention](22-memory-retention.md) |
| Inspect or control appraisal | [Automatic appraisal](35-automatic-appraisal.md#controls) |
| Pause, resume or stop idle policy work | [Creator policies](34-creator-policy.md#operator-controls) |
| Enable task review cues | [Task reviews](38-task-reviews.md) |
| Submit image or audio sources | [Images](29-image-input.md) and [audio](30-audio-input.md) |

File-backed commands require regular files and do not expand shell expressions.
Their schemas state the exact input bounds, status values and recovery behavior.

## Input and output bounds

A long input uses one INPUT_LINE record and at most 31 continuation records.
The device joins fragments before parsing the command, including fragments that cross an apply batch.
A command can produce at most 32 console output records; excess output produces a cut notice.

The profile sets the prompt byte limit: 6144 for 8g, 8192 for 12g, 12288 for 24g and 24576 for 48g.
Model wrappers and selected memory consume prompt capacity too.
An accepted prompt is not silently shortened to fit.

Replay applies recorded input order and blocks an old `quit` command from stopping the restored instance.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
