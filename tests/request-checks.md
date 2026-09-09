# Request and tool checks

Build with `AOTX_AFFECT=ON` and `AOTX_AFFECT=OFF` before running the checks below.
The default profile has 64 agent slots. Device fixtures use one slot and all slots.
Each active slot has distinct input or policy values.

| Changed files | Required checks | Failure detected |
| --- | --- | --- |
| Agent prompt and step | `request_completion`, `prompt_policy`, `reply_limit`, `conversation`, `call_prompt`, `wrap_prompt`, `result_bounds` | Repeated refusal, lost memory choice, invalid task success, failed recovery, incorrect prompt layout or result capacity |
| Tool policy, catalog, CLI, settings | `tool_policy`, `catalog`, `settings`, `call_schema` | Incorrect precedence, changed active-turn policy, disabled call execution, lost settings |
| Transcript wire and drain | `disk_transcript`, `disk_restore`, `disk_journal`, `disk_tool_policy` | Missing refusal status or incorrect record recovery |
| CTRL tool panel and replica | `ctrl_fix`, graphical smoke check | Incorrect setting display, command bytes or refusal status |

`request_completion` supplies oversized inputs with prior memory, then runs repeated ticks.
It checks message, assigned-task and verifier refusal, including a sequence that fails to open.
It restores the recorded memory choice
and admits a short input afterward. A fixed reply tests completion without model arithmetic.
Removing the terminal transition must fail this check.

An accepted stop before sequence admission must record one stopped outcome without a
capacity error. Other pending agents must remain unchanged, and new input must start.

`prompt_policy` builds long tools-off inputs, then short tools-on inputs for the same role.
Each result must retain its own measured room and form a continuation. A shared role byte
count must fail the mixed-policy cases.

`tool_policy` checks all-off prompts and actual post-processing of disabled calls.
Its fixtures check both inherited choices and explicit conversation choices.
Removing the dispatch policy check must fail the disabled-call assertions.

Run a short real conversation with tools off and a working tool call with tools on.
Use two conversations with different choices. Save and restore the instance.
Inspect the graphical controls, console and disk transcript. Check capacity refusal and recovery.

Open the controls while disconnected and check that automatic queries do not fill the
window with repeated errors. Generated console text must not change the tool policy display.
Fixture replies alone do not establish model or graphical operation.

These checks cover request control and its consumers. They do not qualify model arithmetic,
media encoders or answer quality. Re-run those checks when their inputs or code change.
