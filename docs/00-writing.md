<p align="center">
  <a href="../README.md"><img src="../.github/assets/mark.png" width="360" alt="AOTX-1"></a>
</p>

# Writing conventions

[Documentation](README.md) | [Project overview](../README.md) | [Build](06-build.md) | [Operation](07-operation.md) | [API](31-http-gateway.md)

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

Product documentation uses the repository's ASD-STE100 Simplified Technical English rules and American spelling.
The source gate approximates those rules. It does not certify compliance with the complete standard.
Third-party source and license text retain their original wording.

Start at the [documentation index](README.md) for manuals and shared terms.

## Organize a guide

Open with the feature's purpose and the result the reader can obtain.
State prerequisites before commands. Place limits beside the behavior they limit.
Keep procedures separate from binary layouts and historical measurements.

Use descriptive headings and short connected paragraphs.
Use a table for comparable fields, options or limits. Use numbered steps when order matters.
Each code block needs a purpose and enough context to select the correct file, model or instance.

Keep the project title artwork and divider assets unchanged.
Use relative links between manuals and descriptive link text.
Each guide links to the index and related subjects.

## Sentence rules

- Limit descriptive sentences to 25 words and procedural sentences to 20 words.
- Limit a paragraph to six sentences.
- Use active voice, present tense and one instruction per sentence.
- Use one term for each concept; follow the shared glossary.
- Use ASCII punctuation and American spelling.
- Omit first-person wording and conversation.

A comment states a constraint, invariant or reason for its attached code.
It does not narrate an obvious operation or record development history.

## Commands and references

Use fenced blocks with a language label: `sh`, `json`, `ini`, `text`, `c` or another applicable format.
Separate literal commands from sample output. Identify placeholders before the reader uses them.
Document command failures and destructive effects where they apply.

Cite a source path and a symbol when explaining implementation behavior.
Do not cite source line numbers, which change independently of the contract.
Keep format IDs, field names, offsets, constants and digests exact.

Historical measurements retain their original model, build and hardware identities.
Do not present an earlier result as acceptance for a changed binary or model.
Keep release history separate from current operation instructions.

## Source checks

```sh
python3 tools/ste-lint.py README.md docs SECURITY.md CHANGELOG.md
python3 tools/size-gate.py README.md docs SECURITY.md CHANGELOG.md
```

The STE gate checks sentence length, paragraph length, punctuation and patterns in `tools/ste-words.txt`.
The full approved ASD-STE100 dictionary is not included.
Manual review still checks terminology, procedural sentence length, voice and technical accuracy.

<p align="center"><img src="../.github/assets/divider.png" width="720" alt=""></p>

[Documentation](README.md) | [Project overview](../README.md)
