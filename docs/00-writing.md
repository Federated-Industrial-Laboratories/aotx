# Writing rules

All text in this repository follows ASD-STE100 Simplified Technical English: comments, this
documentation, command help, interface strings and commit messages.

## Rules

- Write short sentences. A descriptive sentence has 25 words or fewer. A procedural sentence has
  20 words or fewer.
- Write short paragraphs. A paragraph has six sentences or fewer.
- Use one term for one thing.
- Use the active voice and the present tense. Give one instruction in each sentence.
- Do not write in the first person. Do not write conversation.
- Use ASCII punctuation only.
- A comment gives the constraint, the invariant or the reason for the code it is attached to,
  and nothing else.

## The gate

`tools/ste-lint.py` checks sentence length, paragraph length, punctuation and a list of refused
words in `tools/ste-words.txt`. The gate is an approximation of the standard. The full approved
word list of ASD-STE100 is not part of this repository. A sentence that passes the gate can
still break a rule of the standard; a reviewer checks the rest.

## Spelling

The repository uses American English spelling, as ASD-STE100 does. The gate refuses common
British spellings.
