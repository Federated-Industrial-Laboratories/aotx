# Model accuracy fixtures

These files hold the results of llama.cpp for the three models of the model set. The device
kernels are compared with them. Every file here comes from llama.cpp commit
`6c84c7d5d8833c6e0df69628f75a0f599797934e`, built for the processor only, so all the math is
float32 and no graphics card takes part.

The program that made these files is `aotx-ref`. It is a small program against the llama.cpp
library, and it is not part of this repository. Each command below names the model file, the
input file and the output.

## The model files

| role | file | sha256 |
| --- | --- | --- |
| language | `Qwen3-4B-Q8_0.gguf` | `8c2f07f26af9747e41988551106f149b03eb9b5cb6df636027b6bf6278473300` |
| language, four bits | `Qwen3-4B-Q4_0.gguf` | `ae782b4a90b57dc4faf880855ea4b285a96b4a335544144526dab73df91d4d61` |
| embedding | `Qwen3-Embedding-0.6B-Q8_0.gguf` | `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` |
| reranker | `qwen3-reranker-0.6b-q8_0.gguf` | `22c9979ce4fbcdc5acdc310c6641c32797eff1aa980b8f7a2db8a8ea23429a48` |

The four bit file is made from the eight bit file with `llama-quantize`, in this way:

```
llama-quantize --allow-requantize --token-embedding-type q8_0 \
    Qwen3-4B-Q8_0.gguf Qwen3-4B-Q4_0.gguf Q4_0
```

The model set gives no four bit file, and the half precision source is not on this machine.
The values of the four bit file are therefore two steps away from the released weights. The
option for the token embedding keeps that one tensor at eight bits. Without it the tool makes
it a K quant block type, which the weight reader of this repository does not accept. The file
holds 145 float32 tensors, 1 eight bit tensor and 252 four bit tensors.

## Why some files have no text suffix

The register gate reads files that end with `.md` and `.txt`. The prompt file and the pair
file hold text that a person gives to a model, not text of this repository. They end with
`.dat` and `.bin`, which the gate does not read, and the register rules do not apply to them.

## Files

| file | what it holds | bytes |
| --- | --- | --- |
| `prompts.dat` | 8 prompts, one for each line | 1,800 |
| `lm-0.pos` to `lm-7.pos` | logits of every position, eight bit weights | 238,790 |
| `lm-0-last.f32` to `lm-3-last.f32` | all the logits of the last position, four prompts | 2,430,976 |
| `lm-q4-0.pos` to `lm-q4-7.pos` | logits of every position, four bit weights | 238,280 |
| `embed-lines.bin` | 70 lines, the first 70 lines of the token list fixture | 4,608 |
| `embed.f32` | one vector for each line | 286,720 |
| `rerank-pairs.dat` | 16 query and document pairs | 2,608 |
| `rerank.f32` | one value for each pair | 64 |

The files in the table are 3,203,846 bytes together. Each file that holds float32 values
has a file beside it that ends with `.head`. That file names the command, the model file
and the format. The 6 of them add 2,639 bytes. With this file the directory stays under 3.3 MB,
which is inside the 8 MB that the fixtures may take.

All the logits of one position are 607,744 bytes, because the vocabulary of the language
model holds 151,936 tokens. Four of the eight prompts keep that file. The largest logit and
the largest 32 of every position are kept for all eight prompts.

## The prompts and the chat wrap

```
aotx-ref logits <model file> prompts.dat <out dir> --prefix lm --full 4
aotx-ref logits <model file> prompts.dat <out dir> --prefix lm-q4 --full 0
```

`prompts.dat` holds one prompt for each line. The program changes the two characters
backslash and n into one newline byte, and two backslashes into one backslash. The newline
at the end of the line is not part of the prompt.

The program then puts the prompt in the chat wrap of the model file. The wrap is what
`tokenizer.chat_template` gives for one user message with a generation prompt and no tools:

```
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

The last line ends with a newline. The tokenizer reads the special tokens and adds no begin
token, because the model file sets `tokenizer.ggml.add_bos_token` to false. The prompts are
plain text, a paragraph with a question, two blocks of code, a list, French, Chinese, and a
set of numbers to sort. They give 27, 74, 67, 86, 76, 48, 34 and 93 positions.

## The position files

A `.pos` file starts with lines that begin with `#`. They name the commit, the model file and
its sha256, the prompt, the chat wrap and the token identities of the prompt.

Each row after them holds the position and the identity of the largest logit. Then come 32
pairs of identity and logit, largest first, with a colon between the two parts of a pair. A
logit has five digits after the point.

## Four bit weights against eight bit weights

The `lm-q4-<i>.pos` files come from the four bit file and the same prompts. Of the 505
positions of the eight prompts, 424 give the same largest logit as the eight bit file, which
is 83.96 percent. In 493 of the 505 positions the largest identity of the four bit file is
one of the five largest of the eight bit file. That is 97.62 percent. The lowest agreement of
one prompt is 61.8 percent, on the Chinese prompt, and the highest is 92.5 percent.

## The embedding vectors

```
aotx-ref embed <model file> embed-lines.bin embed.f32
```

`embed-lines.bin` is a byte for byte copy of the first 70 lines of the token list fixture,
with sha256 `20e4b4865a65e06b1067f48ded3c6d40474514d36c12b945955f30fdc69010ca`. The token
list fixture has grown since; its first 70 lines are these. Each line keeps the newline
byte at its end, which is the rule of that fixture. The model file sets
`tokenizer.ggml.add_eos_token` to true, so the tokenizer adds token 151643 after the text.
Line 0 therefore gives 11 tokens, which are the 10 of the token list fixture and that one.

The tokenizer reads a special token as one token and not as the bytes of its name. Lines 60
to 63 hold `<|endoftext|>`, `<|im_start|>` and `<|im_end|>`. Their token runs are the runs
of rows 61 to 64 of the token list fixture, with token 151643 after them. A run that
reads those names as bytes gives 24 tokens for line 63 in place of 8, and a vector the
device never makes.

`embed.f32` holds 70 rows of 1024 float32 values, one row for each line, row after row. The
pooling is the last token, which `qwen3.pooling_type` of the model file asks for. Each row is
scaled to unit length. The largest and the smallest length in the file are both 1.000000.

## The rerank values

```
aotx-ref rerank <model file> rerank-pairs.dat rerank.f32
```

`rerank-pairs.dat` holds 16 lines. Each line holds the query, a tab byte, and the document.
The first 8 pairs are relevant and the last 8 are not. `rerank.f32` holds 16 float32 values,
one for each line, in the order of the lines. A value is the chance that the answer is yes.

The template comes from `tokenizer.chat_template.rerank` of the model file, with `{query}`
and `{document}` replaced:

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: Given a web search query, retrieve relevant passages that answer the query
<Query>: {query}
<Document>: {document}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

llama.cpp reads the last position, applies the head `cls.output.weight`, and applies a
softmax over the two values. The model file names the two class outputs `yes` and `no` in
that order in `qwen3.classifier.output_labels`. Row 0 of `cls.output.weight` is equal, value
for value, to row 9693 of `token_embd.weight`, which is the token `yes`. Row 1 is equal to
row 2152, which is the token `no`. The first value is therefore the chance of yes.

The 8 relevant pairs give values from 0.721314 to 0.999209. The 8 pairs that are not relevant
give values from 0.000002 to 0.000040. The two groups are apart by more than four orders of
magnitude.
