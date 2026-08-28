/* Purpose: Edit the command line and parse commands.
 * Owns: The line buffer, the history, the command table and the console buffer.
 * Launch shape: One thread; the apply step calls these in slot order.
 * Lifetime: The whole run. */
#ifndef CLI_CUH
#define CLI_CUH

#include "model/decode.cuh"
#include "model/model.cuh"
#include "sched/sched.cuh"
#include "seam/seam.cuh"
#include "text/text.cuh"

/* Feed one key event to the line editor. A completed line goes to aotx_cli_line. */
__device__ void aotx_cli_key(const aotx_key_body *key, unsigned long long tick);

/* Parse one command line and act on it. The line comes from the editor or from the terminal.
 * The function writes CONSOLE records for its output and a COMMAND record for the line. */
__device__ void aotx_cli_line(const unsigned char *text, unsigned int length,
                              unsigned long long tick);

/* Give the name of a bus kind, or a dash when the kind is not one of the seven. */
__device__ const char *aotx_cli_kind_name(unsigned int kind);

/* Give the name of a provenance value, or a dash when the message is not a finding. */
__device__ const char *aotx_cli_source_name(unsigned int provenance);

/* Lines the history holds. The oldest line goes out when a new line comes in. */
#define AOTX_CLI_HISTORY   32u

/* Bus messages that the list command shows. The count is the rows of the console panel, so
 * a full list fills the console once. The allowance of a line cuts a list that is longer. */
#define AOTX_CLI_LIST      32u

/* Help lines the help command writes. */
#define AOTX_CLI_HELP      12u

/* Key codes of the window, which are the codes of the window library. The window glue has a
 * check for each one, so a change in that library is a build error and not a wrong key. */
#define AOTX_CLI_KEY_ENTER      257u
#define AOTX_CLI_KEY_BACKSPACE  259u
#define AOTX_CLI_KEY_DELETE     261u
#define AOTX_CLI_KEY_RIGHT      262u
#define AOTX_CLI_KEY_LEFT       263u
#define AOTX_CLI_KEY_DOWN       264u
#define AOTX_CLI_KEY_UP         265u
#define AOTX_CLI_KEY_HOME       268u
#define AOTX_CLI_KEY_END        269u
#define AOTX_CLI_KEY_KP_ENTER   335u

/* Code points the editor takes and the font holds. A code point outside the range is
 * dropped, because one cell of the grid holds one byte. */
#define AOTX_CLI_CODE_FIRST   32u
#define AOTX_CLI_CODE_LAST    126u

/* Key actions of the window. A release is dropped. */
#define AOTX_CLI_RELEASE   0u
#define AOTX_CLI_PRESS     1u
#define AOTX_CLI_REPEAT    2u

/* One console line under construction. The parser fills it and then writes the record. */
typedef struct aotx_cli_out {
    char text[AOTX_BODY_BYTES];
    unsigned int at;
} aotx_cli_out;

/* What the editor holds between key events. One console gives one editor. The working
 * buffers live here and not on the stack. One thread runs the editor and the parser in slot
 * order. A buffer of this size on the stack makes the compiler spill. */
typedef struct aotx_cli_state {
    unsigned char line[AOTX_BODY_BYTES];    /* the line as it stands */
    unsigned int length;                    /* bytes of the line */
    unsigned int cursor;                    /* position in the line, from 0 to length */
    unsigned int history_count;             /* lines the history holds, up to the maximum */
    unsigned int history_first;             /* position of the oldest line in the history */
    unsigned int history_at;                /* how far back the recall went; 0 is the line */
    unsigned int lines;                     /* lines the editor completed since start */
    unsigned int keys;                      /* key events the editor took since start */
    unsigned int history_len[AOTX_CLI_HISTORY];
    unsigned char history[AOTX_CLI_HISTORY][AOTX_BODY_BYTES];
    aotx_cli_out out;                          /* the console line under construction */
    unsigned char taken[AOTX_BODY_BYTES];      /* the completed line the parser reads */
    unsigned long long recent[AOTX_CLI_LIST];  /* sequences a list command reads */
    unsigned int written;                      /* records the line that runs has written */
    unsigned int cut;                          /* lines the allowance did not let through */
} aotx_cli_state;

extern __device__ aotx_cli_state aotx_cli;

/* The quit command sets this flag. The pump reads it after each tick and stops the run.
 * The parser refuses to set the flag while a replay of the journal runs. A quit that a past
 * run typed therefore does not close the run that replays it. */
extern __device__ unsigned int aotx_cli_quit;

/* Commands the parser knows, and what the counter of each one counts. */
typedef struct aotx_cli_counts {
    unsigned int commands;   /* lines the parser took */
    unsigned int unknown;    /* lines whose first word is not a command */
    unsigned int refused;    /* lines a command refused, such as a finding with no source */
    unsigned int appended;   /* bus messages the parser appended */
} aotx_cli_counts;

extern __device__ aotx_cli_counts aotx_cli_count;

/* Give the byte count of a text that ends with a zero byte. The bound stops a text that has
 * no zero byte from a read past the body of a record. */
__device__ __forceinline__ unsigned int aotx_cli_length(const char *text)
{
    unsigned int at = 0u;
    while (at < AOTX_BODY_BYTES && text[at] != '\0') {
        at += 1u;
    }
    return at;
}

__device__ __forceinline__ void aotx_cli_clear(aotx_cli_out *out)
{
    out->at = 0u;
}

/* Add a run of bytes to a console line. Bytes past the end of the body are dropped. */
__device__ __forceinline__ void aotx_cli_add(aotx_cli_out *out, const char *text,
                                             unsigned int length)
{
    for (unsigned int i = 0u; i < length && out->at < AOTX_BODY_BYTES; ++i) {
        out->text[out->at] = text[i];
        out->at += 1u;
    }
}

/* Add a text that ends with a zero byte. */
__device__ __forceinline__ void aotx_cli_say(aotx_cli_out *out, const char *text)
{
    aotx_cli_add(out, text, aotx_cli_length(text));
}

/* Add a decimal value. */
__device__ __forceinline__ void aotx_cli_num(aotx_cli_out *out, unsigned long long value)
{
    char digits[24];
    unsigned int count = aotx_text_utoa(value, digits, (unsigned int)sizeof digits);
    aotx_cli_add(out, digits, count);
}

/* Read a slot of the device ring. A reader never writes the slot, so the sequence field
 * stays as the writer left it. The console and the panels share these three readers. */
__device__ __forceinline__ const volatile aotx_record_header *aotx_cli_slot(
    unsigned long long seq)
{
    const unsigned char *at = aotx_seam.dev.base
                            + ((seq - 1ull) & aotx_seam.dev.mask)
                              * (unsigned long long)AOTX_SLOT_BYTES;
    return (const volatile aotx_record_header *)at;
}

/* Report whether the slot of a sequence holds the published record of that sequence. */
__device__ __forceinline__ int aotx_cli_holds(const volatile aotx_record_header *header,
                                              unsigned long long seq, unsigned int type)
{
    return header->seq == seq && header->magic == AOTX_WIRE_MAGIC
        && header->type == (unsigned char)type;
}

/* Slots that the command layer looks back over for the newest record of a type. This walk
 * runs on one thread inside the tick, so it carries a bound. The tick writes a statistics
 * record every tick, which puts the newest one at the tail. */
#define AOTX_CLI_SPAN   4096ull

/* Find the newest record of a type within the span. The return is zero when the span holds
 * none of them. A slot with a different sequence that is not zero was taken by a newer
 * record. Every older record went out with it, so the walk stops there. A slot that holds
 * zero is under write, which happens at the tail only, and the walk steps over it. */
__device__ __forceinline__ unsigned long long aotx_cli_last(unsigned int type)
{
    unsigned long long tail = aotx_seam.dev.tail;
    unsigned long long window = AOTX_CLI_SPAN;
    if (window > tail) {
        window = tail;
    }
    for (unsigned long long i = 0ull; i < window; ++i) {
        unsigned long long seq = tail - i;
        const volatile aotx_record_header *header = aotx_cli_slot(seq);
        unsigned long long got = header->seq;
        if (got != seq) {
            if (got != 0ull) {
                return 0ull;
            }
            continue;
        }
        if (header->magic == AOTX_WIRE_MAGIC && header->type == (unsigned char)type) {
            return seq;
        }
    }
    return 0ull;
}

/* Report whether the line that runs may write one more record, and take the allowance when
 * it may. The tick start reserves AOTX_CLI_RECORDS_EACH sequences for each input, so a line
 * that wrote more would take a sequence that the reservation does not cover. One sequence of
 * the allowance is kept for the line that states the cut. */
__device__ __forceinline__ int aotx_cli_allow(void)
{
    if (aotx_cli.written + 1u >= (unsigned int)AOTX_CLI_RECORDS_EACH) {
        aotx_cli.cut += 1u;
        return 0;
    }
    aotx_cli.written += 1u;
    return 1;
}

/* The console buffer. The console panel reads its lines from here and not from the record
 * ring. A tick load of thousands of records writes over a console record in a fraction of a
 * second. A line that the operator typed must stay on the panel. The buffer is device state
 * of the run that makes the lines; the disk holds the records, not the buffer. */
#define AOTX_CONSOLE_LINES   256u
#define AOTX_CONSOLE_COLS    160u

/* One line of the console buffer. */
typedef struct aotx_console_line {
    unsigned long long seq;   /* the line number; zero while a writer fills the line */
    unsigned int length;      /* bytes of the line, up to AOTX_CONSOLE_COLS */
    unsigned char text[AOTX_CONSOLE_COLS];
} aotx_console_line;

typedef struct aotx_console_state {
    unsigned long long count;                    /* lines put in since the start of the run */
    aotx_console_line line[AOTX_CONSOLE_LINES];
} aotx_console_state;

extern __device__ aotx_console_state aotx_console;

/* The count of lines is a power of two, so a line number gives its place with a mask. */
typedef char aotx_console_check[((AOTX_CONSOLE_LINES & (AOTX_CONSOLE_LINES - 1u)) == 0u)
                                ? 1 : -1];

/* Put one line in the console buffer. A writer claims a line number, fills the line, and
 * then publishes the number. A reader that sees the number therefore sees the whole line.
 * A line longer than the buffer holds is cut at AOTX_CONSOLE_COLS bytes. */
__device__ __forceinline__ unsigned long long aotx_console_put(const unsigned char *text,
                                                               unsigned int length)
{
    unsigned long long at = atomicAdd(&aotx_console.count, 1ull) + 1ull;
    aotx_console_line *line = &aotx_console.line[(at - 1ull) & (AOTX_CONSOLE_LINES - 1u)];
    if (length > AOTX_CONSOLE_COLS) {
        length = AOTX_CONSOLE_COLS;
    }
    aotx_seam_release_gpu(&line->seq, 0ull);
    for (unsigned int i = 0u; i < length; ++i) {
        line->text[i] = text[i];
    }
    line->length = length;
    aotx_seam_release_gpu(&line->seq, at);
    return at;
}

/* Add bytes to the end of a line that the buffer holds, and give the count that went in.
 * The writer takes the line number away and then fills the line. It puts the number back
 * after that, so a reader that sees the number sees the whole line. The return is below
 * length when the line is full. The return is zero when the buffer no longer holds the
 * line. A reply of a sequence grows the line that the say command started. */
__device__ __forceinline__ unsigned int aotx_console_grow(unsigned long long at,
                                                          const unsigned char *text,
                                                          unsigned int length)
{
    if (at == 0ull) {
        return 0u;
    }
    aotx_console_line *line = &aotx_console.line[(at - 1ull) & (AOTX_CONSOLE_LINES - 1u)];
    if (line->seq != at) {
        return 0u;
    }
    unsigned int held = line->length;
    if (held >= AOTX_CONSOLE_COLS) {
        return 0u;
    }
    unsigned int room = AOTX_CONSOLE_COLS - held;
    if (length > room) {
        length = room;
    }
    aotx_seam_release_gpu(&line->seq, 0ull);
    for (unsigned int i = 0u; i < length; ++i) {
        line->text[held + i] = text[i];
    }
    line->length = held + length;
    aotx_seam_release_gpu(&line->seq, at);
    return length;
}

/* Give the line of a line number, or a null pointer when the buffer no longer holds it. The
 * reader must look at the number again after it copies the bytes. */
__device__ __forceinline__ const volatile aotx_console_line *aotx_console_at(
    unsigned long long at)
{
    const volatile aotx_console_line *line =
        &aotx_console.line[(at - 1ull) & (AOTX_CONSOLE_LINES - 1u)];
    return (line->seq == at) ? line : 0;
}

/* Write one console record and put the same line in the console buffer. Two functions write
 * a console record: this one and aotx_cli_echo. Each one fills the buffer as well, so the
 * panel and the journal never hold different lines. */
__device__ __forceinline__ unsigned long long aotx_console_write(const char *text,
                                                                 unsigned int length)
{
    unsigned long long seq = aotx_seam_write(AOTX_WRITER_CONSOLE, AOTX_CLASS_B,
                                             AOTX_REC_CONSOLE, 0u, text, length);
    aotx_console_put((const unsigned char *)text, length);
    return seq;
}

/* Write one console record and give the number of the buffer line it made. The say command
 * keeps that number, because the reply of its sequence grows the same line. */
__device__ __forceinline__ unsigned long long aotx_console_start(const char *text,
                                                                 unsigned int length)
{
    aotx_seam_write(AOTX_WRITER_CONSOLE, AOTX_CLASS_B, AOTX_REC_CONSOLE, 0u, text, length);
    return aotx_console_put((const unsigned char *)text, length);
}

/* Write the line as one console record and start a new line. The return is the record
 * sequence, or zero when the allowance of the command line is spent. */
__device__ __forceinline__ unsigned long long aotx_cli_console(aotx_cli_out *out)
{
    unsigned long long seq = 0ull;
    if (aotx_cli_allow()) {
        seq = aotx_console_write(out->text, out->at);
    }
    out->at = 0u;
    return seq;
}

/* Write the echo of an input line at a sequence that the apply step keeps for it. The same
 * bytes go in the console buffer. The apply step calls this in the order of the inputs and
 * before the parser writes the answer. The buffer therefore holds the echo above the answer.
 * A replay does not echo. After a restore the buffer holds the lines the parser made again,
 * and no line that came in before the restore. */
__device__ __forceinline__ void aotx_cli_echo(unsigned long long seq,
                                              const unsigned char *text,
                                              unsigned int length)
{
    unsigned int shown = length;
    if (shown > AOTX_BODY_BYTES - 2u) {
        shown = AOTX_BODY_BYTES - 2u;
    }
    aotx_record_header *header = aotx_seam_slot(seq);
    unsigned char *line = aotx_seam_body(header);
    line[0] = (unsigned char)'>';
    line[1] = (unsigned char)' ';
    for (unsigned int i = 0u; i < shown; ++i) {
        line[2u + i] = text[i];
    }
    aotx_seam_publish(header, seq, AOTX_WRITER_CONSOLE, AOTX_CLASS_B, AOTX_REC_CONSOLE, 0u,
                      shown + 2u);
    aotx_console_put(line, shown + 2u);
}

/* ---------------------------------------------------------------------------------------
 * The say command and the reply that streams into the console.
 *
 * The parser cannot launch a kernel, and the tokenizer of the device is a run of kernels.
 * The say command therefore leaves the wrapped prompt bytes in the table below. Nodes of
 * the tick graph then tokenize the table and open the sequence in the same tick.
 * ------------------------------------------------------------------------------------ */

/* The slot of the conductor. The console follows one sequence in this version. */
#define AOTX_SAY_SLOT      0u

/* Bytes of one wrapped prompt. The line the editor takes is at most AOTX_BODY_BYTES. The
 * pieces of the chat wrap add 69 bytes, so this bound holds a full line and more. */
#define AOTX_SAY_BYTES     512u

/* Bytes of a wrapped prompt after the clean step. That step gives at most three bytes for
 * one byte which is not part of a character. */
#define AOTX_SAY_CLEAN     (3u * AOTX_SAY_BYTES)

/* Piece slots and token slots of one prompt. A piece holds one byte at the least. A token
 * holds one byte at the least. The count of each one is therefore under the byte count. */
#define AOTX_SAY_PIECES    AOTX_SAY_BYTES
#define AOTX_SAY_TOKENS    AOTX_SAY_BYTES

/* Blocks of the merge step of this path, and the warps they hold. Each warp holds one piece
 * at a time and takes AOTX_TEXT_WARP_BYTES of the merge memory. */
#define AOTX_SAY_BLOCKS    8u
#define AOTX_SAY_WARPS     (AOTX_SAY_BLOCKS * AOTX_TEXT_WARPS)

/* Bytes of reply that one sequence gives the console in one tick. The bound is the width of
 * a console line, and it is under the body of a record. */
#define AOTX_SAY_TAKE      AOTX_CONSOLE_COLS

/* The sampling the model card of the language model gives for thinking off: temperature 0.7,
 * top_p 0.8, top_k 20. The card gives temperature 0.6 and top_p 0.95 for thinking on, which
 * is what the file's generation settings hold. The wrap of the say command turns thinking
 * off, so the values here are the first set. */
#define AOTX_SAY_TEMPERATURE  0.7f
#define AOTX_SAY_TOP_K        20u
#define AOTX_SAY_TOP_P        0.8f

/* What one slot of the say path holds. The command layer fills the prompt fields; the nodes
 * of the tick graph read them, open the sequence, and then show the reply. */
typedef struct aotx_say_slot {
    unsigned int wanted;          /* 1 when a prompt waits for the tokenize step */
    unsigned int length;          /* bytes of the wrapped prompt */
    unsigned int live;            /* 1 while the console shows the reply of this slot */
    unsigned int tokens;          /* reply tokens the commit made, as the last take saw them */
    unsigned int prompt;          /* prompt tokens the tokenize step gave */
    unsigned int column;          /* 1 when the line that grows is open */
    unsigned long long at;        /* the console line the reply grows into, or zero */
    unsigned long long opened;    /* the tick the sequence opened */
    unsigned char text[AOTX_SAY_TAKE];  /* the bytes of one take, and then the end message */
} aotx_say_slot;

typedef struct aotx_say_state {
    aotx_say_slot slot[AOTX_SEQ_SLOTS];
    unsigned char prompt[AOTX_SEQ_SLOTS][AOTX_SAY_BYTES];
    unsigned int said;            /* say commands the parser took */
    unsigned int refused;         /* say commands the parser refused */
    unsigned int stopped;         /* stop commands that ended a reply */
    unsigned int opened;          /* sequences the say path opened */
    unsigned int shown;           /* takes that put bytes on the console */
} aotx_say_state;

extern __device__ aotx_say_state aotx_say;

/* The memory the tokenizer of this path holds. One block holds every array, so the host
 * glue reads one address and gives the parts to the kernels of the tokenizer. The block is
 * device state of the run and never crosses the seam. */
typedef struct aotx_say_work {
    unsigned char clean[AOTX_SEQ_SLOTS * AOTX_SAY_CLEAN];
    unsigned int start[AOTX_SEQ_SLOTS];
    unsigned int length[AOTX_SEQ_SLOTS];
    unsigned int clean_start[AOTX_SEQ_SLOTS];
    unsigned int clean_length[AOTX_SEQ_SLOTS];
    unsigned int piece_start[AOTX_SEQ_SLOTS * AOTX_SAY_PIECES];
    unsigned int piece_length[AOTX_SEQ_SLOTS * AOTX_SAY_PIECES];
    unsigned int piece_token[AOTX_SEQ_SLOTS * AOTX_SAY_PIECES];
    unsigned int piece_count[AOTX_SEQ_SLOTS];
    unsigned int work[AOTX_SEQ_SLOTS * AOTX_SAY_PIECES];
    unsigned int works;
    unsigned int chunk[AOTX_SEQ_SLOTS * AOTX_SAY_PIECES];
    unsigned int scratch[AOTX_SEQ_SLOTS * AOTX_SAY_CLEAN];
    unsigned char merge[AOTX_SAY_WARPS * AOTX_TEXT_WARP_BYTES];
} aotx_say_work;

extern __device__ aotx_say_work aotx_say_gear;

/* The tokens of each slot and their counts. The check reads them against the golden list. */
extern __device__ unsigned int aotx_say_id[AOTX_SEQ_SLOTS * AOTX_SAY_TOKENS];
extern __device__ unsigned int aotx_say_count[AOTX_SEQ_SLOTS];

/* The two pieces of the chat wrap that the language model file carries. The file gives them
 * as a template with conditions. This path takes two of those conditions: one user message
 * with a generation prompt, and thinking off. The template then gives these bytes exactly.
 * The tokenizer matches a special token before it reads the character classes. Each control
 * name in the wrap therefore becomes one token. */
__device__ static const char aotx_say_head[] = "<|im_start|>user\n";
__device__ static const char aotx_say_tail[] =
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/* Copy a text that ends with a zero byte into a prompt and give the position after it. */
__device__ __forceinline__ unsigned int aotx_say_put(unsigned char *out, unsigned int at,
                                                     const char *text)
{
    for (unsigned int i = 0u; text[i] != '\0' && at < AOTX_SAY_BYTES; ++i) {
        out[at] = (unsigned char)text[i];
        at += 1u;
    }
    return at;
}

/* Put the chat wrap of a text in the prompt table of a slot. The return is 0, or 1 when the
 * slot is busy or the text does not fit. The parser calls this from its serial thread.
 *
 * The function is in the header because the parser runs inside the apply node of the tick.
 * A call from that node into another translation unit makes the node keep a frame. That
 * frame holds the registers of the call, and the spill gate binds it. */
__device__ __forceinline__ int aotx_say_ask(unsigned int slot, const unsigned char *text,
                                            unsigned int length)
{
    if (slot >= AOTX_SEQ_SLOTS) {
        return 1;
    }
    aotx_say_slot *state = &aotx_say.slot[slot];
    if (state->wanted != 0u || state->live != 0u) {
        return 1;
    }
    unsigned int wrap = aotx_cli_length(aotx_say_head) + aotx_cli_length(aotx_say_tail);
    if (length + wrap > AOTX_SAY_BYTES) {
        return 1;
    }
    unsigned char *out = aotx_say.prompt[slot];
    unsigned int at = aotx_say_put(out, 0u, aotx_say_head);
    for (unsigned int i = 0u; i < length; ++i) {
        out[at] = text[i];
        at += 1u;
    }
    at = aotx_say_put(out, at, aotx_say_tail);
    state->length = at;
    state->at = 0ull;
    state->column = 0u;
    state->tokens = 0u;
    state->prompt = 0u;
    state->wanted = 1u;
    return 0;
}

/* Give the name of a sequence state, or a dash when the value is not one of the four. */
__device__ __forceinline__ const char *aotx_say_state_name(unsigned int state)
{
    switch (state) {
    case AOTX_SEQ_STATE_FREE:    return "free";
    case AOTX_SEQ_STATE_PREFILL: return "prefill";
    case AOTX_SEQ_STATE_DECODE:  return "decode";
    case AOTX_SEQ_STATE_DONE:    return "done";
    default:                     return "-";
    }
}

/* Give the name of a model role, or a dash when the value is not one of the four. */
__device__ __forceinline__ const char *aotx_say_role_name(unsigned int role)
{
    switch (role) {
    case AOTX_MODEL_EMBEDDING:   return "embedding";
    case AOTX_MODEL_RERANKER:    return "reranker";
    case AOTX_MODEL_LANGUAGE:    return "language";
    case AOTX_MODEL_LANGUAGE_Q4: return "language-q4";
    default:                     return "-";
    }
}

/* Put a run of reply bytes on the console line of a slot. A newline byte ends that line and
 * the bytes after it start a new one. A line which is full also starts a new one. */
__device__ void aotx_say_show(unsigned int slot, const unsigned char *text,
                              unsigned int length);


/* Ticks that one rate sample covers. The window holds one sample of each slot for each of
 * these ticks. */
#define AOTX_SAY_WINDOW    16u

/* One sample of one slot, written once each tick. The device clock gives the time, so a
 * rate from two samples is a measurement of this run and not the pace of the pump. */
typedef struct aotx_say_sample {
    unsigned long long tick;      /* the tick the sample was taken; zero when empty */
    unsigned long long ns;        /* the device clock at the start of that tick */
    unsigned long long opened;    /* the tick the sequence opened, which keeps two apart */
    unsigned int sampled;         /* reply tokens the sequence had made */
    unsigned int reserved;
} aotx_say_sample;

extern __device__ aotx_say_sample aotx_say_window[AOTX_SEQ_SLOTS][AOTX_SAY_WINDOW];

/* Give the reply tokens each second of a slot, from the oldest and the newest sample of one
 * sequence in the window. Both the tokens and the time come from those two samples. The
 * return is zero when the window holds fewer than two samples of the sequence that runs. */
__device__ __forceinline__ unsigned long long aotx_say_rate(unsigned int slot)
{
    if (slot >= AOTX_SEQ_SLOTS) {
        return 0ull;
    }
    const aotx_say_sample *last = 0;
    for (unsigned int i = 0u; i < AOTX_SAY_WINDOW; ++i) {
        const aotx_say_sample *at = &aotx_say_window[slot][i];
        if (at->tick != 0ull && (last == 0 || at->tick > last->tick)) {
            last = at;
        }
    }
    if (last == 0) {
        return 0ull;
    }
    const aotx_say_sample *first = 0;
    for (unsigned int i = 0u; i < AOTX_SAY_WINDOW; ++i) {
        const aotx_say_sample *at = &aotx_say_window[slot][i];
        if (at->tick == 0ull || at->opened != last->opened) {
            continue;
        }
        if (first == 0 || at->tick < first->tick) {
            first = at;
        }
    }
    if (first == 0 || first == last || last->ns <= first->ns
        || last->sampled < first->sampled) {
        return 0ull;
    }
    return ((unsigned long long)(last->sampled - first->sampled) * 1000000000ull)
           / (last->ns - first->ns);
}

/* The nodes of the say path, in the order the tick graph holds them. The fill step writes
 * the batch table of the tokenizer and clears the work count. The start step opens a
 * sequence for each prompt the tokenize step read. The reply step takes the new bytes of
 * each live sequence and puts them on the console. */
__global__ void aotx_say_fill(void);
__global__ void aotx_say_start(void);
__global__ void aotx_say_reply(void);

/* Host glue: capture the say nodes into the stream that is capturing the tick graph. The
 * say nodes go after the apply node and before the plan of the decode. The reply node goes
 * after the commit of the decode and before the flush. Each one returns 0 when the nodes
 * are in the stream. */
int aotx_cli_say_capture(void *stream);
int aotx_cli_reply_capture(void *stream);

#endif
