/* Purpose: Edit the command line and parse commands.
 * Owns: The line buffer, the history and the command table.
 * Launch shape: One thread; the apply step calls these in slot order.
 * Lifetime: The whole run. */
#ifndef CLI_CUH
#define CLI_CUH

#include "seam/seam.cuh"

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
#define AOTX_CLI_HELP      10u

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

/* Write an unsigned value as decimal digits. Returns the count of bytes written. The
 * device has no library for this, so the module has its own. */
__device__ __forceinline__ unsigned int aotx_cli_utoa(unsigned long long value, char *out,
                                                      unsigned int max)
{
    char digits[20];
    unsigned int count = 0u;
    do {
        digits[count] = (char)('0' + (unsigned int)(value % 10ull));
        value /= 10ull;
        count += 1u;
    } while (value != 0ull && count < 20u);
    unsigned int written = 0u;
    while (count > 0u && written < max) {
        count -= 1u;
        out[written] = digits[count];
        written += 1u;
    }
    return written;
}

/* Write a signed value as decimal digits, with a minus sign for a value below zero. */
__device__ __forceinline__ unsigned int aotx_cli_itoa(long long value, char *out,
                                                      unsigned int max)
{
    if (value >= 0ll) {
        return aotx_cli_utoa((unsigned long long)value, out, max);
    }
    if (max == 0u) {
        return 0u;
    }
    out[0] = '-';
    return 1u + aotx_cli_utoa((unsigned long long)(-value), out + 1, max - 1u);
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
    unsigned int count = aotx_cli_utoa(value, digits, (unsigned int)sizeof digits);
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

/* Write the line as one console record and start a new line. The return is the record
 * sequence, or zero when the allowance of the command line is spent. */
__device__ __forceinline__ unsigned long long aotx_cli_console(aotx_cli_out *out)
{
    unsigned long long seq = 0ull;
    if (aotx_cli_allow()) {
        seq = aotx_seam_write(AOTX_WRITER_CONSOLE, AOTX_CLASS_B, AOTX_REC_CONSOLE, 0u,
                              out->text, out->at);
    }
    out->at = 0u;
    return seq;
}

#endif
