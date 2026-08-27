/* Purpose: Build the command line from key records, with a cursor and a history.
 * Owns: The editor state, the history ring, the quit flag and the console buffer.
 * Launch shape: One thread; the apply step calls the editor in slot order.
 * Lifetime: The whole run. */
#include "cli/cli.cuh"

__device__ aotx_cli_state aotx_cli;
__device__ unsigned int aotx_cli_quit;
__device__ aotx_cli_counts aotx_cli_count;

/* The console buffer of this run. The panel reads it and the disk never holds it. */
__device__ aotx_console_state aotx_console;

/* Put one byte at the cursor and move the cursor past it. A full line takes no more bytes. */
static __device__ __forceinline__ void aotx_cli_insert(unsigned char byte)
{
    if (aotx_cli.length >= AOTX_BODY_BYTES) {
        return;
    }
    for (unsigned int i = aotx_cli.length; i > aotx_cli.cursor; --i) {
        aotx_cli.line[i] = aotx_cli.line[i - 1u];
    }
    aotx_cli.line[aotx_cli.cursor] = byte;
    aotx_cli.cursor += 1u;
    aotx_cli.length += 1u;
}

/* Take the byte at a position out of the line. */
static __device__ __forceinline__ void aotx_cli_erase(unsigned int at)
{
    if (at >= aotx_cli.length) {
        return;
    }
    for (unsigned int i = at; i + 1u < aotx_cli.length; ++i) {
        aotx_cli.line[i] = aotx_cli.line[i + 1u];
    }
    aotx_cli.length -= 1u;
    if (aotx_cli.cursor > aotx_cli.length) {
        aotx_cli.cursor = aotx_cli.length;
    }
}

/* Give the history position of the line that is back steps from the newest. The history is
 * a ring, so the position of the oldest line moves as lines come in. */
static __device__ __forceinline__ unsigned int aotx_cli_slot_of(unsigned int back)
{
    unsigned int newest = aotx_cli.history_count - 1u;
    unsigned int index = newest - (back - 1u);
    return (aotx_cli.history_first + index) % AOTX_CLI_HISTORY;
}

/* Put a history line in the editor. The line that was there is lost, which is what a recall
 * of a terminal does. */
static __device__ __forceinline__ void aotx_cli_recall(unsigned int back)
{
    if (back == 0u) {
        aotx_cli.length = 0u;
        aotx_cli.cursor = 0u;
        return;
    }
    unsigned int slot = aotx_cli_slot_of(back);
    unsigned int length = aotx_cli.history_len[slot];
    if (length > AOTX_BODY_BYTES) {
        length = AOTX_BODY_BYTES;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        aotx_cli.line[i] = aotx_cli.history[slot][i];
    }
    aotx_cli.length = length;
    aotx_cli.cursor = length;
}

/* Put a completed line in the history. The oldest line goes out when the ring is full. */
static __device__ __forceinline__ void aotx_cli_remember(const unsigned char *text,
                                                         unsigned int length)
{
    unsigned int slot;
    if (aotx_cli.history_count < AOTX_CLI_HISTORY) {
        slot = (aotx_cli.history_first + aotx_cli.history_count) % AOTX_CLI_HISTORY;
        aotx_cli.history_count += 1u;
    } else {
        slot = aotx_cli.history_first;
        aotx_cli.history_first = (aotx_cli.history_first + 1u) % AOTX_CLI_HISTORY;
    }
    for (unsigned int i = 0u; i < length; ++i) {
        aotx_cli.history[slot][i] = text[i];
    }
    aotx_cli.history_len[slot] = length;
}

/* Complete the line: keep a copy, clear the editor, remember the line, then parse it. The
 * copy comes first, because the parser writes records and the editor must be ready for the
 * key that follows. */
static __device__ __forceinline__ void aotx_cli_enter(unsigned long long tick)
{
    unsigned char *text = aotx_cli.taken;
    unsigned int length = aotx_cli.length;
    for (unsigned int i = 0u; i < length; ++i) {
        text[i] = aotx_cli.line[i];
    }
    aotx_cli.length = 0u;
    aotx_cli.cursor = 0u;
    aotx_cli.history_at = 0u;
    aotx_cli.lines += 1u;
    if (length == 0u) {
        return;
    }
    aotx_cli_remember(text, length);
    aotx_cli_line(text, length, tick);
}

/* One key event. A character event carries a code point and no key code. A key event carries
 * a key code. A release event changes nothing. */
__device__ void aotx_cli_key(const aotx_key_body *key, unsigned long long tick)
{
    unsigned int action = key->action;
    if (action == AOTX_CLI_RELEASE) {
        return;
    }
    aotx_cli.keys += 1u;

    unsigned int code = key->codepoint;
    if (key->key == 0u) {
        /* This version holds one byte for each cell. A code point above the font is
         * dropped and not cut into bytes that the grid cannot show. */
        if (code >= AOTX_CLI_CODE_FIRST && code <= AOTX_CLI_CODE_LAST) {
            aotx_cli_insert((unsigned char)code);
            aotx_cli.history_at = 0u;
        }
        return;
    }

    switch (key->key) {
    case AOTX_CLI_KEY_BACKSPACE:
        if (aotx_cli.cursor > 0u) {
            aotx_cli.cursor -= 1u;
            aotx_cli_erase(aotx_cli.cursor);
        }
        break;
    case AOTX_CLI_KEY_DELETE:
        aotx_cli_erase(aotx_cli.cursor);
        break;
    case AOTX_CLI_KEY_LEFT:
        if (aotx_cli.cursor > 0u) {
            aotx_cli.cursor -= 1u;
        }
        break;
    case AOTX_CLI_KEY_RIGHT:
        if (aotx_cli.cursor < aotx_cli.length) {
            aotx_cli.cursor += 1u;
        }
        break;
    case AOTX_CLI_KEY_HOME:
        aotx_cli.cursor = 0u;
        break;
    case AOTX_CLI_KEY_END:
        aotx_cli.cursor = aotx_cli.length;
        break;
    case AOTX_CLI_KEY_UP:
        if (aotx_cli.history_at < aotx_cli.history_count) {
            aotx_cli.history_at += 1u;
            aotx_cli_recall(aotx_cli.history_at);
        }
        break;
    case AOTX_CLI_KEY_DOWN:
        if (aotx_cli.history_at > 0u) {
            aotx_cli.history_at -= 1u;
            aotx_cli_recall(aotx_cli.history_at);
        }
        break;
    case AOTX_CLI_KEY_ENTER:
    case AOTX_CLI_KEY_KP_ENTER:
        aotx_cli_enter(tick);
        break;
    default:
        break;
    }
}
