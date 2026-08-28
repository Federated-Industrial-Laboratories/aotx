/* Purpose: Read the settings file that the boot glue, the feeder and the terminal share.
 * Owns: Nothing; the caller owns the table and every buffer that a call fills.
 * Threading: One thread; no function here holds state between calls.
 * Lifetime: The call. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/settings/settings.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* The longest line that the reader takes. A key of 63 bytes, a text of 255 bytes, the
 * equals sign and the spaces around them are far below this count. */
#define AOTX_SETTINGS_LINE_BYTES 512u

/* The block that one read of the file gives. */
#define AOTX_SETTINGS_READ_BYTES 4096u

/* The name of the file that a write makes in the directory of the settings file. The last
 * six characters become the ones that mkstemp gives. */
#define AOTX_SETTINGS_TEMP "aotx-settings-XXXXXX"

/* ---- the tables that keys.h gives ---- */

typedef struct number_row {
    const char  *name;
    unsigned int side;
    unsigned int effect;
    int64_t      value;   /* the default, in the scaled unit */
    int64_t      least;
    int64_t      most;
    int          scale;
} number_row;

typedef struct text_row {
    const char  *name;
    unsigned int side;
    unsigned int effect;
    const char  *value;   /* the default */
} text_row;

#define AOTX_NUMBER_ROW(symbol, name, side, effect, value, least, most, scale) \
    { name, AOTX_SETTING_SIDE_##side, AOTX_SETTING_AT_##effect, value, least, most, scale },
#define AOTX_TEXT_ROW(symbol, name, side, effect, value) \
    { name, AOTX_SETTING_SIDE_##side, AOTX_SETTING_AT_##effect, value },

static const number_row aotx_numbers[AOTX_SETTING_NUMBER_COUNT] = {
    AOTX_SETTING_NUMBERS(AOTX_NUMBER_ROW)
};

static const text_row aotx_texts[AOTX_SETTING_TEXT_COUNT] = {
    AOTX_SETTING_TEXTS(AOTX_TEXT_ROW)
};

#undef AOTX_NUMBER_ROW
#undef AOTX_TEXT_ROW

/* ---- small helpers ---- */

/* Reports whether a byte is a space that the reader drops around a key and a value. */
static int is_space(char c)
{
    return (c == ' ' || c == '\t' || c == '\r') ? 1 : 0;
}

/* Drops the spaces at both ends of a range. */
static void trim(const char **at, size_t *len)
{
    const char *start = *at;
    size_t bytes = *len;
    while (bytes > 0 && is_space(start[0])) {
        start++;
        bytes--;
    }
    while (bytes > 0 && is_space(start[bytes - 1])) {
        bytes--;
    }
    *at = start;
    *len = bytes;
}

/* Gives the count of decimals that a scale carries: none for 1 and four for 10000. */
static int decimals_of(int scale)
{
    int count = 0;
    while (scale > 1) {
        scale /= 10;
        count++;
    }
    return count;
}

/* Writes one reason, when the caller asked for one. */
static void say(char *reason, const char *text)
{
    if (reason != NULL) {
        snprintf(reason, AOTX_SETTINGS_REASON_BYTES, "%s", text);
    }
}

/* ---- the table ---- */

void aotx_settings_defaults(aotx_settings *table)
{
    unsigned int i;
    memset(table, 0, sizeof(*table));
    for (i = 0; i < AOTX_SETTING_NUMBER_COUNT; i++) {
        table->number[i] = aotx_numbers[i].value;
    }
    for (i = 0; i < AOTX_SETTING_TEXT_COUNT; i++) {
        snprintf(table->text[i], AOTX_SETTING_TEXT_BYTES, "%s", aotx_texts[i].value);
    }
}

int aotx_settings_find(const char *key, size_t key_length, unsigned int *index)
{
    unsigned int i;
    for (i = 0; i < AOTX_SETTING_NUMBER_COUNT; i++) {
        if (strlen(aotx_numbers[i].name) == key_length &&
            memcmp(aotx_numbers[i].name, key, key_length) == 0) {
            *index = i;
            return 0;
        }
    }
    for (i = 0; i < AOTX_SETTING_TEXT_COUNT; i++) {
        if (strlen(aotx_texts[i].name) == key_length &&
            memcmp(aotx_texts[i].name, key, key_length) == 0) {
            *index = i;
            return 1;
        }
    }
    return 2;
}

const char *aotx_settings_number_name(unsigned int index)
{
    return (index < AOTX_SETTING_NUMBER_COUNT) ? aotx_numbers[index].name : "";
}

unsigned int aotx_settings_number_side(unsigned int index)
{
    return (index < AOTX_SETTING_NUMBER_COUNT) ? aotx_numbers[index].side : 0u;
}

unsigned int aotx_settings_number_effect(unsigned int index)
{
    return (index < AOTX_SETTING_NUMBER_COUNT) ? aotx_numbers[index].effect : 0u;
}

int64_t aotx_settings_number_least(unsigned int index)
{
    return (index < AOTX_SETTING_NUMBER_COUNT) ? aotx_numbers[index].least : 0;
}

int64_t aotx_settings_number_most(unsigned int index)
{
    return (index < AOTX_SETTING_NUMBER_COUNT) ? aotx_numbers[index].most : 0;
}

int aotx_settings_number_scale(unsigned int index)
{
    return (index < AOTX_SETTING_NUMBER_COUNT) ? aotx_numbers[index].scale
                                               : AOTX_SETTING_SCALE_ONE;
}

const char *aotx_settings_text_name(unsigned int index)
{
    return (index < AOTX_SETTING_TEXT_COUNT) ? aotx_texts[index].name : "";
}

unsigned int aotx_settings_text_side(unsigned int index)
{
    return (index < AOTX_SETTING_TEXT_COUNT) ? aotx_texts[index].side : 0u;
}

/* ---- values as text ---- */

size_t aotx_settings_format(int64_t value, int scale, char *out, size_t out_bytes)
{
    char work[48];
    uint64_t magnitude;
    uint64_t whole;
    uint64_t fraction;
    const char *sign;
    int decimals = decimals_of(scale);
    int used;
    size_t bytes;

    if (out == NULL || out_bytes == 0) {
        return 0;
    }
    /* The magnitude comes from an unsigned type, so the least value of the signed type
     * has a magnitude that the signed type cannot hold. */
    magnitude = (value < 0) ? (uint64_t)(-(value + 1)) + 1u : (uint64_t)value;
    sign = (value < 0) ? "-" : "";
    if (decimals <= 0) {
        used = snprintf(work, sizeof(work), "%s%llu", sign, (unsigned long long)magnitude);
    } else {
        whole = magnitude / (uint64_t)scale;
        fraction = magnitude % (uint64_t)scale;
        if (fraction == 0) {
            used = snprintf(work, sizeof(work), "%s%llu", sign, (unsigned long long)whole);
        } else {
            char digits[16];
            int last = (decimals < 15) ? decimals : 15;
            int d;
            for (d = last - 1; d >= 0; d--) {
                digits[d] = (char)('0' + (int)(fraction % 10u));
                fraction /= 10u;
            }
            /* A value that the scale can hold prints the decimals it needs and no more. */
            while (last > 1 && digits[last - 1] == '0') {
                last--;
            }
            digits[last] = '\0';
            used = snprintf(work, sizeof(work), "%s%llu.%s", sign, (unsigned long long)whole,
                            digits);
        }
    }
    if (used < 0) {
        out[0] = '\0';
        return 0;
    }
    bytes = (size_t)used;
    if (bytes > out_bytes - 1u) {
        bytes = out_bytes - 1u;
    }
    memcpy(out, work, bytes);
    out[bytes] = '\0';
    return bytes;
}

/* Reads a whole number, or a number with at most four decimals, into the scaled unit.
 * Returns 0, or 1 with the reason. */
static int read_number(const char *text, size_t len, int scale, int64_t *out, char *reason)
{
    uint64_t whole = 0;
    uint64_t fraction = 0;
    uint64_t limit = (uint64_t)1 << 62;
    int decimals = decimals_of(scale);
    int places = 0;
    int negative = 0;
    int digits = 0;
    size_t i = 0;

    if (len > 0 && (text[0] == '-' || text[0] == '+')) {
        negative = (text[0] == '-') ? 1 : 0;
        i = 1;
    }
    while (i < len && text[i] >= '0' && text[i] <= '9') {
        whole = whole * 10u + (uint64_t)(text[i] - '0');
        digits++;
        if (whole > limit) {
            say(reason, "the value is too large");
            return 1;
        }
        i++;
    }
    if (digits == 0) {
        say(reason, "the value is not a number");
        return 1;
    }
    if (i < len && text[i] == '.') {
        i++;
        while (i < len && text[i] >= '0' && text[i] <= '9') {
            if (places < 4) {
                fraction = fraction * 10u + (uint64_t)(text[i] - '0');
            } else if (text[i] != '0') {
                say(reason, "the value has more than four decimals");
                return 1;
            }
            places++;
            i++;
        }
        if (places == 0) {
            say(reason, "the value is not a number");
            return 1;
        }
        if (places > 4) {
            say(reason, "the value has more than four decimals");
            return 1;
        }
    }
    if (i != len) {
        say(reason, "the value is not a number");
        return 1;
    }
    /* The decimals of the value go into the decimals of the scale. A key with no decimals
     * takes a decimal of zero only, because a fraction has no place in its unit. */
    while (places < decimals) {
        fraction *= 10u;
        places++;
    }
    while (places > decimals) {
        if (fraction % 10u != 0) {
            say(reason, "the key takes a whole number");
            return 1;
        }
        fraction /= 10u;
        places--;
    }
    if (whole > limit / (uint64_t)scale) {
        say(reason, "the value is too large");
        return 1;
    }
    whole = whole * (uint64_t)scale + fraction;
    if (whole > limit) {
        say(reason, "the value is too large");
        return 1;
    }
    *out = negative ? -(int64_t)whole : (int64_t)whole;
    return 0;
}

/* ---- one key and one value ---- */

/* Applies one key and one value, each as a range of bytes. Returns 0 or 1. */
static int apply(const char *key, size_t key_len, const char *value, size_t value_len,
                 aotx_settings *table, char *reason)
{
    unsigned int index = 0;
    size_t i;
    int kind;

    if (key_len == 0) {
        say(reason, "the line names no key");
        return 1;
    }
    if (key_len >= AOTX_SETTING_KEY_BYTES) {
        say(reason, "the key is longer than 63 bytes");
        return 1;
    }
    for (i = 0; i < key_len; i++) {
        char c = key[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_')) {
            say(reason, "the key takes lower case letters, digits, dots and underscores only");
            return 1;
        }
    }
    kind = aotx_settings_find(key, key_len, &index);
    if (kind == 2) {
        say(reason, "the key is not known");
        return 1;
    }
    if (kind == 1) {
        if (value_len >= AOTX_SETTING_TEXT_BYTES) {
            char text[64];
            snprintf(text, sizeof(text), "the text is longer than %u bytes",
                     AOTX_SETTING_TEXT_BYTES - 1u);
            say(reason, text);
            return 1;
        }
        memcpy(table->text[index], value, value_len);
        table->text[index][value_len] = '\0';
        table->text_given[index] = 1u;
        return 0;
    }
    {
        const number_row *row = &aotx_numbers[index];
        int64_t number = 0;
        if (read_number(value, value_len, row->scale, &number, reason) != 0) {
            return 1;
        }
        if (number < row->least || number > row->most) {
            char got[24];
            char low[24];
            char high[24];
            char text[AOTX_SETTINGS_REASON_BYTES];
            aotx_settings_format(number, row->scale, got, sizeof(got));
            aotx_settings_format(row->least, row->scale, low, sizeof(low));
            aotx_settings_format(row->most, row->scale, high, sizeof(high));
            snprintf(text, sizeof(text), "the value %s is not in the range %s to %s", got, low,
                     high);
            say(reason, text);
            return 1;
        }
        table->number[index] = number;
        table->number_given[index] = 1u;
    }
    return 0;
}

int aotx_settings_set(const char *key, const char *value, aotx_settings *table, char *reason)
{
    const char *key_at = key;
    const char *value_at = value;
    size_t key_len = strlen(key);
    size_t value_len = strlen(value);
    trim(&key_at, &key_len);
    trim(&value_at, &value_len);
    return apply(key_at, key_len, value_at, value_len, table, reason);
}

int aotx_settings_line(const char *line, size_t length, aotx_settings *table, char *reason)
{
    const char *at = line;
    const char *key;
    const char *value;
    const char *mark;
    size_t len = length;
    size_t key_len;
    size_t value_len;

    trim(&at, &len);
    if (len == 0 || at[0] == '#') {
        /* A blank line and a comment line set no key. A number sign that is not the first
         * character is part of the value, because a text value runs to the end of the line. */
        return 0;
    }
    mark = (const char *)memchr(at, '=', len);
    if (mark == NULL) {
        say(reason, "the line has no equals sign");
        return 1;
    }
    key = at;
    key_len = (size_t)(mark - at);
    value = mark + 1;
    value_len = len - key_len - 1u;
    trim(&key, &key_len);
    trim(&value, &value_len);
    return apply(key, key_len, value, value_len, table, reason);
}

/* ---- the file ---- */

/* Keeps one refused line. Every refused line is counted; the first ones keep the reason. */
static void refuse(aotx_settings *table, unsigned int line, const char *reason)
{
    if (table->refused_count < AOTX_SETTINGS_REFUSALS) {
        table->refused[table->refused_count].line = line;
        snprintf(table->refused[table->refused_count].reason, AOTX_SETTINGS_REASON_BYTES, "%s",
                 reason);
    }
    table->refused_count++;
}

/* Reports the error of a file that the reader cannot read at all. */
static int no_file(aotx_settings *table, const char *reason)
{
    table->refused_count = 0;
    refuse(table, 0u, reason);
    return 2;
}

int aotx_settings_read(const char *path, aotx_settings *table)
{
    char block[AOTX_SETTINGS_READ_BYTES];
    char line[AOTX_SETTINGS_LINE_BYTES];
    struct stat info;
    unsigned int number = 1u;
    size_t fill = 0;
    int overflow = 0;
    int fd;

    aotx_settings_defaults(table);
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (errno == ENOENT) {
            /* A file that is not there gives every default, which is a complete result. */
            return 0;
        }
        return no_file(table, "the file does not open");
    }
    if (fstat(fd, &info) != 0 || S_ISDIR(info.st_mode)) {
        close(fd);
        return no_file(table, "the path names a directory");
    }
    for (;;) {
        ssize_t got = read(fd, block, sizeof(block));
        ssize_t i;
        if (got < 0) {
            close(fd);
            return no_file(table, "the file does not read");
        }
        if (got == 0) {
            break;
        }
        for (i = 0; i < got; i++) {
            if (block[i] != '\n') {
                if (fill < sizeof(line)) {
                    line[fill++] = block[i];
                } else {
                    overflow = 1;
                }
                continue;
            }
            if (overflow) {
                char text[64];
                snprintf(text, sizeof(text), "the line is longer than %u bytes",
                         AOTX_SETTINGS_LINE_BYTES);
                refuse(table, number, text);
            } else {
                char reason[AOTX_SETTINGS_REASON_BYTES];
                reason[0] = '\0';
                if (aotx_settings_line(line, fill, table, reason) != 0) {
                    refuse(table, number, reason);
                }
            }
            fill = 0;
            overflow = 0;
            number++;
        }
    }
    close(fd);
    if (fill > 0 || overflow) {
        /* The last line of a file that has no end byte is a line all the same. */
        char reason[AOTX_SETTINGS_REASON_BYTES];
        reason[0] = '\0';
        if (overflow) {
            snprintf(reason, sizeof(reason), "the line is longer than %u bytes",
                     AOTX_SETTINGS_LINE_BYTES);
            refuse(table, number, reason);
        } else if (aotx_settings_line(line, fill, table, reason) != 0) {
            refuse(table, number, reason);
        }
    }
    return (table->refused_count > 0) ? 1 : 0;
}

/* ---- one key written back to the file ---- */

/* Reports whether the line names the key. A comment line and a line with no equals sign
 * name no key. */
static int names_key(const char *line, size_t len, const char *key)
{
    const char *at = line;
    const char *mark;
    size_t bytes = len;
    size_t key_len;
    trim(&at, &bytes);
    if (bytes == 0 || at[0] == '#') {
        return 0;
    }
    mark = (const char *)memchr(at, '=', bytes);
    if (mark == NULL) {
        return 0;
    }
    key_len = (size_t)(mark - at);
    trim(&at, &key_len);
    return (strlen(key) == key_len && memcmp(at, key, key_len) == 0) ? 1 : 0;
}

/* Checks the key and the value of a write. A write puts in the file only what the reader
 * accepts. Returns 0 or 1. */
static int check_pair(const char *key, const char *value, char *reason)
{
    unsigned int index = 0;
    int64_t number = 0;
    int kind = aotx_settings_find(key, strlen(key), &index);
    if (kind == 2) {
        say(reason, "the key is not known");
        return 1;
    }
    if (strchr(value, '\n') != NULL) {
        say(reason, "the value holds an end of line byte");
        return 1;
    }
    if (kind == 1) {
        if (strlen(value) >= AOTX_SETTING_TEXT_BYTES) {
            char text[64];
            snprintf(text, sizeof(text), "the text is longer than %u bytes",
                     AOTX_SETTING_TEXT_BYTES - 1u);
            say(reason, text);
            return 1;
        }
        return 0;
    }
    if (read_number(value, strlen(value), aotx_numbers[index].scale, &number, reason) != 0) {
        return 1;
    }
    if (number < aotx_numbers[index].least || number > aotx_numbers[index].most) {
        char got[24];
        char low[24];
        char high[24];
        char text[AOTX_SETTINGS_REASON_BYTES];
        aotx_settings_format(number, aotx_numbers[index].scale, got, sizeof(got));
        aotx_settings_format(aotx_numbers[index].least, aotx_numbers[index].scale, low,
                             sizeof(low));
        aotx_settings_format(aotx_numbers[index].most, aotx_numbers[index].scale, high,
                             sizeof(high));
        snprintf(text, sizeof(text), "the value %s is not in the range %s to %s", got, low,
                 high);
        say(reason, text);
        return 1;
    }
    return 0;
}

/* Writes every byte to the file, or returns -1. */
static int put(int fd, const char *data, size_t bytes)
{
    size_t done = 0;
    while (done < bytes) {
        ssize_t wrote = write(fd, data + done, bytes - done);
        if (wrote <= 0) {
            return -1;
        }
        done += (size_t)wrote;
    }
    return 0;
}

/* Copies the file into the new one and puts the new line in the place of each line that
 * names the key. Returns the count of lines replaced, or -1. */
static int copy_file(int source, int out, const char *key, const char *line, size_t line_bytes,
                     char *last)
{
    char block[AOTX_SETTINGS_READ_BYTES];
    char held[AOTX_SETTINGS_LINE_BYTES];
    size_t fill = 0;
    int overflow = 0;
    int replaced = 0;
    if (source < 0) {
        return 0;
    }
    for (;;) {
        ssize_t got = read(source, block, sizeof(block));
        ssize_t i;
        if (got < 0) {
            return -1;
        }
        if (got == 0) {
            break;
        }
        for (i = 0; i < got; i++) {
            *last = block[i];
            if (block[i] != '\n') {
                if (fill < sizeof(held)) {
                    held[fill++] = block[i];
                } else {
                    /* A line that is longer than the buffer keeps every byte, because the
                     * write puts back the bytes that it did not hold. */
                    overflow = 1;
                    if (put(out, held, fill) != 0 || put(out, &block[i], 1) != 0) {
                        return -1;
                    }
                    fill = 0;
                }
                continue;
            }
            if (!overflow && names_key(held, fill, key)) {
                if (put(out, line, line_bytes) != 0) {
                    return -1;
                }
                replaced++;
            } else if (put(out, held, fill) != 0 || put(out, "\n", 1) != 0) {
                return -1;
            }
            fill = 0;
            overflow = 0;
        }
    }
    if (fill > 0 || overflow) {
        /* The last line of a file that has no end byte is a line all the same. */
        if (!overflow && names_key(held, fill, key)) {
            if (put(out, line, line_bytes) != 0) {
                return -1;
            }
            replaced++;
            *last = '\n';
        } else if (put(out, held, fill) != 0) {
            return -1;
        }
    }
    return replaced;
}

int aotx_settings_write_key(const char *path, const char *key, const char *value, char *reason)
{
    char temp[AOTX_SETTINGS_LINE_BYTES + 64];
    char line[AOTX_SETTINGS_LINE_BYTES];
    const char *cut;
    size_t line_bytes;
    char last = '\n';
    int source;
    int out;
    int replaced;

    if (check_pair(key, value, reason) != 0) {
        return 1;
    }
    if (strlen(path) + sizeof(AOTX_SETTINGS_TEMP) + 2u > sizeof(temp)) {
        say(reason, "the path of the file is too long");
        return 1;
    }
    line_bytes = (size_t)snprintf(line, sizeof(line), "%s = %s\n", key, value);
    if (line_bytes >= sizeof(line)) {
        say(reason, "the line of the key and the value is too long");
        return 1;
    }
    /* The new file goes in the directory of the old one. The rename then stays in one
     * file system, and no reader sees a file that is half written. */
    cut = strrchr(path, '/');
    if (cut != NULL) {
        snprintf(temp, sizeof(temp), "%.*s/%s", (int)(cut - path), path, AOTX_SETTINGS_TEMP);
    } else {
        snprintf(temp, sizeof(temp), "%s", AOTX_SETTINGS_TEMP);
    }
    out = mkstemp(temp);
    if (out < 0) {
        say(reason, "the temporary file does not open");
        return 1;
    }
    if (fchmod(out, 0644) != 0) {
        close(out);
        unlink(temp);
        say(reason, "the temporary file does not take its mode");
        return 1;
    }
    source = open(path, O_RDONLY);
    if (source < 0 && errno != ENOENT) {
        close(out);
        unlink(temp);
        say(reason, "the file does not open");
        return 1;
    }
    replaced = copy_file(source, out, key, line, line_bytes, &last);
    if (source >= 0) {
        close(source);
    }
    if (replaced < 0) {
        close(out);
        unlink(temp);
        say(reason, "the file does not copy");
        return 1;
    }
    if (replaced == 0) {
        /* A file whose last line has no end byte gets one, so the new line stands alone. */
        if ((last != '\n' && put(out, "\n", 1) != 0) || put(out, line, line_bytes) != 0) {
            close(out);
            unlink(temp);
            say(reason, "the new line does not write");
            return 1;
        }
    }
    if (fsync(out) != 0 || close(out) != 0) {
        unlink(temp);
        say(reason, "the file does not reach the disk");
        return 1;
    }
    if (rename(temp, path) != 0) {
        unlink(temp);
        say(reason, "the file does not take its name");
        return 1;
    }
    return 0;
}
