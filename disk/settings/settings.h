/* Purpose: Declare the reader of the settings file that the boot glue and the feeder share.
 * Owns: Nothing; the caller owns the table it passes.
 * Threading: One thread; no function here holds state between calls.
 * Lifetime: The call. */
#ifndef AOTX_SETTINGS_H
#define AOTX_SETTINGS_H

#include <stddef.h>
#include <stdint.h>

#include "cuda/settings/keys.h"

/* The settings file holds one `key = value` a line. A `#` starts a comment. Blank lines
 * are skipped. Spaces around the key and the value are dropped. A number value is a whole
 * number or a number with at most four decimals. A text value runs to the end of the line. */
#define AOTX_SETTINGS_FILE_DEFAULT "aotx.settings"
#define AOTX_SETTINGS_REFUSALS     32u
#define AOTX_SETTINGS_REASON_BYTES 128u

typedef struct aotx_settings_refusal {
    unsigned int line;                        /* the line of the file, from 1 */
    char         reason[AOTX_SETTINGS_REASON_BYTES];
} aotx_settings_refusal;

/* The table a read fills. Every number holds its default until the file names it; `given`
 * marks the keys the file named, so a caller publishes only those. A number is held in
 * its scaled unit (keys.h). */
typedef struct aotx_settings {
    int64_t      number[AOTX_SETTING_NUMBER_COUNT];
    char         text[AOTX_SETTING_TEXT_COUNT][AOTX_SETTING_TEXT_BYTES];
    uint8_t      number_given[AOTX_SETTING_NUMBER_COUNT];
    uint8_t      text_given[AOTX_SETTING_TEXT_COUNT];
    unsigned int refused_count;               /* lines refused, all of them counted */
    aotx_settings_refusal refused[AOTX_SETTINGS_REFUSALS];  /* the first ones, with reasons */
} aotx_settings;

/* Fill the table with every default. */
void aotx_settings_defaults(aotx_settings *table);

/* Read the file into the table. A file that is not there gives every default and 0. The
 * reader refuses an unknown key, a value outside its range, a malformed line and a text
 * that is too long. A refused line is counted and recorded with its reason. The rest of
 * the file is read, and the result is 1. A file that cannot be read for another reason
 * gives 2 and the reason in refused[0]. */
int aotx_settings_read(const char *path, aotx_settings *table);

/* Apply one line as the file would. Returns 0 when the line set a key or was blank, 1
 * when it was refused (the reason written to `reason`, AOTX_SETTINGS_REASON_BYTES). */
int aotx_settings_line(const char *line, size_t length, aotx_settings *table, char *reason);

/* Apply one key and one value as text, as `set <key> <value>` would. Same results. */
int aotx_settings_set(const char *key, const char *value, aotx_settings *table, char *reason);

/* Find a key. Returns 0 and fills `index` for a number key, 1 for a text key, 2 when the
 * key is unknown. */
int aotx_settings_find(const char *key, size_t key_length, unsigned int *index);

/* The name, the side, the effect, the scale and the range of a number setting. */
const char *aotx_settings_number_name(unsigned int index);
unsigned int aotx_settings_number_side(unsigned int index);
unsigned int aotx_settings_number_effect(unsigned int index);
int64_t aotx_settings_number_least(unsigned int index);
int64_t aotx_settings_number_most(unsigned int index);
int aotx_settings_number_scale(unsigned int index);

/* The name and the side of a text setting. */
const char *aotx_settings_text_name(unsigned int index);
unsigned int aotx_settings_text_side(unsigned int index);

/* Write a scaled value as text: a whole number, or a number with the decimals the scale
 * gives, trailing zeros dropped. Returns the bytes written, without the end byte. */
size_t aotx_settings_format(int64_t value, int scale, char *out, size_t out_bytes);

/* Write the key to the file in place. The line that names it is replaced, or a line is
 * added at the end. The rest of the file is kept byte for byte. The file is written to a
 * temporary name in the same directory and renamed. Returns 0, or 1 with the reason. */
int aotx_settings_write_key(const char *path, const char *key, const char *value, char *reason);

#endif
