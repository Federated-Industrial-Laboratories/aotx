/* Purpose: Fill the rows of the Settings screen from the file or from a running system.
 * Owns: Nothing; the caller gives the rows and owns them.
 * Threading: One thread.
 * Lifetime: One frame. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <stdio.h>
#include <string.h>

/* When a change of a key takes effect. */
static const char *effect_name(unsigned int effect)
{
    switch (effect) {
    case AOTX_SETTING_AT_BOOT:     return "at the next boot";
    case AOTX_SETTING_AT_TICK:     return "at the next tick";
    case AOTX_SETTING_AT_SEQUENCE: return "at the next sequence";
    case AOTX_SETTING_AT_TASK:     return "at the next task";
    case AOTX_SETTING_AT_REQUEST:  return "at the next request";
    case AOTX_SETTING_AT_FRAME:    return "at the next frame";
    case AOTX_SETTING_AT_READ:     return "when the terminal reads the file";
    default:                       return "-";
    }
}

/* The value that a running system holds for a key, or a null pointer. */
static const aotx_mirror_setting_row *live_row(const aotx_tui *tui, const char *key)
{
    unsigned int i;
    if (tui->have_shot == 0) {
        return NULL;
    }
    for (i = 0; i < AOTX_MIRROR_SETTING_ROWS; i++) {
        const aotx_mirror_setting_row *row = &tui->shot.tables.setting[i];
        if (row->key[0] != '\0' && strncmp(row->key, key, AOTX_MIRROR_TEXT_BYTES) == 0) {
            return row;
        }
    }
    return NULL;
}

unsigned int aotx_rows_settings(aotx_tui *tui, char *out, unsigned int rows,
                                unsigned int cols)
{
    unsigned int count = 0;
    unsigned int i;
    for (i = 0; i < AOTX_SETTING_NUMBER_COUNT && count < rows; i++) {
        const char *key = aotx_settings_number_name(i);
        const aotx_mirror_setting_row *live = live_row(tui, key);
        char value[32];
        char least[32];
        char most[32];
        int scale = aotx_settings_number_scale(i);
        aotx_settings_format(live != NULL ? live->value : tui->settings.number[i], scale,
                             value, sizeof(value));
        aotx_settings_format(aotx_settings_number_least(i), scale, least, sizeof(least));
        aotx_settings_format(aotx_settings_number_most(i), scale, most, sizeof(most));
        snprintf(out + (size_t)count * cols, cols, "%-22s %-10s %s to %s, %s%s", key,
                 value, least, most, effect_name(aotx_settings_number_effect(i)),
                 (live != NULL) ? ", live" : "");
        count++;
    }
    for (i = 0; i < AOTX_SETTING_TEXT_COUNT && count < rows; i++) {
        const char *key = aotx_settings_text_name(i);
        unsigned int effect = (aotx_settings_text_side(i) == AOTX_SETTING_SIDE_TERMINAL)
                              ? AOTX_SETTING_AT_READ : AOTX_SETTING_AT_BOOT;
        snprintf(out + (size_t)count * cols, cols, "%-22s %-10s text, %s", key,
                 (tui->settings.text[i][0] != '\0') ? tui->settings.text[i] : "-",
                 effect_name(effect));
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- the file is %s", tui->settings_path);
        count++;
    }
    return count;
}
