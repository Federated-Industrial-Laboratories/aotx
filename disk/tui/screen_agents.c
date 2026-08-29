/* Purpose: Fill the rows of the Help screen and of the Agents screen.
 * Owns: Nothing; the caller gives the rows and owns them.
 * Threading: One thread.
 * Lifetime: One frame. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "disk/tui/tui.h"

#include <stdio.h>
#include <string.h>

/* The state of an agent, as the panels name it. */
static const char *agent_state(unsigned int state)
{
    switch (state) {
    case 0u:  return "idle";
    case 1u:  return "ready";
    case 2u:  return "thinks";
    case 3u:  return "waits";
    case 4u:  return "ends";
    default:  return "-";
    }
}

unsigned int aotx_rows_help(aotx_tui *tui, char *out, unsigned int rows, unsigned int cols)
{
    unsigned int count = 0;
    unsigned int i;
    (void)tui;
    if (rows == 0) {
        return 0;
    }
    snprintf(out, cols, "keys of the frame");
    count = 1;
    for (i = 0; aotx_tui_keys[i].key != NULL && count < rows; i++) {
        snprintf(out + (size_t)count * cols, cols, "  %-4s opens %s",
                 aotx_tui_keys[i].key, aotx_tui_keys[i].label);
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "  Esc  leaves a screen, or opens Menu");
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "  Tab  moves the focus in the picture");
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols,
                 "  Alt with an arrow pans a small terminal; Ctrl L draws the frame again");
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "lines that a screen sends");
        count++;
    }
    for (i = 0; aotx_tui_actions[i].screen != NULL && count < rows; i++) {
        if (aotx_tui_actions[i].line[0] == '\0') {
            continue;
        }
        snprintf(out + (size_t)count * cols, cols, "  %-8s %-5s %s",
                 aotx_tui_actions[i].screen, aotx_tui_actions[i].key,
                 aotx_tui_actions[i].line);
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols,
                 "the command help at the console shows every command of the system");
        count++;
    }
    return count;
}

unsigned int aotx_rows_agents(aotx_tui *tui, char *out, unsigned int rows,
                              unsigned int cols)
{
    unsigned int count = 0;
    unsigned int i;
    if (rows == 0) {
        return 0;
    }
    if (tui->have_shot == 0) {
        snprintf(out, cols, "no system runs; F9 starts one");
        return 1;
    }
    /* The requests that wait come first, so the cursor falls on one and the keys y and n
     * answer it. */
    for (i = 0; i < AOTX_MIRROR_REQUEST_ROWS && count < rows; i++) {
        const aotx_mirror_request_row *row = &tui->shot.tables.request[i];
        if (row->request == 0u) {
            continue;
        }
        snprintf(out + (size_t)count * cols, cols,
                 "%u waits, agent %u, tool %.15s, %.40s", row->request, row->agent,
                 row->tool_name, row->argument);
        count++;
    }
    if (count == 0 && count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- no request waits");
        count++;
    }
    for (i = 0; i < AOTX_MIRROR_AGENT_ROWS && count < rows; i++) {
        const aotx_mirror_agent_row *row = &tui->shot.tables.agent[i];
        if (row->role_name[0] == '\0') {
            continue;
        }
        snprintf(out + (size_t)count * cols, cols,
                 "- agent %u, role %.15s, %s, task %u, turn %u", row->id, row->role_name,
                 agent_state(row->state), row->task, row->turn);
        count++;
    }
    if (count < rows) {
        snprintf(out + (size_t)count * cols, cols, "- agents live %u of %u",
                 tui->shot.head.agents_live, tui->shot.head.slots);
        count++;
    }
    return count;
}
