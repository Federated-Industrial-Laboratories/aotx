/* Purpose: Prove the parity gate refuses a Bus word that the parser does not take.
 * Owns: Nothing; a fixture of the parity gate, read by no program of the system.
 * Threading: Not applicable; constant data.
 * Lifetime: The check that runs the gate over it. */
#ifndef AOTX_TUI_ACTIONS_H
#define AOTX_TUI_ACTIONS_H

typedef struct pair_row {
    const char *one;
    const char *two;
} pair_row;

typedef struct triple_row {
    const char *one;
    const char *two;
    const char *three;
} triple_row;

typedef struct word_row {
    const char *word;
} word_row;

static const pair_row aotx_tui_screens[] = {
    { "help", "F1" },
    { NULL, NULL }
};

static const pair_row aotx_tui_keys[] = {
    { "F1", "Help" },
    { NULL, NULL }
};

static const triple_row aotx_tui_menu[] = {
    { "console", "", "console" },
    { NULL, NULL, NULL }
};

static const word_row aotx_tui_bus_kinds[] = {
    { "finding" },
    { "rank" },
    { "question" },
    { "answer" },
    { "handoff" },
    { "cost" },
    { "note" },
    { "unknown" },
    { NULL }
};

static const triple_row aotx_tui_actions[] = {
    { "help", "Enter", "help" },
    { NULL, NULL, NULL }
};

#endif
