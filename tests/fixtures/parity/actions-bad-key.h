/* Purpose: Prove the parity gate refuses a key of the key bar that no screen takes.
 * Owns: Nothing; a fixture of the parity gate, read by no program of the system.
 * Threading: Not applicable; constant data.
 * Lifetime: The check that runs the gate over it. */
#ifndef AOTX_TUI_ACTIONS_H
#define AOTX_TUI_ACTIONS_H

typedef struct aotx_tui_screen {
    const char *name;
    const char *key;
} aotx_tui_screen;

typedef struct aotx_tui_keybar {
    const char *key;
    const char *label;
} aotx_tui_keybar;

typedef struct aotx_tui_action {
    const char *screen;
    const char *key;
    const char *line;
} aotx_tui_action;

typedef struct aotx_tui_menu_row {
    const char *label;
    const char *screen;
    const char *panel;
} aotx_tui_menu_row;

typedef struct aotx_tui_argument {
    const char *word;
} aotx_tui_argument;

static const aotx_tui_screen aotx_tui_screens[] = {
    { "help",   "F1" },
    { "agents", "F3" },
    { NULL, NULL }
};

/* The last row names a key that no screen of the table above takes. */
static const aotx_tui_keybar aotx_tui_keys[] = {
    { "F1", "Help" },
    { "F3", "Agents" },
    { "F5", "Models" },
    { NULL, NULL }
};

static const aotx_tui_menu_row aotx_tui_menu[] = {
    { "console", "", "console" },
    { NULL, NULL, NULL }
};

static const aotx_tui_argument aotx_tui_bus_kinds[] = {
    { "finding" }, { "rank" }, { "question" }, { "answer" },
    { "handoff" }, { "cost" }, { "note" }, { NULL }
};

static const aotx_tui_action aotx_tui_actions[] = {
    { "help",   "Enter", "help" },
    { "agents", "y",     "authorize <id>" },
    { NULL, NULL, NULL }
};

#endif
