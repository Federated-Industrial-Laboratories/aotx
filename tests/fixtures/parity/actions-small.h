/* Purpose: Give the parity gate one action table with no difference of its own.
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

static const aotx_tui_screen aotx_tui_screens[] = {
    { "help", "F1" },
    { NULL, NULL }
};

static const aotx_tui_keybar aotx_tui_keys[] = {
    { "F1", "Help" },
    { NULL, NULL }
};

static const aotx_tui_action aotx_tui_actions[] = {
    { "help", "Enter", "help" },
    { NULL, NULL, NULL }
};

#endif
