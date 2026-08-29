/* Purpose: Name the screens, the key bar and the command line that each action sends.
 * Owns: The three tables; the terminal program reads them and writes nothing to them.
 * Threading: Not applicable; constant data with no state.
 * Lifetime: The whole run; the tables are constant. */
#ifndef AOTX_TUI_ACTIONS_H
#define AOTX_TUI_ACTIONS_H

/* One screen of the terminal program. `key` is the key that opens it, or the empty text
 * when a screen opens from another screen only. */
typedef struct aotx_tui_screen {
    const char *name;
    const char *key;
} aotx_tui_screen;

/* One field of the key bar at the foot of the frame. */
typedef struct aotx_tui_keybar {
    const char *key;
    const char *label;
} aotx_tui_keybar;

/* One action of a screen. `line` is the command line that the screen sends to a running
 * system, with the parts the screen fills written between angle brackets. An empty line
 * marks an action that changes the disk and sends nothing. */
typedef struct aotx_tui_action {
    const char *screen;
    const char *key;
    const char *line;
} aotx_tui_action;

/* One row of the Menu screen. A screen name opens that screen. A panel name moves the
 * live picture to that panel and closes the Menu screen. Both are local actions and send
 * no command line. */
typedef struct aotx_tui_menu_row {
    const char *label;
    const char *screen;
    const char *panel;
} aotx_tui_menu_row;

typedef struct aotx_tui_argument {
    const char *word;
} aotx_tui_argument;

static const aotx_tui_screen aotx_tui_screens[] = {
    { "help",     "F1" },
    { "menu",     "F2" },
    { "agents",   "F3" },
    { "bus",      "F4" },
    { "models",   "F5" },
    { "tools",    "F6" },
    { "skills",   "F7" },
    { "settings", "F8" },
    { "system",   "F9" },
    { "quit",     "F10" },
    { "session",  "F11" },
    { "picker",   "" },
    { NULL, NULL }
};

static const aotx_tui_keybar aotx_tui_keys[] = {
    { "F1",  "Help" },
    { "F2",  "Menu" },
    { "F3",  "Agents" },
    { "F4",  "Bus" },
    { "F5",  "Models" },
    { "F6",  "Tools" },
    { "F7",  "Skills" },
    { "F8",  "Settings" },
    { "F9",  "System" },
    { "F10", "Quit" },
    { "F11", "Session" },
    { NULL, NULL }
};

static const aotx_tui_menu_row aotx_tui_menu[] = {
    { "console",  "",         "console" },
    { "agents",   "agents",   "" },
    { "bus",      "bus",      "" },
    { "models",   "models",   "" },
    { "tools",    "tools",    "" },
    { "settings", "settings", "" },
    { "session",  "session",  "" },
    { NULL, NULL, NULL }
};

/* The Bus screen shows all messages in its first row. These are the kinds that the other
 * rows can send as an argument. The parity gate holds this table against the parser. */
static const aotx_tui_argument aotx_tui_bus_kinds[] = {
    { "finding" },
    { "rank" },
    { "question" },
    { "answer" },
    { "handoff" },
    { "cost" },
    { "note" },
    { NULL }
};

static const aotx_tui_action aotx_tui_actions[] = {
    { "help",     "Enter", "help" },
    { "menu",     "Enter", "" },
    { "agents",   "Enter", "agents" },
    { "agents",   "y",     "authorize <id>" },
    { "agents",   "n",     "refuse <id>" },
    { "bus",      "Enter", "bus <kind>" },
    { "models",   "Enter", "" },
    { "tools",    "Enter", "import <path>" },
    { "tools",    "m",     "module <name>" },
    { "tools",    "x",     "remove <name>" },
    { "tools",    "p",     "" },
    { "skills",   "Enter", "import <path>" },
    { "skills",   "m",     "module <name>" },
    { "skills",   "x",     "remove <name>" },
    { "skills",   "p",     "" },
    { "settings", "Enter", "set <key> <value>" },
    { "settings", "s",     "settings" },
    { "system",   "Enter", "" },
    { "system",   "r",     "" },
    { "system",   "x",     "quit" },
    { "quit",     "Enter", "" },
    { "session",  "Ctrl-Enter", "say <text>" },
    { "session",  "Alt-Enter",  "task <agent> <text>" },
    { "session",  "y",          "authorize <id>" },
    { "session",  "n",          "refuse <id>" },
    { "session",  "s",          "spawn <role>" },
    { "session",  "p",          "agent <id> pages <n|auto>" },
    { "session",  "c",          "agent <id> compact" },
    { "picker",   "Enter", "import <path>" },
    { NULL, NULL, NULL }
};

#endif
