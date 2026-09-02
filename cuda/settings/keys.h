/* Purpose: Name every setting with its side, its effect, its default and its range.
 * Owns: Nothing; constants only, included by both sides.
 * Launch shape: Not applicable; plain C with no CUDA symbol.
 * Lifetime: The layout of the settings table; a change here changes both sides. */
#ifndef AOTX_SETTINGS_KEYS_H
#define AOTX_SETTINGS_KEYS_H

/* A key is at most AOTX_SETTING_KEY_BYTES - 1 bytes of lower case letters, digits, dots and
 * underscores. A text value is at most AOTX_SETTING_TEXT_BYTES - 1 bytes. */
#define AOTX_SETTING_KEY_BYTES   64u
#define AOTX_SETTING_TEXT_BYTES  256u

/* A number with four decimals travels as the value times AOTX_SETTING_SCALE_FIXED. A whole
 * number travels with a scale of 1. */
#define AOTX_SETTING_SCALE_ONE   1
#define AOTX_SETTING_SCALE_FIXED 10000

/* The side that reads a setting. A boot setting is read from the file by the boot glue and
 * makes no record. A device setting is a SETTING record the device applies. A terminal
 * setting is read by the terminal program and makes no record. */
#define AOTX_SETTING_SIDE_BOOT     1u
#define AOTX_SETTING_SIDE_DEVICE   2u
#define AOTX_SETTING_SIDE_TERMINAL 3u

/* When a change takes effect. */
#define AOTX_SETTING_AT_BOOT     1u  /* the next boot */
#define AOTX_SETTING_AT_TICK     2u  /* the next tick */
#define AOTX_SETTING_AT_SEQUENCE 3u  /* the next sequence that opens */
#define AOTX_SETTING_AT_TASK     4u  /* the next task that opens */
#define AOTX_SETTING_AT_REQUEST  5u  /* the next tool request */
#define AOTX_SETTING_AT_FRAME    6u  /* the next frame */
#define AOTX_SETTING_AT_READ     7u  /* when the terminal program reads the file */

/* The number settings. X(symbol, name, side, effect, default, least, most, scale).
 * The default, the least and the most are in the scaled unit. The order here is the
 * order of the device table and of every list a program prints. */
#ifdef AOTX_AFFECT
#define AOTX_SETTING_AFFECT_NUMBERS(X) \
    X(AOTX_SET_AFFECT_ON,       "affect.on",              DEVICE, SEQUENCE, 0,      0, 1,     AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_QUALITY_ON,      "quality.on",             DEVICE, SEQUENCE, 0,      0, 1,     AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_AFFECT_PROBE_GAIN, "affect.probe_gain",    DEVICE, SEQUENCE, 0,      0, 10000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_DECAY_FAST, "affect.decay_fast",    DEVICE, SEQUENCE, 5000,   0, 9900,  AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_DECAY_SLOW, "affect.decay_slow",    DEVICE, SEQUENCE, 9000,   0, 9900,  AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_GAIN_FAST, "affect.gain_fast",      DEVICE, SEQUENCE, 5000,   0, 20000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_GAIN_SLOW, "affect.gain_slow",      DEVICE, SEQUENCE, 1000,   0, 20000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_CAP_VALENCE, "affect.cap_valence",  DEVICE, SEQUENCE, 10000,  0, 10000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_CAP_AROUSAL, "affect.cap_arousal",  DEVICE, SEQUENCE, 10000,  0, 10000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_TEMPERATURE_GAIN, "affect.temperature_gain", DEVICE, SEQUENCE, 0, -10000, 10000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_VOICE_GAIN, "affect.voice_gain",    DEVICE, SEQUENCE, 0, -10000, 10000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_STEER_GAIN, "affect.steer_gain",    DEVICE, SEQUENCE, 0,      0, 10000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_AFFECT_BUDGET,   "affect.budget",          DEVICE, SEQUENCE, 2500,   0, 40000, AOTX_SETTING_SCALE_FIXED)
#else
#define AOTX_SETTING_AFFECT_NUMBERS(X)
#endif

#define AOTX_SETTING_NUMBERS(X) \
    X(AOTX_SET_WINDOW_ON,        "window.on",             BOOT,   BOOT,     0,     0,  1,       AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_TUI_ON,           "tui.on",                BOOT,   BOOT,     0,     0,  1,       AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_TUI_ESCAPE_MS,    "tui.escape_ms",         TERMINAL, READ,   25,    5,  500,     AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_TICK_PERIOD_MS,   "tick.period_ms",        DEVICE, TICK,     10,    1,  1000,    AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_DECODE_BUDGET_MS, "decode.budget_ms",      DEVICE, TICK,     120,   10, 10000,   AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_PREFILL_TOKENS,   "decode.prefill_tokens", DEVICE, TICK,     512,   32, 512,     AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_REPLY_LIMIT,      "decode.reply_limit",    DEVICE, SEQUENCE, 256,   1,  8191,    AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_AUTO_CONTINUE,    "decode.auto_continue",  DEVICE, TICK,     0,     0,  1,       AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_TEMPERATURE,      "sample.temperature",    DEVICE, SEQUENCE, 0,     0,  20000,   AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_TOP_P,            "sample.top_p",          DEVICE, SEQUENCE, 10000, 1,  10000,   AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_TOP_K,            "sample.top_k",          DEVICE, SEQUENCE, 0,     0,  256,     AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_MIN_P,            "sample.min_p",          DEVICE, SEQUENCE, 0,     0,  10000,   AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_REPEAT_PENALTY,   "sample.repeat_penalty", DEVICE, SEQUENCE, 10000, 1,  20000,   AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_REPEAT_WINDOW,    "sample.repeat_window",  DEVICE, SEQUENCE, 0,     0,  8191,    AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_PRESENCE_PENALTY, "sample.presence_penalty", DEVICE, SEQUENCE, 0, -20000, 20000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_FREQUENCY_PENALTY,"sample.frequency_penalty",DEVICE,SEQUENCE, 0, -20000, 20000, AOTX_SETTING_SCALE_FIXED) \
    X(AOTX_SET_SAMPLE_SEED,      "sample.seed",           DEVICE, SEQUENCE, 0,     0,  2147483647, AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_THINK_LIMIT,      "decode.think_limit",    DEVICE, SEQUENCE, -1,   -1,  8191,    AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_AGENT_BUDGET,     "agent.budget",          DEVICE, TASK,     8,     1,  64,      AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_TOOL_DEADLINE,    "tool.deadline_ticks",   DEVICE, REQUEST,  500,   1,  1000000, AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_MIRROR_HZ,        "mirror.hz",             DEVICE, FRAME,    30,    1,  120,     AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_AGENT_PAGES,      "agent.pages",           DEVICE, TASK,     0,     0,  4096,    AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_RECALL_K,         "agent.recall_k",        DEVICE, TASK,     4,     0,  16,      AOTX_SETTING_SCALE_ONE) \
    X(AOTX_SET_COMPACT_AT,       "agent.compact_at",      DEVICE, TASK,     128,   8,  1024,    AOTX_SETTING_SCALE_ONE) \
    AOTX_SETTING_AFFECT_NUMBERS(X)

/* The text settings. X(symbol, name, side, effect, default). An empty default means the
 * program's own default applies. */
#define AOTX_SETTING_TEXTS(X) \
    X(AOTX_SET_JOURNAL_DIR,  "journal.dir",  BOOT,     BOOT, "journal") \
    X(AOTX_SET_MODELS_DIR,   "models.dir",   BOOT,     BOOT, "models") \
    X(AOTX_SET_MODELS_ROLES, "models.roles", BOOT,     BOOT, "") \
    X(AOTX_SET_MODULES_DIR,  "modules.dir",  BOOT,     BOOT, "modules") \
    X(AOTX_SET_TOOLS_ROOT,   "tools.root",   BOOT,     BOOT, "") \
    X(AOTX_SET_DERIVE_LIST,  "derive.list",  BOOT,     BOOT, "") \
    X(AOTX_SET_TUI_COLOR,    "tui.color",    TERMINAL, READ, "none") \
    X(AOTX_SET_TUI_BOX,      "tui.box",      TERMINAL, READ, "ascii") \
    X(AOTX_SET_TUI_SPLASH,   "tui.splash",   TERMINAL, READ, "auto")

/* The index of each setting in its list, in list order. */
#define AOTX_SETTING_INDEX(symbol, name, side, effect, ...) symbol,
enum aotx_setting_number { AOTX_SETTING_NUMBERS(AOTX_SETTING_INDEX) AOTX_SETTING_NUMBER_COUNT };
enum aotx_setting_text { AOTX_SETTING_TEXTS(AOTX_SETTING_INDEX) AOTX_SETTING_TEXT_COUNT };
#undef AOTX_SETTING_INDEX

#endif
