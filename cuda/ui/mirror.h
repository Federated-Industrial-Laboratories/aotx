/* Purpose: Define the layout of the mirror, the snapshot of the grid a terminal program reads.
 * Owns: Nothing; layouts and constants only, included by both sides.
 * Launch shape: Not applicable; plain C with no CUDA symbol.
 * Lifetime: The mirror layout version; a change to any struct increments AOTX_MIRROR_LAYOUT. */
#ifndef AOTX_UI_MIRROR_H
#define AOTX_UI_MIRROR_H

#include <stdint.h>

#define AOTX_MIRROR_MAGIC        0x52524D41u   /* "AMRR" in little-endian byte order */
#define AOTX_MIRROR_LAYOUT       7u
#define AOTX_MIRROR_COLS         160u
#define AOTX_MIRROR_ROWS         50u
#define AOTX_MIRROR_CELLS        (AOTX_MIRROR_COLS * AOTX_MIRROR_ROWS)
#define AOTX_MIRROR_SLOTS        2u
#define AOTX_MIRROR_PANELS       6u
#define AOTX_MIRROR_NAME_BYTES   16u
#define AOTX_MIRROR_TABLES_EVERY 10u

/* The seam glue writes the shape. The feeder writes the attached count. The mirror node
 * writes the device ring use. A client reads the two live fields with acquire loads. */
typedef struct aotx_mirror_preamble {
    uint32_t magic;
    uint32_t layout;
    uint32_t slots;             /* AOTX_MIRROR_SLOTS */
    uint32_t slot_bytes;        /* bytes of one snapshot */
    uint32_t cols;
    uint32_t rows;
    uint32_t attached;          /* terminals attached, written by the feeder */
    uint32_t reserved;
    uint64_t device_ring_used;  /* records not copied to the host ring */
    uint64_t device_ring_slots; /* capacity of the device ring */
} aotx_mirror_preamble;

typedef struct aotx_mirror_panel {
    char     name[AOTX_MIRROR_NAME_BYTES];  /* end byte included */
    uint16_t row, col, rows, cols;          /* the rectangle in the grid */
} aotx_mirror_panel;

/* The head of a snapshot. The sequence is the publish field: zero while the device writes
 * the slot, the frame number when the slot is whole. A reader takes a copy when two
 * acquire loads of the sequence agree on a value that is not zero. */
typedef struct aotx_mirror_head {
    uint64_t sequence;          /* the publish field */
    uint64_t tick;
    uint64_t boot_id;
    uint64_t drain_lag_ms;
    uint64_t held;              /* ticks held since boot */
    uint64_t model_mb;          /* model megabytes placed after boot */
    uint64_t reserved_model;    /* zero; keeps the cell grid on a 16-byte boundary */
    uint32_t agents_live;
    uint32_t slots;             /* AOTX_SLOTS of the build */
    uint32_t requests_waiting;
    uint32_t language;          /* the resident language role, or ~0u */
    uint32_t cursor_row, cursor_col;  /* the editor's cursor cell */
    uint32_t focus;             /* 0 the console, 1 the agents panel */
    uint32_t tables_sequence;   /* the frame the tables slot was last written at */
    char     profile[16];       /* end byte included */
    char     arch[8];           /* "sm_86", end byte included */
    aotx_mirror_panel panel[AOTX_MIRROR_PANELS];
} aotx_mirror_head;

/* One cell: the glyph and the attribute the raster uses. */
typedef struct aotx_mirror_cell {
    uint8_t  glyph;
    uint8_t  attribute;
} aotx_mirror_cell;

/* The fixed layout holds the largest profile. A row with an empty name is unused. */
#define AOTX_MIRROR_AGENT_ROWS    256u
#define AOTX_MIRROR_REQUEST_ROWS  16u
#define AOTX_MIRROR_MODEL_ROWS    5u
#define AOTX_MIRROR_MODULE_ROWS   256u
#define AOTX_MIRROR_SETTING_ROWS  48u
#define AOTX_MIRROR_TEXT_BYTES    48u

typedef struct aotx_mirror_agent_row {
    uint32_t id, role, state, task, request, turn, pages;
    char     role_name[AOTX_MIRROR_NAME_BYTES];
} aotx_mirror_agent_row;

typedef struct aotx_mirror_request_row {
    uint32_t request, agent, tool;
    char     tool_name[AOTX_MIRROR_NAME_BYTES];
    char     argument[AOTX_MIRROR_TEXT_BYTES];
} aotx_mirror_request_row;

typedef struct aotx_mirror_model_row {
    uint32_t role, quant, resident;
    char     name[AOTX_MIRROR_TEXT_BYTES];
} aotx_mirror_model_row;

typedef struct aotx_mirror_module_row {
    uint32_t kind, state;
    char     name[AOTX_MIRROR_TEXT_BYTES];
    char     reason[AOTX_MIRROR_TEXT_BYTES];
} aotx_mirror_module_row;

typedef struct aotx_mirror_setting_row {
    int64_t  value;
    uint32_t scale, effect;
    char     key[AOTX_MIRROR_TEXT_BYTES];
} aotx_mirror_setting_row;

typedef struct aotx_mirror_tables {
    aotx_mirror_agent_row   agent[AOTX_MIRROR_AGENT_ROWS];
    aotx_mirror_request_row request[AOTX_MIRROR_REQUEST_ROWS];
    aotx_mirror_model_row   model[AOTX_MIRROR_MODEL_ROWS];
    aotx_mirror_module_row  module[AOTX_MIRROR_MODULE_ROWS];
    aotx_mirror_setting_row setting[AOTX_MIRROR_SETTING_ROWS];
} aotx_mirror_tables;

typedef struct aotx_mirror_snapshot {
    aotx_mirror_head   head;
    aotx_mirror_cell   cell[AOTX_MIRROR_CELLS];
    aotx_mirror_tables tables;
} aotx_mirror_snapshot;

#endif
