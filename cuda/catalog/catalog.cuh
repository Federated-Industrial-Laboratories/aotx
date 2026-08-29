/* Purpose: Hold the catalog of installed modules and take the records that change it.
 * Owns: The catalog table, the arena of texts and the table of imports that arrive.
 * Launch shape: Device functions; the apply step calls them from its serial thread.
 * Lifetime: The whole run.
 *
 * A module is a skill, a role or a tool. The catalog is device state. Every entry comes
 * from a class A record, so a restore rebuilds it from the journal and opens no file. The
 * arena holds every text of every module as a run of (offset, length). The table and the
 * arena are device globals, after the note store of the embedding search. */
#ifndef AOTX_CATALOG_CUH
#define AOTX_CATALOG_CUH

#include "profile/profile.cuh"
#include "seam/wire.h"

/* The state of one entry. A refused entry keeps its name and its reason until the next
 * import of that name, so the modules command shows what went wrong. */
#define AOTX_CATALOG_FREE       0u
#define AOTX_CATALOG_ARRIVING   1u
#define AOTX_CATALOG_INSTALLED  2u
#define AOTX_CATALOG_REFUSED    3u

/* Where a tool runs. A built-in tool is a tool the repository carries in code. */
#define AOTX_CATALOG_SIDE_DEVICE 0u
#define AOTX_CATALOG_SIDE_HOST   1u
#define AOTX_CATALOG_SIDE_BUILT  2u

/* Whether a tool waits for the operator. A role may add the wait to a tool that says
 * never. A role may not take the wait off a tool that says always. */
#define AOTX_CATALOG_AUTH_NEVER  0u
#define AOTX_CATALOG_AUTH_ALWAYS 1u

/* Bytes of a name, which is the bytes the head record carries. */
#define AOTX_CATALOG_NAME_BYTES  AOTX_IMPORT_NAME_BYTES

/* Argument keys of one tool. */
#define AOTX_CATALOG_ARGS        4u

/* Skills a role puts in every one of its prompts, in the order the manifest gives. */
#define AOTX_CATALOG_ROLE_SKILLS 8u

/* The value that stands for no entry of the catalog. */
#define AOTX_CATALOG_NO_ENTRY    AOTX_MODULE_SLOTS

/* Built-in tools the device puts in the catalog before the first tick. */
#define AOTX_CATALOG_BUILT_IN    4u

/* Imports that may arrive at one time. A head that finds no free row is refused. */
#define AOTX_CATALOG_ARRIVING_MAX 8u

/* Free runs of the arena. The list is kept in the order of the offsets. A free run beside
 * another one joins it, so the count does not grow without a bound. */
#define AOTX_CATALOG_FREE_RUNS   64u

/* Words of one mask over the entries of the catalog. */
#define AOTX_CATALOG_MASK_WORDS  ((AOTX_MODULE_SLOTS + 31u) / 32u)

/* Bytes of the head of a skill file: the two keys between the two lines of three dashes.
 * A skill directory that holds the skill file alone sends the head and the body as one
 * file. The head then takes this much beside the bound of a body. */
#define AOTX_CATALOG_HEAD_BYTES  512u

/* Bytes of the overlay of one role: the duty sentence the prompt of that role starts
 * with. An import of a role whose overlay is longer is refused with this figure. */
#define AOTX_CATALOG_OVERLAY_BYTES 1536u

/* Bytes of the tool list and the skill list that one prompt takes. A role that allows
 * more tools than fit gets the first that fit, in catalog order. The count of the cut
 * stands in the catalog counters. The figure bounds a run inside the prompt table and
 * sizes no table. It therefore stands here and not in a profile header. */
#define AOTX_CATALOG_LIST_BYTES  2048u

/* Why an import was refused. The console line and the bus note name the reason, and the
 * figure beside it is the bound the module went past, or zero. */
#define AOTX_CATALOG_WHY_NONE    0u
#define AOTX_CATALOG_WHY_NAME    1u   /* the name is not 1 to 63 bytes of a-z, 0-9 and _ */
#define AOTX_CATALOG_WHY_HEAD    2u   /* the name of the manifest is not the name of the head */
#define AOTX_CATALOG_WHY_KIND    3u   /* the kind of the manifest is not the kind of the head */
#define AOTX_CATALOG_WHY_KEY     4u   /* the manifest holds a key this kind does not take */
#define AOTX_CATALOG_WHY_SHAPE   5u   /* a line of the manifest is not key: value */
#define AOTX_CATALOG_WHY_BODY    6u  /* the body is longer than AOTX_SKILL_BYTES */
#define AOTX_CATALOG_WHY_ARGS    7u   /* the tool names more than AOTX_CATALOG_ARGS keys */
#define AOTX_CATALOG_WHY_ARENA   8u   /* the arena cannot hold the text */
#define AOTX_CATALOG_WHY_TABLE   9u   /* the table holds no free entry */
#define AOTX_CATALOG_WHY_TWICE   10u  /* a second head reached an import that arrives */
#define AOTX_CATALOG_WHY_PART    11u  /* a part names a file or a run the head does not hold */
#define AOTX_CATALOG_WHY_MISSING 12u  /* the manifest gives no kind, or no name */
#define AOTX_CATALOG_WHY_VALUE   13u  /* a value is not one the key takes */
#define AOTX_CATALOG_WHY_SKILLS  14u  /* the role names more than AOTX_CATALOG_ROLE_SKILLS */
#define AOTX_CATALOG_WHY_EMPTY   15u  /* the kind needs a body and the import carries none */
#define AOTX_CATALOG_WHY_BUSY    16u  /* no free row for one more import that arrives */
#define AOTX_CATALOG_WHY_FILES   17u  /* the head counts files that its byte counts deny */

/* Why a remove was refused. */
#define AOTX_CATALOG_GONE_NONE   0u
#define AOTX_CATALOG_GONE_UNKNOWN 1u  /* no entry holds that name */
#define AOTX_CATALOG_GONE_ROLE   2u   /* an agent runs on that role */
#define AOTX_CATALOG_GONE_TOOL   3u   /* a request of that tool is in flight */
#define AOTX_CATALOG_GONE_BUILT  4u   /* the entry is a built-in tool */
#define AOTX_CATALOG_GONE_ARRIVING 5u /* an import of that name arrives */

/* One run of the arena: the offset of the first byte and the byte count. */
typedef struct aotx_catalog_run {
    unsigned int at;
    unsigned int length;
} aotx_catalog_run;

/* The row of a tool. The keys are runs of the arena, because they stand in the manifest
 * text the arena holds. A built-in tool names the number the tool module knows it by. */
typedef struct aotx_catalog_tool {
    unsigned int     side;        /* AOTX_CATALOG_SIDE_* */
    unsigned int     authorize;   /* AOTX_CATALOG_AUTH_* */
    unsigned int     deadline;    /* ticks a reply may take; zero takes the setting */
    unsigned int     timeout;     /* seconds the feeder lets the program run */
    unsigned int     arguments;   /* argument keys, up to AOTX_CATALOG_ARGS */
    unsigned int     built_in;    /* AOTX_TOOL_* of a built-in tool, or zero */
    aotx_catalog_run key[AOTX_CATALOG_ARGS];
    aotx_catalog_run entry;       /* the kernel symbol of a device tool */
} aotx_catalog_tool;

/* The row of a role. The masks stand over the entries of the catalog. The skill list keeps
 * the order the manifest gave, which a mask cannot. The mask beside it says whether a role
 * holds a skill in one read. */
typedef struct aotx_catalog_role {
    unsigned int     model;       /* AOTX_MODEL_* of the language role */
    unsigned int     budget;      /* turns for each task; zero takes the setting */
    unsigned int     pages;       /* transcript pages; zero takes the setting */
    unsigned int     skills;      /* skills the list holds */
    unsigned int     system_bytes; /* bytes the system block of a prompt of this role took */
    unsigned int     tools[AOTX_CATALOG_MASK_WORDS];
    unsigned int     needs_auth[AOTX_CATALOG_MASK_WORDS];
    unsigned int     skill_mask[AOTX_CATALOG_MASK_WORDS];
    unsigned int     skill[AOTX_CATALOG_ROLE_SKILLS];  /* entries, in the order given */
    aotx_catalog_run overlay;     /* the duty sentence of the role */
} aotx_catalog_role;

/* One entry of the catalog. */
typedef struct aotx_catalog_entry {
    unsigned int       state;     /* AOTX_CATALOG_* */
    unsigned int       kind;      /* AOTX_MODULE_* */
    unsigned int       name_len;
    unsigned int       why;       /* AOTX_CATALOG_WHY_* of a refused entry */
    unsigned int       figure;    /* the bound the reason names, or zero */
    unsigned int       unknown;   /* names of the tools list and the skills list not known */
    char               name[AOTX_CATALOG_NAME_BYTES];
    aotx_catalog_run   manifest;  /* the manifest text */
    aotx_catalog_run   body;      /* the skill body; a role keeps its overlay in the row */
    aotx_catalog_run   description;
    aotx_catalog_run   version;
    unsigned char      digest[32];
    unsigned long long tick;      /* the tick of the import */
    unsigned long long seq;       /* the record sequence of the import that committed it */
    aotx_catalog_tool  tool;
    aotx_catalog_role  role;
} aotx_catalog_entry;

/* One import that arrives. The bytes of the new module land in runs of their own. An
 * import of a name that exists therefore replaces that entry whole at the commit. The runs
 * of the entry that stands are freed at the commit. */
typedef struct aotx_catalog_arriving {
    unsigned int       import;    /* the number of the import; zero when the row is free */
    unsigned int       entry;     /* the entry the head claimed */
    unsigned int       kind;
    unsigned int       files;
    unsigned int       file_bytes[AOTX_IMPORT_FILES];
    unsigned int       got[AOTX_IMPORT_FILES];   /* bytes of each file that arrived */
    aotx_catalog_run   run[AOTX_IMPORT_FILES];
    aotx_catalog_run   held[AOTX_IMPORT_FILES];  /* the runs of the entry that stood */
    unsigned char      digest[32];
    unsigned long long tick;
} aotx_catalog_arriving;

/* The counts the catalog keeps for the panel, the commands and the checks. */
typedef struct aotx_catalog_counts {
    unsigned int heads;       /* heads the apply took */
    unsigned int parts;       /* parts the apply took */
    unsigned int installed;   /* imports that ended INSTALLED */
    unsigned int refused;     /* imports that ended REFUSED */
    unsigned int removed;     /* entries a remove freed */
    unsigned int gone;        /* removes the catalog refused */
    unsigned int replaced;    /* imports that took the entry of a name that stood */
    unsigned int list_cut;    /* prompts whose tool list did not fit the bound */
    unsigned int room_cut;    /* tool results the room of a prompt cut, skill bodies among
                               * them; each one says so in the bytes that went in */
    unsigned int skill_used;  /* skill_use calls that gave a body */
    unsigned int skill_lost;  /* skill_use calls that named no installed skill */
    unsigned int cancelled;   /* imports a head of the same name took the entry from */
    unsigned int last_why;    /* the reason of the refusal that came last */
    unsigned int asked;       /* import lines the console sent to the feeder */
    unsigned int dropped;     /* imports that had not landed when a replay ended */
} aotx_catalog_counts;

/* A remove line waits here until the tick commit node writes its record, exactly as a set
 * line waits. The apply holds the state hash in its own hand until it ends, so a record
 * written inside the apply would fold into nothing. */
#define AOTX_CATALOG_PENDING_MAX 16u

typedef struct aotx_catalog_state {
    aotx_catalog_entry    entry[AOTX_MODULE_SLOTS];
    aotx_catalog_arriving arriving[AOTX_CATALOG_ARRIVING_MAX];
    aotx_catalog_run      free_run[AOTX_CATALOG_FREE_RUNS];
    unsigned int          frees;       /* free runs the list holds */
    unsigned int          used;        /* arena bytes the entries hold */
    unsigned int          conductor;   /* the entry of the role the console speaks to */
    unsigned int          verifier;    /* the entry of the role that judges a result */
    unsigned int          pending_count;
    char                  pending[AOTX_CATALOG_PENDING_MAX][AOTX_CATALOG_NAME_BYTES];
    aotx_catalog_counts   count;
} aotx_catalog_state;

extern __device__ aotx_catalog_state aotx_catalog;
extern __device__ unsigned char aotx_catalog_arena[AOTX_CATALOGUE_BYTES];

/* The bytes of a run of the arena. A run of no length gives the start of the arena, and
 * the caller reads no byte of it. */
__device__ __forceinline__ const char *aotx_catalog_text(aotx_catalog_run run)
{
    return (const char *)aotx_catalog_arena + run.at;
}

/* Report whether an entry holds a mask bit of another entry. */
__device__ __forceinline__ int aotx_catalog_mask_has(const unsigned int *mask,
                                                     unsigned int entry)
{
    if (entry >= AOTX_MODULE_SLOTS) {
        return 0;
    }
    return (mask[entry >> 5] & (1u << (entry & 31u))) != 0u;
}

__device__ __forceinline__ void aotx_catalog_mask_set(unsigned int *mask,
                                                      unsigned int entry)
{
    if (entry < AOTX_MODULE_SLOTS) {
        mask[entry >> 5] |= 1u << (entry & 31u);
    }
}

__device__ __forceinline__ void aotx_catalog_mask_clear(unsigned int *mask)
{
    for (unsigned int i = 0u; i < AOTX_CATALOG_MASK_WORDS; ++i) {
        mask[i] = 0u;
    }
}

/* The name of an entry, which ends with a zero byte. An entry that holds no module gives
 * a dash. A line of the console then shows a dash and no bytes of a name that went. */
__device__ __forceinline__ const char *aotx_catalog_name(unsigned int entry)
{
    if (entry >= AOTX_MODULE_SLOTS
        || aotx_catalog.entry[entry].state == AOTX_CATALOG_FREE
        || aotx_catalog.entry[entry].name_len == 0u) {
        return "-";
    }
    return aotx_catalog.entry[entry].name;
}

/* Report whether an entry is installed and of a kind. */
__device__ __forceinline__ int aotx_catalog_is(unsigned int entry, unsigned int kind)
{
    if (entry >= AOTX_MODULE_SLOTS) {
        return 0;
    }
    return aotx_catalog.entry[entry].state == AOTX_CATALOG_INSTALLED
        && aotx_catalog.entry[entry].kind == kind;
}

/* Find the installed entry of a name, or AOTX_MODULE_SLOTS when the catalog holds none.
 * The compare runs in place, so the call keeps no array of its own. */
__device__ __forceinline__ unsigned int aotx_catalog_find(const char *name,
                                                          unsigned int length,
                                                          unsigned int kind)
{
    if (name == 0 || length == 0u || length >= AOTX_CATALOG_NAME_BYTES) {
        return AOTX_MODULE_SLOTS;
    }
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        const aotx_catalog_entry *row = &aotx_catalog.entry[i];
        if (row->state != AOTX_CATALOG_INSTALLED || row->name_len != length) {
            continue;
        }
        if (kind != 0u && row->kind != kind) {
            continue;
        }
        unsigned int at = 0u;
        while (at < length && row->name[at] == name[at]) {
            at += 1u;
        }
        if (at == length) {
            return i;
        }
    }
    return AOTX_MODULE_SLOTS;
}

/* Find any entry of a name, whatever its state. An import of a name that stands takes
 * that entry, so a refused entry is taken again by the next import of its name. */
__device__ __forceinline__ unsigned int aotx_catalog_find_any(const char *name,
                                                              unsigned int length)
{
    if (name == 0 || length == 0u || length >= AOTX_CATALOG_NAME_BYTES) {
        return AOTX_MODULE_SLOTS;
    }
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        const aotx_catalog_entry *row = &aotx_catalog.entry[i];
        if (row->state == AOTX_CATALOG_FREE || row->name_len != length) {
            continue;
        }
        unsigned int at = 0u;
        while (at < length && row->name[at] == name[at]) {
            at += 1u;
        }
        if (at == length) {
            return i;
        }
    }
    return AOTX_MODULE_SLOTS;
}

/* The name of a state, of a kind and of a side, for the console. */
__device__ __forceinline__ const char *aotx_catalog_state_name(unsigned int state)
{
    switch (state) {
    case AOTX_CATALOG_ARRIVING:  return "arriving";
    case AOTX_CATALOG_INSTALLED: return "installed";
    case AOTX_CATALOG_REFUSED:   return "refused";
    default:                     return "free";
    }
}

__device__ __forceinline__ const char *aotx_catalog_kind_name(unsigned int kind)
{
    switch (kind) {
    case AOTX_MODULE_SKILL: return "skill";
    case AOTX_MODULE_ROLE:  return "role";
    case AOTX_MODULE_TOOL:  return "tool";
    default:                return "-";
    }
}

__device__ __forceinline__ const char *aotx_catalog_side_name(unsigned int side)
{
    switch (side) {
    case AOTX_CATALOG_SIDE_HOST:  return "host";
    case AOTX_CATALOG_SIDE_BUILT: return "built in";
    default:                      return "device";
    }
}

/* The reason of a refused import, as one text. */
__device__ const char *aotx_catalog_why_name(unsigned int why);

/* The reason of a refused remove, as one text. */
__device__ const char *aotx_catalog_gone_name(unsigned int gone);

/* Take one IMPORT record or one REMOVE record, live or replayed. An import body is a head
 * when its part field is zero and a part after that. The return is 0 when the record went
 * in, and 1 when the catalog refused it. The tick comes from the device clock, as it does
 * in the steps of the tick. The apply step makes one call from its serial thread, so the
 * two record types take one call site. */
__device__ int aotx_catalog_apply(unsigned int type, const void *body,
                                  unsigned int body_len, unsigned long long seq);

/* Take one REMOVE record. The commit of the tick calls this for a remove line. */
__device__ int aotx_catalog_remove(const aotx_remove_body *body);

/* Judge a remove with no change to the catalog. The result is one of the AOTX_CATALOG_GONE
 * values, and entry names the row of a name the catalog holds. */
__device__ unsigned int aotx_catalog_remove_judge(const char *name, unsigned int length,
                                                  unsigned int *entry);

/* Write the record of every remove line the tick holds, fold each into the state hash and
 * apply it. The tick commit node calls this on one thread, beside the settings commit. */
__device__ void aotx_catalog_commit(unsigned long long tick);

/* Read one manifest text into an entry. The reader is a state machine over key: value
 * lines with # comments, and over the two-key head of a skill file. The return is
 * AOTX_CATALOG_WHY_NONE when the text is a manifest of the kind the head named. The at
 * field takes the offset of the manifest in the arena. Every run of the entry then points
 * into the arena, and the reader copies nothing. */
__device__ unsigned int aotx_catalog_manifest_read(aotx_catalog_entry *row,
                                                   unsigned int at, unsigned int length,
                                                   unsigned int kind, unsigned int *figure);

/* Take a run of the arena by first fit, and give a run back to the free list. The return
 * of the take is 0 when the run is in hand and 1 when the arena holds no run of that
 * length. A length of zero gives a run of no length and takes no bytes. */
__device__ int aotx_catalog_take_run(unsigned int length, aotx_catalog_run *run);
__device__ void aotx_catalog_free_run(aotx_catalog_run run);

/* Give every arena run of an entry back. The entry keeps its name and its reason. */
__device__ void aotx_catalog_release(aotx_catalog_entry *row);

/* Report whether the free list of the arena is sound. The list stands in the order of the
 * offsets. No run of it touches or overlaps the run before it. Every run is inside the
 * arena. The free bytes and the bytes the entries hold are the whole arena. The return is
 * 1 when every one of those holds. */
__device__ int aotx_catalog_arena_sound(void);

/* Give the offset after the head of a skill file, which stands between two lines of three
 * dashes. The return is the offset of the first byte of the body, or at when the text
 * carries no such head. */
__device__ unsigned int aotx_catalog_head_end(unsigned int at, unsigned int length);

/* Read the entry of the role of the console and of the role that judges a result. A path
 * of the engine names each one, so the catalog keeps the entry of each after a change. */
__device__ void aotx_catalog_anchor(void);

/* Put the built-in tools in the catalog. The call is idempotent: a second call changes
 * nothing. One thread makes it, before the first tick. */
__device__ void aotx_catalog_built_in(void);

/* Drop every import that had not landed when a replay of the journal ended. The number of
 * an import is unique while that import arrives and no longer. A run that was killed in
 * the middle of an import leaves a head with no last part. The entry of such an import
 * goes free, its runs go back, and one console line names the count. The apply calls this
 * on the record of the restore, from its serial thread. */
__device__ void aotx_catalog_restore_end(unsigned long long tick);

/* Fill the catalog with the built-in tools and an empty arena. The glue launches this
 * once, before the first tick. */
__global__ void aotx_catalog_boot(void);

/* Host glue: put the built-in tools in the catalog before the tick capture starts. The
 * return is zero when the catalog holds them. */
int aotx_catalog_open(void);

/* The run of one argument key of a tool entry. A key that is not one of the tool gives a
 * run of no length. */
__device__ __forceinline__ aotx_catalog_run aotx_catalog_arg_key(unsigned int entry,
                                                                 unsigned int key)
{
    aotx_catalog_run run;
    run.at = 0u;
    run.length = 0u;
    if (entry < AOTX_MODULE_SLOTS && key < aotx_catalog.entry[entry].tool.arguments) {
        run = aotx_catalog.entry[entry].tool.key[key];
    }
    return run;
}

/* Keep the bytes the system block of a prompt of a role took, and read them back. The
 * turn takes the room that is left, so a role with a long list gives its turn less. */
__device__ __forceinline__ void aotx_catalog_system_seen(unsigned int role,
                                                         unsigned int bytes)
{
    if (role < AOTX_MODULE_SLOTS) {
        aotx_catalog.entry[role].role.system_bytes = bytes;
    }
}

__device__ __forceinline__ unsigned int aotx_catalog_system_bytes(unsigned int role)
{
    return (role < AOTX_MODULE_SLOTS) ? aotx_catalog.entry[role].role.system_bytes : 0u;
}

/* Report whether a role may call a tool, and whether that tool waits for the operator. A
 * tool whose manifest says always waits for every role. */
__device__ __forceinline__ int aotx_catalog_may_call(unsigned int role, unsigned int tool)
{
    if (aotx_catalog_is(tool, AOTX_MODULE_TOOL) == 0 || role >= AOTX_MODULE_SLOTS) {
        return 0;
    }
    return aotx_catalog_mask_has(aotx_catalog.entry[role].role.tools, tool);
}

__device__ __forceinline__ int aotx_catalog_needs_auth(unsigned int role, unsigned int tool)
{
    if (aotx_catalog_is(tool, AOTX_MODULE_TOOL) == 0) {
        return 0;
    }
    if (aotx_catalog.entry[tool].tool.authorize == AOTX_CATALOG_AUTH_ALWAYS) {
        return 1;
    }
    if (role >= AOTX_MODULE_SLOTS) {
        return 0;
    }
    return aotx_catalog_mask_has(aotx_catalog.entry[role].role.needs_auth, tool);
}

/* Add the tool list and the skill list of a role to a prompt, in the shape the chat
 * template of the model file gives. The return is the position after the lists. The count
 * of the tools that did not fit the bound goes in the catalog counters. */
__device__ unsigned int aotx_catalog_tool_list(unsigned char *out, unsigned int at,
                                               unsigned int role);

/* Add the body of every skill of a role to a prompt, in the order the manifest gave. */
__device__ unsigned int aotx_catalog_skill_bodies(unsigned char *out, unsigned int at,
                                                  unsigned int role);

#endif
