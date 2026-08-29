/* Purpose: Start the system: context, modules, reservation, rings.
 * Owns: The boot record and the module table.
 * Launch shape: Host glue only; no kernels.
 * Lifetime: From start to the first tick. */
#ifndef AOTX_BOOT_CUH
#define AOTX_BOOT_CUH

#include "sched/sched.cuh"
#include "seam/seam.cuh"

/* What the command line of the boot program gives. A null text field is not given. */
typedef struct aotx_boot_options {
    const char *journal;
    const char *derive;          /* record types the drain makes lines from; null is default */
    const char *models;          /* directory of the model files, or none */
    const char *roles;           /* roles of the model file list to load; null is the default */
    const char *root;            /* the one directory a file read may reach; null is none */
    const char *modules;         /* directory of the module directories, or none */
    const char *settings;        /* the settings file; null takes the default path */
    unsigned long long ticks;    /* ticks to run; zero runs on until the record target */
    unsigned long long workload; /* records the tick load writes for each tick */
    unsigned long long records;  /* record target of a run that has no tick count */
    unsigned int blocks;         /* blocks of the tick load */
    int restore;
    int clock_only;
    int solo;                    /* run with no disk side programs */
    int window;                  /* show the panels in a window on the display */
    int tui;                     /* start the terminal program beside the system */
    int tui_attached;            /* a terminal started this run and is attached already */
    int version;                 /* write the version line and stop */
} aotx_boot_options;

/* The reader of the settings file is the C library of the disk side. The glue holds a
 * pointer to its table and reads the fields the boot keys name. */
struct aotx_settings;

/* Read the settings file and take the boot keys the command line did not give. The path is
 * the one the command line names, or aotx.settings beside the parent of the journal
 * directory. A command line option wins over the file for the same key. Every refused line
 * of the file is written, one a line. The path that was read goes in path. The return is 0
 * when the file was read or was not there, and 1 when it could not be read. */
int aotx_boot_settings(aotx_boot_options *options, struct aotx_settings *table,
                       char *path, unsigned int bytes);

/* Write the version, the profile, the architecture and the slot count on one line. */
void aotx_boot_version(void);

/* Read the card and refuse a profile that its free memory cannot hold. The check runs
 * before any placement. The return is 0, or 2 for a refusal. */
int aotx_boot_card_check(void);

/* Write the record that states the card and the build. The record follows the boot record
 * of the run. A run whose check did not pass writes nothing. */
void aotx_boot_card_note(void);

/* Write the options of the program. */
void aotx_boot_usage(void);

/* Read the command line. The return is zero when every option is known. */
int aotx_boot_parse(int argc, char **argv, aotx_boot_options *options);

/* The clock module proves the raw module path: load, graph node, launch, sample. */
int aotx_boot_clock_check(unsigned long long *sample);

/* The disk side programs of this run. A value of zero means the program does not run. */
typedef struct aotx_boot_children {
    int drain;
    int feed;
    int restore;
    int tui;
} aotx_boot_children;

/* Find a program that sits beside this one. The return is zero when the path is found. */
int aotx_boot_sibling(const char *name, char *path, unsigned int bytes);

/* Start the program that writes the journal. The derive list names the record types the
 * drain makes lines from. A null pointer gives the default of the drain. */
int aotx_boot_start_drain(aotx_boot_children *children, const aotx_seam_rings *rings,
                          const char *journal, const char *derive);

/* Start the program that writes the inbound ring. A key descriptor of zero or more gives
 * the feeder the read end of the key pipe. The feeder makes a key record of each frame.
 * A root which is not null lets the feeder read a file below that directory, and it then
 * reads the requests of the journal directory. The settings path names the file the feeder
 * reads; the feeder publishes the device keys of that file before the first line. The
 * feeder listens for a terminal in the journal directory and gives it the mirror. */
int aotx_boot_start_feed(aotx_boot_children *children, const aotx_seam_rings *rings,
                         int keys_fd, const char *root, const char *journal,
                         const char *settings, const char *modules);

/* Start the terminal program beside this one. It attaches to the socket of the feeder in
 * the journal directory and draws the panels of the mirror. The return is zero when the
 * program runs. */
int aotx_boot_start_tui(aotx_boot_children *children, const char *journal);

/* Replay the journal: start the restore program, run ticks until it ends and the inbound
 * ring is empty, then write the restore record. */
int aotx_boot_replay(aotx_boot_children *children, const aotx_seam_rings *rings,
                     const char *journal, aotx_pump *pump);

/* Wait for every program that still runs. */
void aotx_boot_stop(aotx_boot_children *children);

/* Check the model files of a directory, stream their tensors into the weights region, and
 * build the vocabulary of the tokenizer. The return is zero when every file is in place,
 * and 2 when a file does not match its record in the model file list. */
/* The stop reader lets a signal end a load which takes tens of seconds. The boot program
 * gives aotx_boot_signal, and a caller with no signal path gives a null pointer. The roles
 * text names the entries to load, with commas between them; a null pointer takes the
 * default list of cuda/model/roles.h. */
int aotx_boot_models(const char *dir, const char *roles, int (*stopped)(void));

/* Give the memory of the vocabulary tables back. */
void aotx_boot_models_release(void);

/* The model file reader of the disk side gives this structure. The name is enough here,
 * because the glue holds a pointer to it and never reads a field of it. */
struct aotx_modelfile;

/* Open and close the pinned buffers and the copy stream of a model load. */
int aotx_boot_weights_open(void);
void aotx_boot_weights_close(void);

/* Place every tensor of one model file in the weights region and put the tensors of that
 * file in the device table. The cursor moves by the bytes of each tensor that is placed. */
int aotx_boot_weights_place(struct aotx_modelfile *file, unsigned int model,
                            unsigned long long *cursor, unsigned int *placed,
                            unsigned int *left);

/* Read the flag that the quit command sets. A value above zero stops the run. */
unsigned int aotx_boot_quit(void);

/* Give the number of the stop signal that came, or zero. The tick loop and the window loop
 * read this, so a signal ends the run through the path the quit command takes. */
int aotx_boot_signal(void);

/* Open the state file with placing. Later calls replace its word. The close call writes
 * closed. Each line includes the current Unix seconds. */
int aotx_boot_phase_open(const char *journal);
int aotx_boot_phase_set(const char *word);
void aotx_boot_phase_close(void);

/* Run the tick pump on its own thread and the window on the thread that calls this. The
 * return is zero when the window closed or the quit command stopped the run. The derive
 * text goes in the report of the window, so a run states what the drain made lines from. */
int aotx_boot_window_run(aotx_pump *pump, int keys_fd, const char *derive);

#endif
