/* Purpose: Start the system: context, modules, reservation, rings.
 * Owns: The boot record and the module table.
 * Launch shape: Host glue only; no kernels.
 * Lifetime: From start to the first tick. */
#ifndef AOTX_BOOT_CUH
#define AOTX_BOOT_CUH

#include "sched/sched.cuh"
#include "seam/seam.cuh"

/* The clock module proves the raw module path: load, graph node, launch, sample. */
int aotx_boot_clock_check(unsigned long long *sample);

/* The disk side programs of this run. A value of zero means the program does not run. */
typedef struct aotx_boot_children {
    int drain;
    int feed;
    int restore;
} aotx_boot_children;

/* Find a program that sits beside this one. The return is zero when the path is found. */
int aotx_boot_sibling(const char *name, char *path, unsigned int bytes);

/* Start the program that writes the journal. The derive list names the record types the
 * drain makes lines from. A null pointer gives the default of the drain. */
int aotx_boot_start_drain(aotx_boot_children *children, const aotx_seam_rings *rings,
                          const char *journal, const char *derive);

/* Start the program that writes the inbound ring. A key descriptor of zero or more gives
 * the feeder the read end of the key pipe. The feeder makes a key record of each frame. */
int aotx_boot_start_feed(aotx_boot_children *children, const aotx_seam_rings *rings,
                         int keys_fd);

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
 * gives aotx_boot_signal, and a caller with no signal path gives a null pointer. */
int aotx_boot_models(const char *dir, int (*stopped)(void));

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

/* Run the tick pump on its own thread and the window on the thread that calls this. The
 * return is zero when the window closed or the quit command stopped the run. The derive
 * text goes in the report of the window, so a run states what the drain made lines from. */
int aotx_boot_window_run(aotx_pump *pump, int keys_fd, const char *derive);

#endif
