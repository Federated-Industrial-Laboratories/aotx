/* Purpose: Queue, place, record and list models loaded while the system runs.
 * Owns: The model file list, the load queue and the resident model rows.
 * Launch shape: One thread for command, apply and commit work; host glue moves file bytes.
 * Lifetime: From the model file list load to the end of the run. */
#ifndef AOTX_MODEL_LOAD_CUH
#define AOTX_MODEL_LOAD_CUH

#include "cli/cli.cuh"
#include "model/model.cuh"
#include "profile/profile.cuh"
#include "seam/wire.h"

#define AOTX_MODEL_FILES_MAX    8u
#define AOTX_MODEL_LOAD_MAX     8u
#define AOTX_MODEL_LOAD_NONE    0u
#define AOTX_MODEL_LOAD_DIGEST  1u
#define AOTX_MODEL_LOAD_FILE    2u
#define AOTX_MODEL_LOAD_REGION  3u
#define AOTX_MODEL_LOAD_DESC    4u

typedef struct aotx_model_file_row {
    unsigned long long bytes;
    unsigned int model;       /* place in the model file list and tensor table */
    unsigned int role;        /* AOTX_MODEL_* from the entry role */
    unsigned char digest[32];
    char name[64];
    char file[64];
} aotx_model_file_row;

typedef struct aotx_model_load_row {
    aotx_model_body body;
    unsigned int source;      /* row of file[] */
    unsigned int target;      /* role named by the command */
    unsigned int slot;        /* descriptor role that receives the file */
    unsigned int replayed;    /* the MODEL record already exists */
} aotx_model_load_row;

typedef struct aotx_model_resident_row {
    aotx_model_body body;
    unsigned int source;
    unsigned int slot;
    unsigned int active;
    unsigned int reserved;
} aotx_model_resident_row;

typedef struct aotx_model_load_state {
    aotx_model_file_row file[AOTX_MODEL_FILES_MAX];
    aotx_model_load_row pending[AOTX_MODEL_LOAD_MAX];
    aotx_model_resident_row resident[AOTX_MODEL_ROLES];
    aotx_model_load_row placed;
    unsigned int files;
    unsigned int pending_count;
    unsigned int placed_ready;
    unsigned int refused;
    unsigned int replay_bad;
    unsigned long long placed_bytes;
} aotx_model_load_state;

extern __device__ aotx_model_load_state aotx_model_load;

/* Queue a judged model load from a live command. */
__device__ void aotx_model_load_command(aotx_cli_out *out, const char *role,
                                        unsigned int role_len, const char *name,
                                        unsigned int name_len, unsigned int extra,
                                        unsigned long long tick);

/* Show the resident role, file, digest and placement tick. */
__device__ void aotx_model_show_command(aotx_cli_out *out);

/* Apply a replayed MODEL body. A refusal returns one. */
__device__ int aotx_model_load_apply(const aotx_model_body *body);

/* Write a live MODEL record after placement, fold it and make it resident. */
__device__ void aotx_model_load_commit(unsigned long long tick);

/* Mark the start and the result of placement between two ticks. */
__global__ void aotx_model_load_begin(void);
__global__ void aotx_model_load_finish(unsigned int success, unsigned int reason,
                                       unsigned long long bytes);

/* Read the model file list and publish the rows a live command judges. */
int aotx_model_load_open(const char *dir, const char *roles,
                         unsigned long long cursor);

struct aotx_pump;

/* Place one queued file. A nonzero return means a replay cannot continue. */
int aotx_model_load_step(struct aotx_pump *pump);

#endif
