/* Purpose: Declare steer vectors, voice bias profiles and page instruments.
 * Owns: The registered conduct tables and the page mass table.
 * Launch shape: One block for each residual row; one thread flushes page records.
 * Lifetime: From model load to model release. */
#ifndef AOTX_MODEL_CONDUCT_CUH
#define AOTX_MODEL_CONDUCT_CUH

#include "model/forward.cuh"

#define AOTX_CONDUCT_VECTORS       16u
#define AOTX_CONDUCT_VOICES        16u
#define AOTX_CONDUCT_NAME_BYTES    32u
#define AOTX_CONDUCT_LAYERS        64u
#define AOTX_CONDUCT_BIASES        128u
#define AOTX_PAGE_FLUSH_TICKS      64u

typedef struct aotx_steer_vector {
    unsigned long long value; /* layer-major float values on the device */
    unsigned long long layers; /* bit for each layer held by the file */
    unsigned int hidden;
    unsigned int layer_count;
    float potency;
    char name[AOTX_CONDUCT_NAME_BYTES];
} aotx_steer_vector;

typedef struct aotx_voice_bias {
    unsigned int token[AOTX_CONDUCT_BIASES];
    float bias[AOTX_CONDUCT_BIASES];
    unsigned int count;
    char name[AOTX_CONDUCT_NAME_BYTES];
} aotx_voice_bias;

typedef struct aotx_conduct_table {
    aotx_steer_vector vector[AOTX_CONDUCT_VECTORS];
    aotx_voice_bias voice[AOTX_CONDUCT_VOICES];
    unsigned int vectors;
    unsigned int voices;
    unsigned int refused;
} aotx_conduct_table;

extern __device__ aotx_conduct_table aotx_conduct;
extern __device__ float aotx_page_mass[AOTX_SLOTS][AOTX_KV_PAGES_EACH];

__device__ unsigned int aotx_conduct_vector(const char *name, unsigned int length);
__device__ unsigned int aotx_conduct_voice(const char *name, unsigned int length);
__device__ float aotx_conduct_bias(unsigned int profile, unsigned int token);
__device__ void aotx_page_flush(unsigned long long tick);

int aotx_conduct_register_vector(const char *name, const unsigned int *layers,
                                 unsigned int layer_count, unsigned int hidden,
                                 const float *device_values, float potency);
int aotx_conduct_register_voice(const char *name, const unsigned int *tokens,
                                const float *bias, unsigned int count);
void aotx_conduct_release(void);
int aotx_conduct_load_store(const char *dir);

/* Resolve one profile string through the vocabulary that model load built. */
__global__ void aotx_conduct_token(const unsigned char *text, unsigned int length,
                                   unsigned int *token);

#endif
