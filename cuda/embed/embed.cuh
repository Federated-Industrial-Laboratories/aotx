/* Purpose: Compute embeddings, keep the notes they stand for, and search them.
 * Owns: The note store: one unit vector, one record sequence and the text of each note.
 * Launch shape: One block for each sequence; the threads hold the hidden width.
 * Lifetime: The whole run. */
#ifndef EMBED_CUH
#define EMBED_CUH

#include "model/forward.cuh"

/* The embedding head. The vector of a sequence is the last row of the sequence after the
 * last norm, made a unit vector. The model file states this pooling with the value 3. */
#define AOTX_EMBED_POOL_LAST   3u

__global__ void aotx_embed_pool(unsigned int role);

/* Notes the store holds. A note that finds the store full is refused and counted, so the
 * store keeps the notes that went in first. */
#define AOTX_EMBED_NOTES   1024u

/* The widest vector the store keeps. The embedding role of this system gives 1,024. */
#define AOTX_EMBED_WIDTH   1024u

/* Bytes of the text of one note. The value is the argument of a tool call. */
#define AOTX_EMBED_TEXT    160u

/* Notes that one search gives back. */
#define AOTX_EMBED_HITS    4u

/* The note store. A note is one finding: its unit vector, the record sequence of the bus
 * message that carries it, and its text. The vectors are unit vectors, so the cosine of
 * two of them is their dot product. */
typedef struct aotx_embed_store {
    float vector[AOTX_EMBED_NOTES][AOTX_EMBED_WIDTH];
    unsigned long long seq[AOTX_EMBED_NOTES];   /* the record of the finding */
    unsigned int len[AOTX_EMBED_NOTES];
    unsigned char text[AOTX_EMBED_NOTES][AOTX_EMBED_TEXT];
    unsigned int count;     /* notes the store holds, up to AOTX_EMBED_NOTES */
    unsigned int width;     /* floats of one vector, from the first note that went in */
    unsigned int refused;   /* notes a width or a full store refused */
} aotx_embed_store;

extern __device__ aotx_embed_store aotx_embed_notes;

/* Put one note in the store and give its place, or the note count when the store refused
 * it. The caller holds one thread for each note it adds. */
__device__ unsigned int aotx_embed_keep(const float *vector, unsigned int width,
                                        const char *text, unsigned int length,
                                        unsigned long long seq);

/* What one search launch takes. Every pointer names device memory of the caller. */
typedef struct aotx_embed_query {
    const float *vector;       /* one row for each query, width floats apart */
    const unsigned int *live;  /* 1 when the query of that place is to be searched */
    unsigned int *hit;         /* AOTX_EMBED_HITS places for each query */
    float *score;              /* the cosine of each of those places */
    const unsigned int *count; /* queries of the batch; the kernel reads it at the launch */
    unsigned int width;        /* floats of one vector; the model gives it at the capture */
} aotx_embed_query;

/* Search the store for every live query of a batch. One block takes one query; the threads
 * of the block hold the width. The nearest AOTX_EMBED_HITS notes go in hit and score, from
 * the nearest. A place that no note fills gives the note count and a score of minus one.
 *
 * The batch of a tick is dynamic and the parameters of a captured node are not. The count
 * therefore comes through a pointer into device state. The grid covers the largest batch,
 * and a block above the count exits at once. */
__global__ void aotx_embed_search(aotx_embed_query set);

#endif
