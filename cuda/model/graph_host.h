/* Purpose: Share the state of one model role between the glue files of the module.
 * Owns: Nothing; the state lives in model_host.cu.
 * Launch shape: Not applicable; host glue support.
 * Lifetime: From the graph capture to the close. */
#ifndef AOTX_MODEL_GRAPH_HOST_H
#define AOTX_MODEL_GRAPH_HOST_H

#include <cuda_runtime.h>

#include "model/forward.cuh"

/* The state of one role. The head node is the one node that a call changes. The output
 * head takes every row of the batch, or one row of each sequence. */
typedef struct aotx_model_hold {
    aotx_model_work work;
    aotx_model_run *pinned;
    aotx_model_desc desc;
    void *piece[20];
    unsigned int pieces;
    cudaStream_t stream;
    cudaEvent_t event;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaGraphNode_t head_node;
    const void *head_w;
    unsigned int head_type;
    unsigned int head_n;
    unsigned int head_k;
    const half *head_x;
    unsigned int head_m;
    float *head_y;
    unsigned int max_tokens;
    unsigned int max_rows;
    unsigned int ready;
} aotx_model_hold;

/* The state of one role, or a null pointer when the role is outside the table. */
aotx_model_hold *aotx_model_hold_of(unsigned int role);

/* The first byte of a tensor of the model, or a null pointer when the model has none. */
const void *aotx_model_tensor(unsigned long long at);

/* Put one matrix node in the capture. The tile of the launch is the tile of the header. */
void aotx_model_matrix(cudaStream_t stream, const void *w, unsigned int type,
                       unsigned int n, unsigned int k, const half *x, unsigned int m,
                       float *y);

/* Put the nodes of one layer, and then the nodes of the head, in the capture. */
void aotx_model_capture_layer(aotx_model_hold *hold, unsigned int role, unsigned int layer);
void aotx_model_capture_head(aotx_model_hold *hold, unsigned int role);

#endif
