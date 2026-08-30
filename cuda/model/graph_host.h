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
    unsigned int role;      /* the role the capture runs */
    unsigned int wave;      /* blocks of a launch that takes a run of rows */
    unsigned int decode;    /* 1 while the capture builds the graph of the decode */
} aotx_model_hold;

/* The state of one role, or a null pointer when the role is outside the table. */
aotx_model_hold *aotx_model_hold_of(unsigned int role);

/* The first byte of a tensor of the model, or a null pointer when the model has none. */
const void *aotx_model_tensor(unsigned long long at);

/* Put one matrix node in the capture. The tile of the launch is the tile of the header.
 * The capture of the decode puts a node that takes its batch from the call block. It puts
 * a module node in front of that node where the module serves a batch of one row. */
void aotx_model_matrix(aotx_model_hold *hold, const void *w, unsigned int type,
                       unsigned int n, unsigned int k, const half *x, unsigned int m,
                       float *y, unsigned int which);

/* Put the module node of one matrix product in the capture. The return is 1 when the node
 * is in the capture, and the compiled node then exits at a batch of one row. */
unsigned int aotx_model_module_node(aotx_model_hold *hold, const void *w, unsigned int type,
                                    unsigned int n, unsigned int k, const half *x, float *y,
                                    dim3 grid, unsigned int which);

/* Make the decode ready before a tick capture starts. The call captures the forward pass
 * of the language role and loads the module of the memory bound product. The buffers and
 * the module are made here, because those calls are not allowed inside a capture. The
 * return is zero when the decode is ready to go in a tick. */
int aotx_decode_open(void);

/* Rebuild the language hold and its child graph after its weights change. */
int aotx_decode_replace(unsigned int role);

/* Keep the address of the batch counts of one role. A capture that takes its batch from
 * the call block then has a pointer to that count. */
void aotx_model_batch_of(unsigned int role);

/* The address of the batch count that a matrix node of a role reads. */
const unsigned int *aotx_decode_batch_word(unsigned int role, unsigned int which);

/* Nodes of the child graph that holds the forward pass of the decode. */
unsigned int aotx_decode_nodes(void);

/* Give the child graph of the decode and the module of the memory bound product back. */
void aotx_decode_close(void);

/* Put the nodes of one layer, and then the nodes of the head, in the capture. */
void aotx_model_capture_layer(aotx_model_hold *hold, unsigned int role, unsigned int layer);
void aotx_model_capture_head(aotx_model_hold *hold, unsigned int role);

#endif
