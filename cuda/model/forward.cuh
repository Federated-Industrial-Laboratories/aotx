/* Purpose: Declare the forward pass: its buffers, its call block and its kernels.
 * Owns: The call block, the buffer block and the cache position of each agent slot.
 * Launch shape: One block for each row, or for each row and head; the batch is the tokens.
 * Lifetime: From the graph capture at model load to the close at the end of the run. */
#ifndef AOTX_MODEL_FORWARD_CUH
#define AOTX_MODEL_FORWARD_CUH

#include <cuda_fp16.h>

#include "model/kv_layout.cuh"
#include "model/matrix.cuh"
#include "model/model.cuh"

/* Tokens that one pass takes. The tick budget for prefill is 512 tokens, so a prompt which
 * is longer goes through the pass in pieces. */
#define AOTX_MODEL_MAX_TOKENS   512u

/* Sequences that one pass takes. One sequence holds one agent slot of the page cache. */

/* The norm weight that a norm launch applies. */
#define AOTX_MODEL_NORM_ATTN    0u
#define AOTX_MODEL_NORM_FFN     1u
#define AOTX_MODEL_NORM_OUT     2u

/* The rows that the output head takes: every token, or the last token of each sequence. */
#define AOTX_MODEL_ROWS_ALL     0u
#define AOTX_MODEL_ROWS_LAST    1u

/* Threads of an elementwise launch, and of a launch that reduces one row. */
#define AOTX_MODEL_ROW_THREADS  256u

/* The widest head the attention kernel takes. One warp of 32 lanes holds a head, so a lane
 * holds four elements of it. A head must also be a whole number of lanes. */
#define AOTX_MODEL_HEAD_MAX     128u

/* Tokens that one attention block takes. The block holds one warp for each token. */
#define AOTX_MODEL_ATTN_TOKENS  4u
#define AOTX_MODEL_ATTN_THREADS (32u * AOTX_MODEL_ATTN_TOKENS)

/* The batch at which the tensor core product takes over from the memory bound product.
 * Below this count the read of the weights is the whole cost. The memory bound product
 * reads each weight one time and applies it to every row of the batch.
 *
 * The count is measured and not assumed. The memory bound product reads the weights again
 * for each group of AOTX_GEMV_BATCH rows, so its cost rises in steps. A decode tick of
 * the 4B file takes 35.98 ms at 8 rows, 73.29 at 16, 105.76 at 24 and 138.97 at 32.
 *
 * The tensor core product pays for the whole AOTX_GEMM_TILE_M rows of its tile at every
 * batch. The same tick takes 73.63 ms at 16 rows and 78.09 at 64. The two products are
 * level at 16 rows, within 0.5 percent. The memory bound product therefore keeps 16 and
 * the tensor core product takes 17 and above. */
#define AOTX_MODEL_TENSOR_MIN   17u

/* Which count of the call block a matrix node takes as its batch. A layer product takes
 * every token; the output head takes one row for each sequence. */
#define AOTX_MODEL_BATCH_TOKENS 0u
#define AOTX_MODEL_BATCH_ROWS   1u

/* Grid rows of a matrix node that takes its batch from the call block. The tile of the
 * tensor core product is AOTX_GEMM_TILE_N columns wide. One block of the memory bound
 * product takes AOTX_GEMV_ROWS_CTA rows of the weight tensor. This count of tiles
 * therefore gives the blocks that the memory bound product needs, and no block is left
 * over in either product. */
#define AOTX_MODEL_PRODUCT_ROWS (AOTX_GEMM_TILE_N / AOTX_GEMV_ROWS_CTA)

/* Candidates that the sample kernel holds in shared memory. The kernel raises its threshold
 * until the candidates fit, so a vocabulary of any size passes through this bound. */
#define AOTX_MODEL_PICK_MAX     256u

/* The range of logits below the largest that the sample kernel looks at. A logit further
 * below the largest gives a probability under 1e-18 after the softmax. */
#define AOTX_MODEL_PICK_SPAN    42.0f

/* How a sample is taken. A temperature of zero gives the largest logit. The neutral
 * values leave every logit unchanged and keep the greedy result. */
typedef struct aotx_model_how {
    float temperature;           /* the divisor of the logits */
    unsigned int top_k;          /* candidates kept, or zero for every candidate */
    float top_p;                 /* probability mass kept, from zero to one */
    float min_p;                 /* least probability relative to the largest */
    float repeat_penalty;        /* divisor for a repeated positive logit */
    unsigned int repeat_window;  /* recent tokens checked for repetition */
    float presence_penalty;      /* subtraction when a token is present */
    float frequency_penalty;     /* subtraction for each use of a token */
    unsigned long long seed;     /* the seed of the random stream */
    int think_limit;             /* tokens in a thinking span, or -1 for no limit */
    unsigned int reserved;
} aotx_model_how;

/* The parameters of one pass. The graph copies this block to the device before the first
 * kernel, so every kernel of the pass reads the batch of the call. Every pointer here names
 * memory of the device, because the kernels read and write it. */
typedef struct aotx_model_run {
    const int *ids;              /* the token of every row, sequence after sequence */
    const unsigned int *offset;  /* first row of each sequence, and the row count at seqs */
    const unsigned int *agent;   /* the page cache slot of each sequence */
    float *logits;               /* the vocabulary row of each output row, or null */
    float *pooled;               /* one hidden row for each sequence, or null */
    float *score;                /* one score for each sequence, or null */
    int *token;                  /* the sampled token of each sequence, or null */
    unsigned int *draw;          /* the stream position each draw took, or null */
    unsigned long long seed;     /* the seed the sample kernel takes */
    unsigned int seqs;
    unsigned int tokens;         /* rows of the batch */
    unsigned int rows;           /* rows the output head takes */
    unsigned int select;         /* AOTX_MODEL_ROWS_ALL or AOTX_MODEL_ROWS_LAST */
    unsigned int top_k;
    float top_p;
    float temperature;
    const aotx_model_how *how;   /* the sample of each sequence, or null for the four above */
    unsigned int telemetry;      /* one writes token statistics for language decode */
} aotx_model_run;

extern __device__ aotx_model_run aotx_model_call[AOTX_MODEL_ROLES];

/* The buffers of one pass. The host glue allocates them at the capture and writes this
 * block once, so a kernel takes the role and the layer alone. */
typedef struct aotx_model_work {
    float *resid;      /* tokens by hidden: the residual stream */
    half *x;           /* tokens by hidden: the input of a projection */
    float *q;          /* tokens by heads by head_dim */
    half *qh;          /* the same, after the head norm and the angle */
    float *k;          /* tokens by kv_heads by head_dim */
    float *v;          /* tokens by kv_heads by head_dim */
    half *att;         /* tokens by heads by head_dim: the attention result */
    float *proj;       /* tokens by hidden: the result of a projection back to hidden */
    float *gate;       /* tokens by ffn */
    float *up;         /* tokens by ffn */
    half *act;         /* tokens by ffn: the gate and the up together */
    half *xnorm;       /* tokens by hidden: the residual stream after the last norm */
    half *sel;         /* rows by hidden: the rows the output head takes */
    float *head;       /* rows by the head width: the result of the output head */
    unsigned int *row; /* the token row that each output row takes */
    unsigned int *base;/* the first cache position of each sequence */
    unsigned long long weights; /* the first byte of the weights region */
    aotx_kvl_shape shape;
    unsigned int max_tokens;
    unsigned int max_rows;
} aotx_model_work;

extern __device__ aotx_model_work aotx_model_space[AOTX_MODEL_ROLES];

/* The cache positions that each agent slot holds. A pass writes its keys and values after
 * them and then moves them on, so a long prompt goes through the pass in pieces. */
extern __device__ unsigned int aotx_model_seen[AOTX_SLOTS];

/* The draws each agent slot has taken. The slot and this count key the random stream, so a
 * replay of the seed and the count gives the token again. */
extern __device__ unsigned int aotx_model_draw[AOTX_SLOTS];

/* Rows that found no cache page. A count above zero means the caller did not answer the
 * page requests of the pass, and the result of the pass is not correct. */
extern __device__ unsigned int aotx_model_faults;

/* The block type of the output tensor and of the class tensor of each role. The descriptor
 * states the type of the projections and of the token embedding. These two stand beside
 * them, because a file may hold its head in another type. */
extern __device__ unsigned int aotx_model_head_type[AOTX_MODEL_ROLES][2];

/* Report whether a role gives the logits of a vocabulary. Two roles hold a language model:
 * the one of the run and the one of the four bit file the accuracy gate reads. */
__device__ __host__ __forceinline__ int aotx_model_is_language(unsigned int role)
{
    return role == AOTX_MODEL_LANGUAGE || role == AOTX_MODEL_LANGUAGE_Q4;
}

/* Find the sequence of a row of the batch. The offsets go up, so the search is a bisection.
 * Every kernel that needs a position calls this, because a row carries no sequence. */
__device__ __forceinline__ unsigned int aotx_model_which(const unsigned int *offset,
                                                         unsigned int seqs, unsigned int row)
{
    unsigned int low = 0u;
    unsigned int high = seqs;
    while (low + 1u < high) {
        unsigned int mid = (low + high) / 2u;
        if (offset[mid] <= row) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return low;
}

/* Read the cache positions of every sequence and ask the page cache for the pages that this
 * pass needs. The caller answers the requests before it launches the pass. */
__global__ void aotx_model_request(unsigned int role);

/* Write the first cache position of every sequence, one thread for each sequence. */
__global__ void aotx_model_open_rows(unsigned int role);

/* Move the cache position of every sequence on by the tokens it gave. */
__global__ void aotx_model_shut_rows(unsigned int role);

/* Read the token embedding row of every token into the residual stream and the half copy. */
__global__ void aotx_model_gather(unsigned int role);

/* Apply one root mean square norm over the hidden width, one block for each token. */
__global__ void aotx_model_norm(unsigned int role, unsigned int layer, unsigned int which);

/* Add the projection result into the residual stream. */
__global__ void aotx_model_residual(unsigned int role);

/* Apply the gate to the up value: the gate through a sigmoid linear unit, then a product. */
__global__ void aotx_model_swiglu(unsigned int role);

/* Norm each head of the query and the key, and turn them by the position angle. Write the
 * key and the value of every token into the pages of its slot. */
__global__ void aotx_model_qkv(unsigned int role, unsigned int layer);

/* Attention over the pages, one warp for each token of one head. */
__global__ void aotx_model_attend(unsigned int role, unsigned int layer);

/* Build the row list of the output head and gather those rows. */
__global__ void aotx_model_select(unsigned int role);

/* Take one token for each sequence from the logits of its last row. */
__global__ void aotx_model_pick(unsigned int role);

/* The two matrix nodes of a captured graph. The batch comes from the call block, so a node
 * takes the shape of the tick without a change of its parameters. The tensor core node
 * runs at AOTX_MODEL_TENSOR_MIN rows and above; the memory bound node runs below that
 * count. Each one exits at once when the batch is not its own.
 *
 * The batch pointer names the token count or the row count of the call block. The grid of
 * the tensor core node is the tiles of its result. The grid of the memory bound node is
 * one block for each run of AOTX_GEMV_ROWS_CTA rows of the weight tensor.
 * The module value is 1 when a raw module node stands in front of the memory bound node.
 * That node takes a batch of one row, and the memory bound node then exits. */
__global__ void aotx_model_product(const unsigned int *batch, const void *w,
                                   unsigned int type, unsigned int n, unsigned int k,
                                   const half *x, float *y);
__global__ void aotx_model_line(const unsigned int *batch, const void *w, unsigned int type,
                                unsigned int n, unsigned int k, const half *x, float *y,
                                unsigned int module);

/* Fill the descriptor of one model, one thread for each tensor name. The thread forms the
 * name, mixes it as the tensor table does, and finds the tensor. A name the table does not
 * hold gives AOTX_MODEL_ABSENT, one count in missing[0], and its number in missing[1]. */
__global__ void aotx_model_bind(unsigned int role, unsigned int model, unsigned int count,
                                unsigned int *missing);

/* The grid of a matrix launch, from the columns of the weight and the rows of the batch.
 * The tile comes from matrix.cuh, so the graph and the kernel keep one definition. */
__device__ __host__ __forceinline__ unsigned int aotx_model_tiles_n(unsigned int n)
{
    return (n + AOTX_GEMM_TILE_N - 1u) / AOTX_GEMM_TILE_N;
}

__device__ __host__ __forceinline__ unsigned int aotx_model_tiles_m(unsigned int m)
{
    return (m + AOTX_GEMM_TILE_M - 1u) / AOTX_GEMM_TILE_M;
}

/* Read the model files of a directory and fill the descriptor of each role the list names.
 * A null list takes the default list. The return is zero when every role of the list is in
 * the model record and has the tensors it needs. */
int aotx_model_describe(const char *dir, const char *roles);

/* Capture the forward graph of one role for a token count. The buffers are sized here. */
int aotx_model_open(unsigned int role, unsigned int max_tokens);

/* Give the buffers and the graph of one role back. */
void aotx_model_shut(unsigned int role);

/* Set the cache position of every agent slot to zero, and clear the fault count. */
void aotx_model_forget(void);

/* Rows of the last passes that found no cache page. */
unsigned int aotx_model_faulted(void);

/* Nodes of the captured graph of one role. The shape of the graph never changes, so this
 * count is the shape. It holds three nodes before the layers, 14 for each layer, the nodes
 * of the head, and one node after them. */
unsigned int aotx_model_nodes(unsigned int role);

/* Ask the page cache for the pages that the next pass needs. The caller answers the
 * requests with aotx_kv_serve before it calls the pass. */
int aotx_model_pages(unsigned int role, const unsigned int *offset, unsigned int seqs,
                     const unsigned int *agent);

/* Launch one pass. The call block goes to the device, the head node takes the row count,
 * and the graph runs once. The return is zero when the pass ran. */
int aotx_model_launch(unsigned int role, const aotx_model_run *set);

/* One forward pass which takes a token for each sequence from its last row. The token, the
 * seed and the stream position of each draw go to the caller, so a record of the step can
 * name the sample. The draw buffer holds one position for each sequence, or is null. */
int aotx_model_sample(unsigned int role, const int *ids, const unsigned int *offset,
                      unsigned int seqs, const unsigned int *agent,
                      const aotx_model_how *how, int *token, unsigned int *draw,
                      unsigned long long *seed);

/* Set the stream position of every agent slot to zero. The cache positions do not change,
 * so a caller may take the same draw again. */
void aotx_model_restream(void);

#endif
