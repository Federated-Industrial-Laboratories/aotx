/* Purpose: Compute the forward pass of a model.
 * Owns: The model descriptors and the layer parameters.
 * Launch shape: One block for each tile; the batch is the number of tokens of all sequences.
 * Lifetime: From model load to the end of the run. */
#ifndef MODEL_CUH
#define MODEL_CUH

#include <cuda_fp16.h>

#include "model/names.h"
#include "seam/wire.h"

/* The model roles, in the order the manifest names them. The fourth is the language model
 * at Q4_0, loaded only when the caller names it. */
#define AOTX_MODEL_EMBEDDING   0u
#define AOTX_MODEL_RERANKER    1u
#define AOTX_MODEL_LANGUAGE    2u
#define AOTX_MODEL_LANGUAGE_Q4 3u
#define AOTX_MODEL_ROLES       4u

/* Weight block types, as the model file names them. */
#define AOTX_WEIGHT_F32        0u
#define AOTX_WEIGHT_F16        1u
#define AOTX_WEIGHT_Q4_0       2u
#define AOTX_WEIGHT_Q8_0       8u
#define AOTX_WEIGHT_Q4_K       12u
#define AOTX_WEIGHT_Q5_K       13u
#define AOTX_WEIGHT_Q6_K       14u

/* The device offsets of one layer's tensors, in bytes from the start of the weights region.
 * A tensor the model does not have holds AOTX_MODEL_ABSENT. */
#define AOTX_MODEL_ABSENT      (~0ull)

/* The rule that pairs the elements of a head for the rotary turn. The split rule pairs
 * element i with element i plus half the head. The adjacent rule pairs element 2i with the
 * element after it. The architecture of the file selects the rule. */
#define AOTX_ROPE_PAIRS_SPLIT    0u
#define AOTX_ROPE_PAIRS_ADJACENT 1u

typedef struct aotx_model_layer {
    unsigned long long attn_norm;   /* [hidden] F32 */
    unsigned long long attn_q;      /* [heads * head_dim][hidden] */
    unsigned long long attn_k;      /* [kv_heads * head_dim][hidden] */
    unsigned long long attn_v;      /* [kv_heads * head_dim][hidden] */
    unsigned long long attn_o;      /* [hidden][heads * head_dim] */
    unsigned long long attn_q_norm; /* [head_dim] F32 */
    unsigned long long attn_k_norm; /* [head_dim] F32 */
    unsigned long long ffn_norm;    /* [hidden] F32 */
    unsigned long long ffn_gate;    /* [ffn][hidden] */
    unsigned long long ffn_up;      /* [ffn][hidden] */
    unsigned long long ffn_down;    /* [hidden][ffn] */
    unsigned long long ffn_router;  /* [experts][hidden] F32 */
} aotx_model_layer;

/* One model: its shape, its block types and where its tensors are. The host glue fills it
 * from the model file's metadata and the tensor table; the device reads it. */
typedef struct aotx_model_desc {
    unsigned int role;
    unsigned int layers;
    unsigned int probe_layer;       /* layer derived from the store fraction */
    unsigned int hidden;            /* embedding_length */
    unsigned int ffn;               /* feed_forward_length */
    unsigned int heads;             /* attention.head_count */
    unsigned int kv_heads;          /* attention.head_count_kv */
    unsigned int head_dim;          /* attention.key_length; value_length is equal */
    unsigned int vocab;             /* rows of token_embd */
    unsigned int context;           /* context_length */
    unsigned int weight_type;       /* block type of the projection tensors */
    unsigned int embd_type;         /* block type of token_embd */
    unsigned int tied_output;       /* 1 when output is token_embd */
    unsigned int pooling;           /* the file's pooling_type, 0 when absent */
    float rope_theta;               /* rope.freq_base */
    unsigned char kind[AOTX_MODEL_MAX_LAYERS]; /* one kind for each layer */
    float rms_eps;                  /* attention.layer_norm_rms_epsilon */
    unsigned int rope_pairs;        /* AOTX_ROPE_PAIRS_SPLIT or AOTX_ROPE_PAIRS_ADJACENT */
    unsigned int expert_count;      /* expert_count, zero for dense layers */
    unsigned int expert_used_count; /* expert_used_count */
    unsigned long long token_embd;  /* [vocab][hidden] */
    unsigned long long output_norm; /* [hidden] F32 */
    unsigned long long output;      /* [vocab][hidden], or AOTX_MODEL_ABSENT when tied */
    unsigned long long cls_output;  /* [2][hidden] for the reranker, else AOTX_MODEL_ABSENT */
    unsigned long long rope_freqs;  /* [head_dim / 2] F32 angle divisors, or AOTX_MODEL_ABSENT */
    aotx_model_layer layer[AOTX_MODEL_MAX_LAYERS];
    /* The block type of each layer tensor, by layer and slot. A file may hold one tensor
     * of a layer in a type that differs from the type of the other tensors. */
    unsigned char layer_type[AOTX_MODEL_MAX_LAYERS][AOTX_LAYER_TENSOR_SLOTS];
} aotx_model_desc;

extern __device__ aotx_model_desc aotx_model[AOTX_MODEL_ROLES];

/* Matrix kernels. The weight w is a row-major [n][k] tensor in the block type given. The
 * input x holds m rows of k activations in half precision. The output y receives m rows of n
 * sums in single precision. The batch m is the number of tokens. There is no path for one
 * token that differs in kind from the path for many.
 *
 * The caller gives k as a multiple of 32, the block length of the quantized types. A K
 * type holds 256 weights in a super block, so its k is a multiple of 256. */

/* Tensor-core product for m of 16 or more: one block computes one tile of y. */
__global__ void aotx_model_gemm(const void *w, unsigned int type, unsigned int n,
                                unsigned int k, const half *x, unsigned int m, float *y);

/* Memory-bound product for small m: one warp reads a run of rows of W once and applies
 * them to every row of x. */
__global__ void aotx_model_gemv(const void *w, unsigned int type, unsigned int n,
                                unsigned int k, const half *x, unsigned int m, float *y);

/* Dequantize rows of W to half precision, for tests and for the token embedding gather. */
__global__ void aotx_model_dequant(const void *w, unsigned int type, unsigned int k,
                                   unsigned int first_row, unsigned int rows, half *out);

/* The forward pass over a batch of sequences. The array ids holds every sequence's token ids
 * one after the other. The entry offset[s] is the first token of sequence s, and offset[seqs]
 * is the total. The entry agent[s] names the key value cache slot the sequence writes.
 *
 * The buffer logits, when given, receives the vocabulary row of every token. The buffer pooled,
 * when given, receives one hidden row for each sequence after pooling and normalization. The
 * pass is a captured graph launched by the host glue in cuda/model/model_host.cu. Returns 0
 * on success. */
int aotx_model_prefill(unsigned int role, const int *ids, const unsigned int *offset,
                       unsigned int seqs, const unsigned int *agent, float *logits,
                       float *pooled);

/* The reranker's score for each sequence: the probability of "yes" against "no" at the last
 * position, from the cls_output rows. Returns 0 on success. */
int aotx_model_rerank(const int *ids, const unsigned int *offset, unsigned int seqs,
                      const unsigned int *agent, float *score);

/* Build one descriptor in a target role from the named entry of a model file list. */
int aotx_model_describe_one(const char *dir, const char *name, unsigned int target);

#endif
