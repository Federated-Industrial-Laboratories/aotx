/* Purpose: Turn the blocks of a weight tensor into half values, one row at a time.
 * Owns: Nothing; the caller owns the input and the output.
 * Launch shape: One block for each row of the batch; the threads take the columns.
 * Lifetime: One launch. */
#include "model/matrix.cuh"

/* The rows of one launch. A block takes one row and steps by the grid. A grid smaller than
 * the batch is therefore correct, and a grid larger than the batch does no work. */
template <unsigned int TYPE>
__device__ __forceinline__ void aotx_dequant_rows(const unsigned char *w, unsigned int k,
                                                  unsigned int first_row, unsigned int rows,
                                                  half *out)
{
    unsigned long long stride = aotx_matrix_row_bytes(TYPE, k);
    for (unsigned int r = blockIdx.x; r < rows; r += gridDim.x) {
        const unsigned char *row = w + stride * (unsigned long long)(first_row + r);
        half *line = out + (size_t)r * k;
        for (unsigned int j = threadIdx.x; j < k; j += blockDim.x) {
            line[j] = aotx_matrix_weight<TYPE>(row, j);
        }
    }
}

/* The dispatch by block type. A type the reader does not know writes nothing. A caller
 * which gives a wrong type therefore sees an unchanged output and not a wrong one. */
__global__ void aotx_model_dequant(const void *w, unsigned int type, unsigned int k,
                                   unsigned int first_row, unsigned int rows, half *out)
{
    const unsigned char *base = (const unsigned char *)w;
    switch (type) {
    case AOTX_WEIGHT_Q8_0:
        aotx_dequant_rows<AOTX_WEIGHT_Q8_0>(base, k, first_row, rows, out);
        break;
    case AOTX_WEIGHT_Q4_0:
        aotx_dequant_rows<AOTX_WEIGHT_Q4_0>(base, k, first_row, rows, out);
        break;
    case AOTX_WEIGHT_Q4_K:
        aotx_dequant_rows<AOTX_WEIGHT_Q4_K>(base, k, first_row, rows, out);
        break;
    case AOTX_WEIGHT_Q5_K:
        aotx_dequant_rows<AOTX_WEIGHT_Q5_K>(base, k, first_row, rows, out);
        break;
    case AOTX_WEIGHT_Q6_K:
        aotx_dequant_rows<AOTX_WEIGHT_Q6_K>(base, k, first_row, rows, out);
        break;
    case AOTX_WEIGHT_F16:
        aotx_dequant_rows<AOTX_WEIGHT_F16>(base, k, first_row, rows, out);
        break;
    case AOTX_WEIGHT_F32:
        aotx_dequant_rows<AOTX_WEIGHT_F32>(base, k, first_row, rows, out);
        break;
    default:
        break;
    }
}
