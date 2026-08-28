/* Purpose: Write one typed message as a record, with the writer and the sequence stamped.
 * Owns: The bus state, the sequence table of every writer, and the message buffer.
 * Launch shape: One thread for each message.
 * Lifetime: The whole run. */
#include "bus/bus.cuh"
#include "seam/seam.cuh"

/* The table starts at zero, so the first message of a writer carries writer_seq 1. */
__device__ aotx_bus_state aotx_bus;

/* The buffer of the last messages, which the bus panel reads. */
__device__ aotx_bus_buffer aotx_bus_lines;

/* Put one message in the buffer. A writer claims a message number, fills the line, and then
 * publishes the number. A reader that sees the number therefore sees the whole line. */
static __device__ __forceinline__ void aotx_bus_put(unsigned long long seq,
                                                    unsigned int writer, unsigned int kind,
                                                    unsigned int provenance,
                                                    const char *text, unsigned int length)
{
    unsigned long long at = atomicAdd(&aotx_bus_lines.count, 1ull) + 1ull;
    aotx_bus_line *line = &aotx_bus_lines.line[(at - 1ull) & (AOTX_BUS_LINES - 1u)];
    aotx_seam_release_gpu(&line->at, 0ull);
    line->seq = seq;
    line->writer = writer;
    line->kind = (unsigned char)kind;
    line->provenance = (unsigned char)provenance;
    for (unsigned int i = 0u; i < length; ++i) {
        line->text[i] = text[i];
    }
    line->text_len = length;
    aotx_seam_release_gpu(&line->at, at);
}

/* The rules that a message must meet. A finding states where its content came from, and no
 * other kind carries a provenance value. A writer with no entry in the table is refused. */
static __device__ __forceinline__ int aotx_bus_takes(unsigned int writer, unsigned int kind,
                                                     unsigned int provenance)
{
    /* A writer is a system writer that the derived file names, or an agent slot. An
     * identity between the two has no name on the disk side, so it is refused here. */
    if (writer >= AOTX_BUS_WRITER_MAX) {
        return 0;
    }
    if (writer > AOTX_WRITER_CONSOLE && writer < AOTX_WRITER_AGENT_BASE) {
        return 0;
    }
    if (kind < AOTX_BUS_FINDING || kind > AOTX_BUS_NOTE) {
        return 0;
    }
    if (kind == AOTX_BUS_FINDING) {
        return (provenance >= AOTX_PROV_COMPUTED && provenance <= AOTX_PROV_TESTIMONY);
    }
    return provenance == 0u;
}

__device__ unsigned long long aotx_bus_append(unsigned int writer, unsigned int kind,
                                              unsigned int provenance, const char *text,
                                              unsigned int length, unsigned long long re_seq,
                                              unsigned long long corrects_seq, float score,
                                              unsigned long long tick)
{
    if (!aotx_bus_takes(writer, kind, provenance)) {
        atomicAdd(&aotx_bus.refused, 1ull);
        return 0ull;
    }
    if (text == 0) {
        length = 0u;
    }
    if (length > AOTX_BUS_TEXT_BYTES) {
        length = AOTX_BUS_TEXT_BYTES;
    }

    /* The count of the writer comes from the table and not from the caller, so two writers
     * cannot give the same message number. */
    unsigned int mine = atomicAdd(&aotx_bus.writer_seq[writer], 1u) + 1u;
    unsigned long long seq = aotx_seam_claim(1u);
    aotx_record_header *header = aotx_seam_slot(seq);
    aotx_bus_body *body = (aotx_bus_body *)aotx_seam_body(header);
    body->kind = (unsigned char)kind;
    body->provenance = (unsigned char)provenance;
    body->reserved0 = 0u;
    body->writer_seq = mine;
    body->re_seq = re_seq;
    body->corrects_seq = corrects_seq;

    /* Only a rank carries a score, and a score stays in the range the schema gives. */
    float value = 0.0f;
    if (kind == AOTX_BUS_RANK) {
        value = (score < 0.0f) ? 0.0f : ((score > 1.0f) ? 1.0f : score);
    }
    body->score = value;
    body->text_len = length;
    for (unsigned int i = 0u; i < length; ++i) {
        body->text[i] = text[i];
    }
    for (unsigned int i = length; i < AOTX_BUS_TEXT_BYTES; ++i) {
        body->text[i] = 0;   /* the text of an earlier record in this slot does not survive */
    }

    /* The body length counts the fixed fields and the text bytes that carry data. */
    aotx_seam_publish_at(header, seq, writer, AOTX_CLASS_B, AOTX_REC_BUS, 0u,
                         32u + length, tick);
    /* The buffer takes the same message, so the panel keeps it after the ring drops it. */
    aotx_bus_put(seq, writer, kind, provenance, text, length);
    atomicAdd(&aotx_bus.appended, 1ull);
    return seq;
}
