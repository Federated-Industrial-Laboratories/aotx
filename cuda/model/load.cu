/* Purpose: Judge model commands and keep the queue and resident model rows.
 * Owns: The device model load state and its console report buffer.
 * Launch shape: One thread for a command, a replay apply or a placement mark.
 * Lifetime: From the model file list load to the end of the run. */
#include "model/load.cuh"
#include "cognitive/live.cuh"

#include "model/decode.cuh"
#include "sched/sched.cuh"

__device__ aotx_model_load_state aotx_model_load;
static __device__ aotx_cli_out aotx_model_load_out;

static __device__ const char *aotx_model_role_name[AOTX_MODEL_ROLES] = {
    "embedding", "reranker", "language", "language-q4"
};

static __device__ int aotx_model_bytes_are(const char *left, unsigned int left_len,
                                           const char *right, unsigned int right_max)
{
    unsigned int i = 0u;
    while (i < left_len && i < right_max && right[i] != '\0' && left[i] == right[i]) {
        i += 1u;
    }
    return (i == left_len && i < right_max && right[i] == '\0') ? 1 : 0;
}

static __device__ unsigned int aotx_model_find_name(const char *name, unsigned int length)
{
    for (unsigned int i = 0u; i < aotx_model_load.files; ++i) {
        if (aotx_model_bytes_are(name, length, aotx_model_load.file[i].name,
                                 (unsigned int)sizeof aotx_model_load.file[i].name)) {
            return i;
        }
    }
    return AOTX_MODEL_FILES_MAX;
}

static __device__ unsigned int aotx_model_find_file(const char *name, unsigned int length)
{
    for (unsigned int i = 0u; i < aotx_model_load.files; ++i) {
        if (aotx_model_bytes_are(name, length, aotx_model_load.file[i].file,
                                 (unsigned int)sizeof aotx_model_load.file[i].file)) {
            return i;
        }
    }
    return AOTX_MODEL_FILES_MAX;
}

static __device__ int aotx_model_language(unsigned int role)
{
    return role == AOTX_MODEL_LANGUAGE || role == AOTX_MODEL_LANGUAGE_Q4;
}

static __device__ int aotx_model_live(unsigned int role)
{
    if (aotx_live.intake_mode && aotx_live.phase != AOTX_LIVE_IDLE && aotx_model_is_language(role)) return 1;
    if (role == AOTX_MODEL_EMBEDDING && aotx_live.text_mode && aotx_live.phase != AOTX_LIVE_IDLE) return 1;
    for (unsigned int i = 0u; i < AOTX_SLOTS; ++i) {
        const aotx_seq *seq = &aotx_seqs.slot[i];
        if (seq->state == AOTX_SEQ_STATE_FREE || seq->state == AOTX_SEQ_STATE_DONE) {
            continue;
        }
        if (seq->role == role
            || (aotx_model_language(seq->role) && aotx_model_language(role))) {
            return 1;
        }
    }
    return 0;
}

static __device__ unsigned int aotx_model_slot(unsigned int target)
{
    if (AOTX_MODELS_RESIDENT == 1u && aotx_model_language(target)) {
        for (unsigned int i = 0u; i < AOTX_MODEL_ROLES; ++i) {
            if (aotx_model_load.resident[i].active != 0u
                && aotx_model_language(aotx_model_load.resident[i].slot)) {
                return aotx_model_load.resident[i].slot;
            }
        }
    }
    return target;
}

static __device__ unsigned int aotx_model_role(const char *name, unsigned int length)
{
    for (unsigned int i = 0u; i < AOTX_MODEL_ROLES; ++i) {
        if (aotx_model_bytes_are(name, length, aotx_model_role_name[i], 16u)) {
            return i;
        }
    }
    return AOTX_MODEL_ROLES;
}

static __device__ void aotx_model_copy(char *out, unsigned int max,
                                       const char *in, unsigned int length)
{
    for (unsigned int i = 0u; i < max; ++i) {
        out[i] = (i < length) ? in[i] : '\0';
    }
}

static __device__ void aotx_model_queue(unsigned int source, unsigned int target,
                                        unsigned int slot, unsigned int replayed,
                                        const aotx_model_body *given)
{
    unsigned int at = aotx_model_load.pending_count;
    aotx_model_load_row *row = &aotx_model_load.pending[at];
    row->source = source;
    row->target = target;
    row->slot = slot;
    row->replayed = replayed;
    if (given != 0) {
        row->body = *given;
    } else {
        const aotx_model_file_row *file = &aotx_model_load.file[source];
        row->body.tick = 0ull;
        for (unsigned int i = 0u; i < 32u; ++i) {
            row->body.digest[i] = file->digest[i];
        }
        unsigned int role_len = 0u;
        while (role_len < 16u && aotx_model_role_name[target][role_len] != '\0') {
            role_len += 1u;
        }
        unsigned int file_len = 0u;
        while (file_len < sizeof file->file && file->file[file_len] != '\0') {
            file_len += 1u;
        }
        aotx_model_copy(row->body.role, (unsigned int)sizeof row->body.role,
                        aotx_model_role_name[target], role_len);
        aotx_model_copy(row->body.file, (unsigned int)sizeof row->body.file,
                        file->file, file_len);
    }
    aotx_model_load.pending_count = at + 1u;
}

__device__ void aotx_model_load_command(aotx_cli_out *out, const char *role,
                                        unsigned int role_len, const char *name,
                                        unsigned int name_len, unsigned int extra,
                                        unsigned long long tick)
{
    (void)tick;
    if (aotx_seam.replaying != 0ull) {
        return;
    }
    if (role_len == 0u || name_len == 0u || extra != 0u) {
        aotx_cli_say(out, "model load: give a role and a model name");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    unsigned int target_role = aotx_model_role(role, role_len);
    if (target_role >= AOTX_MODEL_ROLES) {
        aotx_cli_say(out, "model load: there is no role ");
        aotx_cli_add(out, role, role_len);
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    unsigned int source = aotx_model_find_name(name, name_len);
    if (source >= aotx_model_load.files) {
        aotx_cli_say(out, "model load: the model file list has no model ");
        aotx_cli_add(out, name, name_len);
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    unsigned int source_role = aotx_model_load.file[source].role;
    if (target_role != source_role) {
        aotx_cli_say(out, "model load: the manifest holds that name under another role");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    if (aotx_model_live(target_role)) {
        aotx_cli_say(out, "model load: a sequence runs on that role; give stop first");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    if (aotx_model_load.pending_count >= AOTX_MODEL_LOAD_MAX) {
        aotx_cli_say(out, "model load: the tick holds too many model lines; give it again");
        aotx_cli_console(out);
        aotx_cli_count.refused += 1u;
        return;
    }
    aotx_model_queue(source, target_role, aotx_model_slot(target_role), 0u, 0);
    aotx_cli_say(out, "model load: ");
    aotx_cli_add(out, role, role_len);
    aotx_cli_say(out, " takes ");
    aotx_cli_add(out, name, name_len);
    aotx_cli_say(out, " between ticks");
    aotx_cli_console(out);
}

static __device__ void aotx_model_hex(aotx_cli_out *out, unsigned char byte)
{
    char digit[2];
    unsigned int high = (unsigned int)(byte >> 4);
    unsigned int low = (unsigned int)(byte & 15u);
    digit[0] = (char)((high < 10u) ? ('0' + high) : ('a' + high - 10u));
    digit[1] = (char)((low < 10u) ? ('0' + low) : ('a' + low - 10u));
    aotx_cli_add(out, digit, 2u);
}

static __device__ unsigned int aotx_model_text_length(const char *text, unsigned int max)
{
    unsigned int length = 0u;
    while (length < max && text[length] != '\0') {
        length += 1u;
    }
    return length;
}

__device__ void aotx_model_show_command(aotx_cli_out *out)
{
    aotx_cli_say(out, "models: role file sha256 tick");
    aotx_cli_console(out);
    unsigned int shown = 0u;
    for (unsigned int r = 0u; r < AOTX_MODEL_ROLES; ++r) {
        const aotx_model_resident_row *row = &aotx_model_load.resident[r];
        if (row->active == 0u) {
            continue;
        }
        aotx_cli_say(out, "  ");
        aotx_cli_add(out, row->body.role,
                     aotx_model_text_length(row->body.role, 16u));
        aotx_cli_say(out, " ");
        aotx_cli_add(out, row->body.file,
                     aotx_model_text_length(row->body.file, 64u));
        aotx_cli_say(out, " ");
        for (unsigned int i = 0u; i < 32u; ++i) {
            aotx_model_hex(out, row->body.digest[i]);
        }
        aotx_cli_say(out, " ");
        aotx_cli_num(out, row->body.tick);
        aotx_cli_console(out);
        shown += 1u;
    }
    if (shown == 0u) {
        aotx_cli_say(out, "  no resident models");
        aotx_cli_console(out);
    }
}

static __device__ void aotx_model_stall(void)
{
    aotx_stall_body body;
    body.host_ring_free = aotx_sched.free_bytes;
    body.held_count = aotx_sched.held_count | AOTX_STALL_MODEL_LOAD;
    aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_STALL, 0u,
                    &body, (unsigned int)sizeof body);
}

static __device__ void aotx_model_hold_line(const char *tail, unsigned long long bytes)
{
    aotx_cli_out *out = &aotx_model_load_out;
    aotx_cli_clear(out);
    aotx_cli_say(out, "model load: ");
    aotx_cli_say(out, tail);
    if (bytes != 0ull) {
        aotx_cli_say(out, " ");
        aotx_cli_num(out, bytes >> 20);
        aotx_cli_say(out, " MB placed");
    }
    aotx_console_write(out->text, out->at);
    aotx_cli_clear(out);
}

__global__ void aotx_model_load_begin(void)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u || aotx_model_load.pending_count == 0u) {
        return;
    }
    aotx_sched.held_count += 1ull;
    aotx_model_stall();
    aotx_model_hold_line("the tick holds for placement", 0ull);
}

static __device__ void aotx_model_resident(const aotx_model_load_row *load)
{
    if (AOTX_MODELS_RESIDENT == 1u && aotx_model_language(load->target)) {
        for (unsigned int i = 0u; i < AOTX_MODEL_ROLES; ++i) {
            if (aotx_model_language(aotx_model_load.resident[i].slot)) {
                aotx_model_load.resident[i].active = 0u;
            }
        }
    }
    aotx_model_resident_row *row = &aotx_model_load.resident[load->slot];
    row->body = load->body;
    row->source = load->source;
    row->slot = load->slot;
    row->active = 1u;
}

static __device__ void aotx_model_pop(void)
{
    for (unsigned int i = 1u; i < aotx_model_load.pending_count; ++i) {
        aotx_model_load.pending[i - 1u] = aotx_model_load.pending[i];
    }
    if (aotx_model_load.pending_count != 0u) {
        aotx_model_load.pending_count -= 1u;
    }
}

__global__ void aotx_model_load_finish(unsigned int success, unsigned int reason,
                                       unsigned long long bytes)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u || aotx_model_load.pending_count == 0u) {
        return;
    }
    aotx_model_load_row load = aotx_model_load.pending[0];
    if (success != 0u) {
        aotx_model_load.placed_bytes += bytes;
        if (load.replayed != 0u) {
            aotx_model_resident(&load);
        } else {
            aotx_model_load.placed = load;
            aotx_model_load.placed_ready = 1u;
        }
        aotx_model_hold_line("placement is complete;", bytes);
    } else {
        aotx_model_load.refused += 1u;
        const char *why = (reason == AOTX_MODEL_LOAD_DIGEST) ? "the digest differs"
                        : (reason == AOTX_MODEL_LOAD_REGION) ? "the weights region is full"
                        : (reason == AOTX_MODEL_LOAD_DESC) ? "the descriptor was refused"
                        : "the file did not read";
        aotx_model_hold_line(why, 0ull);
    }
    aotx_model_stall();
    aotx_model_pop();
}

__device__ int aotx_model_load_apply(const aotx_model_body *body)
{
    unsigned int role_len = 0u;
    while (role_len < sizeof body->role && body->role[role_len] != '\0') {
        role_len += 1u;
    }
    unsigned int file_len = 0u;
    while (file_len < sizeof body->file && body->file[file_len] != '\0') {
        file_len += 1u;
    }
    unsigned int target = aotx_model_role(body->role, role_len);
    unsigned int source = aotx_model_find_file(body->file, file_len);
    if (target >= AOTX_MODEL_ROLES || source >= aotx_model_load.files
        || aotx_model_load.pending_count >= AOTX_MODEL_LOAD_MAX) {
        aotx_model_load.refused += 1u;
        aotx_model_load.replay_bad = 1u;
        return 1;
    }
    for (unsigned int i = 0u; i < 32u; ++i) {
        if (body->digest[i] != aotx_model_load.file[source].digest[i]) {
            aotx_model_load.refused += 1u;
            aotx_model_load.replay_bad = 1u;
            return 1;
        }
    }
    unsigned int role = target;
    unsigned int source_role = aotx_model_load.file[source].role;
    if (role != source_role) {
        aotx_model_load.refused += 1u;
        aotx_model_load.replay_bad = 1u;
        return 1;
    }
    aotx_model_queue(source, role, aotx_model_slot(role), 1u, body);
    aotx_sched.held = 1ull;
    aotx_sched.held_count += 1ull;
    aotx_model_stall();
    aotx_model_hold_line("the replay holds for placement", 0ull);
    return 0;
}

__device__ void aotx_model_load_commit(unsigned long long tick)
{
    if (aotx_model_load.placed_ready == 0u) {
        return;
    }
    aotx_model_load.placed.body.tick = tick;
    unsigned long long seq = aotx_seam_claim(1u);
    aotx_record_header *header = aotx_seam_slot(seq);
    aotx_model_body *body = (aotx_model_body *)aotx_seam_body(header);
    *body = aotx_model_load.placed.body;
    aotx_seam_publish(header, seq, AOTX_WRITER_CONSOLE, AOTX_CLASS_A, AOTX_REC_MODEL,
                      0u, (unsigned int)sizeof *body);
    aotx_seam.apply.state_hash = aotx_seam_fnv1a(aotx_seam.apply.state_hash,
                                                 aotx_seam_body_of(seq),
                                                 (unsigned int)sizeof *body);
    aotx_seam.apply.applied_count += 1ull;
    aotx_model_resident(&aotx_model_load.placed);
    aotx_model_load.placed_ready = 0u;
}
