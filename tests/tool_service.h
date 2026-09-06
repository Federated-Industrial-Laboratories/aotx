/* Purpose: Check memory readiness and the records of completed tool requests.
 * Owns: The request fixtures and their record checks.
 * Launch shape: One thread for each request; the production tool step publishes results.
 * Lifetime: One run of tool_test.cu. */
#ifndef AOTX_TEST_TOOL_SERVICE_H
#define AOTX_TEST_TOOL_SERVICE_H

#include "tool/module.cuh"

/* Compare every published part with the completed request, including empty results. */
static unsigned int aotx_tool_service_records(unsigned int count, unsigned int cls)
{
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    aotx_seam_state seam;
    unsigned int done[AOTX_SLOTS];
    unsigned int wrong = 0u;
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(done, aotx_tool_done, sizeof done),
                       "cudaMemcpyFromSymbol");
    for (unsigned int slot = 0u; slot < count; ++slot) {
        const aotx_request *hold = &table->slot[slot];
        unsigned int parts = (hold->result_len + AOTX_TOOL_REPLY_BYTES - 1u)
                           / AOTX_TOOL_REPLY_BYTES;
        if (parts == 0u) parts = 1u;
        if (done[slot] == 0u || hold->result_seq == 0ull) {
            wrong += 1u;
            continue;
        }
        for (unsigned int part = 0u; part < parts; ++part) {
            unsigned char record[AOTX_SLOT_BYTES];
            unsigned long long seq = hold->result_seq + part;
            const unsigned char *at = seam.dev.base
                + ((seq - 1ull) & seam.dev.mask) * AOTX_SLOT_BYTES;
            aotx_check_runtime(cudaMemcpy(record, at, sizeof record, cudaMemcpyDeviceToHost),
                               "cudaMemcpy");
            const aotx_record_header *header = (const aotx_record_header *)record;
            const aotx_tool_reply_body *body =
                (const aotx_tool_reply_body *)(record + AOTX_HEADER_BYTES);
            unsigned int offset = part * AOTX_TOOL_REPLY_BYTES;
            unsigned int bytes = hold->result_len - offset;
            if (bytes > AOTX_TOOL_REPLY_BYTES) bytes = AOTX_TOOL_REPLY_BYTES;
            unsigned int status = (part + 1u == parts) ? hold->status : AOTX_TOOL_OK;
            if (header->seq != seq || header->cls != cls
                || header->type != AOTX_REC_TOOL_REPLY || header->body_len != sizeof *body
                || ((header->flags & AOTX_FLAG_REPLAY) != 0u) != (seam.replaying != 0ull)
                || body->agent != slot || body->request != hold->request
                || header->writer != ((cls == AOTX_CLASS_A && hold->status != AOTX_TOOL_LATE)
                    ? AOTX_WRITER_FEEDER : AOTX_WRITER_AGENT_BASE + slot)
                || body->part != part || body->parts != parts || body->status != status
                || body->len != bytes || memcmp(body->bytes, hold->result + offset, bytes)) {
                wrong += 1u;
            }
        }
    }
    free(table);
    return wrong;
}

static void aotx_tool_service_check(unsigned int wrong, const char *what,
                                     unsigned int count, unsigned int *applied,
                                     unsigned int *failed)
{
    *applied += 1u;
    if (wrong != 0u) {
        printf("tool: %s at %u requests has %u wrong results\n", what, count, wrong);
        *failed += 1u;
    }
}

/* Read the real advertisement and admit an unlisted but permitted memory call. */
__global__ void aotx_tool_service_admit(unsigned int count, unsigned int role,
                                         unsigned int tool, unsigned int ready,
                                         unsigned char *lists, unsigned int *wrong,
                                         aotx_tool_call *calls)
{
    unsigned int slot = threadIdx.x;
    if (slot >= count) return;
    unsigned int length = aotx_catalog_tool_list(lists + slot * AOTX_SAY_BYTES, 0u, role);
    if (length < AOTX_SAY_BYTES) lists[slot * AOTX_SAY_BYTES + length] = '\0';
    const char *text = (tool == AOTX_TOOL_MEMORY_WRITE)
        ? "<tool_call>{\"name\":\"memory_write\",\"arguments\":{\"provenance\":\"computed\",\"text\":\"a note\"}}</tool_call>"
        : "<tool_call>{\"name\":\"memory_recall\",\"arguments\":{\"text\":\"a note\"}}</tool_call>";
    unsigned int bytes = 0u;
    while (text[bytes] != '\0') bytes += 1u;
    aotx_tool_call *call = &calls[slot];
    unsigned int bad = (aotx_tool_parse((const unsigned char *)text, bytes, call) != 1);
    bad += (aotx_catalog_may_call(role, call->entry) == 0);
    bad += (aotx_tool_available(call->entry) != (int)ready);
    if (ready == 0u && bad == 0u) {
        unsigned int expected = aotx_tool_embed.made[slot] * AOTX_SLOTS + slot + 1u;
        unsigned int id = aotx_tool_request(slot, call, 1u, 50ull);
        aotx_request *hold = &aotx_requests.slot[slot];
        bad += (id != expected || hold->request != expected || aotx_tool_done[slot] != 1u
                || hold->status != AOTX_TOOL_ERROR || hold->auth != AOTX_AUTH_NONE
                || hold->entry != call->entry || hold->arg_len == 0u
                || aotx_tool_embed.state[slot] != AOTX_TOOL_EMBED_NONE);
    }
    wrong[slot] = bad;
}

static void aotx_tool_service_availability(unsigned int count, unsigned int ready,
                                            unsigned int *applied, unsigned int *failed)
{
    unsigned int role = aotx_test_catalog_entry("service_role", AOTX_MODULE_ROLE);
    if (role >= AOTX_MODULE_SLOTS) {
        aotx_test_module module;
        aotx_test_module_text(&module, AOTX_MODULE_ROLE, "service_role",
            "kind: role\nname: service_role\nmodel: language\n"
            "tools: memory_write,memory_recall,skill_use,fs_read\nbody: overlay.txt\n",
            "Use the tools when needed.");
        aotx_test_import_direct(&module, 900u);
        aotx_test_module_free(&module);
        role = aotx_test_catalog_entry("service_role", AOTX_MODULE_ROLE);
        if (role >= AOTX_MODULE_SLOTS) {
            aotx_tool_service_check(1u, "memory role import", count, applied, failed);
            return;
        }
    }
    unsigned char *lists = (unsigned char *)aotx_tool_test_take(count * AOTX_SAY_BYTES);
    unsigned int *wrong = (unsigned int *)aotx_tool_test_take(count * sizeof(unsigned int));
    aotx_tool_call *calls = (aotx_tool_call *)aotx_tool_test_take(count * sizeof *calls);
    unsigned int marks[AOTX_SLOTS];
    char *text = (char *)calloc(count * AOTX_SAY_BYTES, 1u);
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    const unsigned int tools[2] = { AOTX_TOOL_MEMORY_RECALL, AOTX_TOOL_MEMORY_WRITE };
    for (unsigned int k = 0u; k < 2u; ++k) {
        aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
        aotx_tool_service_admit<<<1, AOTX_SLOTS>>>(count, role, tools[k], ready,
                                                   lists, wrong, calls);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_check_runtime(cudaMemcpy(marks, wrong, count * sizeof(unsigned int),
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpy(text, lists, count * AOTX_SAY_BYTES,
                                      cudaMemcpyDeviceToHost), "cudaMemcpy");
        aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                           "cudaMemcpyFromSymbol");
        unsigned int bad = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            const char *list = text + i * AOTX_SAY_BYTES;
            bad += marks[i];
            bad += ((strstr(list, "\"name\": \"memory_write\"") != NULL) != (ready != 0u));
            bad += ((strstr(list, "\"name\": \"memory_recall\"") != NULL) != (ready != 0u));
            bad += (strstr(list, "\"name\": \"skill_use\"") == NULL
                    || strstr(list, "\"name\": \"fs_read\"") == NULL);
            if (ready == 0u) {
                table->slot[i].result[table->slot[i].result_len] = '\0';
                bad += (strstr(table->slot[i].result, "install an embedding model") == NULL
                        || strstr(table->slot[i].result, "--roles embedding") == NULL);
            }
        }
        if (ready == 0u) {
            unsigned long long before, after;
            unsigned long long hash = aotx_tool_test_hash(&before);
            aotx_tool_step<<<1, AOTX_SLOTS>>>(50ull);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            bad += aotx_tool_service_records(count, AOTX_CLASS_B);
            bad += (aotx_tool_test_hash(&after) != hash || after - before != count);
            aotx_tool_step<<<1, AOTX_SLOTS>>>(51ull);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            aotx_tool_test_hash(&before);
            bad += (before != after);
        }
        aotx_tool_service_check(bad, "memory readiness", count, applied, failed);
    }
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
    cudaFree(lists);
    cudaFree(wrong);
    cudaFree(calls);
    free(text);
    free(table);
}

/* Set the output of a device module at its existing handoff to the tool step. */
__global__ void aotx_tool_service_complete(unsigned int count, unsigned int mode,
                                            unsigned int module, aotx_tool_call *calls)
{
    unsigned int slot = threadIdx.x;
    if (slot >= count) return;
    aotx_tool_call *call = &calls[slot];
    call->tool = (mode < 3u) ? AOTX_TOOL_SKILL_USE : AOTX_TOOL_FS_READ;
    call->entry = aotx_catalog_built_entry(call->tool);
    const char *name = (mode == 1u) ? "no_service_skill" : "service_skill";
    while (name[call->arg_len] != '\0') {
        call->arg[call->arg_len] = name[call->arg_len];
        call->arg_len += 1u;
    }
    if (mode >= 6u && mode <= 9u) {
        call->tool = AOTX_TOOL_NONE;
        call->entry = module;
    }
    if (mode == 4u) {
        aotx_tool_over_request(slot, call, 50ull);
        return;
    }
    if (mode == 5u) {
        aotx_tool_outcome_arm(slot, AOTX_TOOL_ERROR);
        aotx_tool_outcome_request(slot, call, 50ull);
        return;
    }
    unsigned int id = aotx_tool_request(slot, call, (mode == 3u), 50ull);
    if (mode == 3u) aotx_agent_authorize(id, 0u, 50ull);
    if (mode >= 6u && mode <= 8u) {
        aotx_tool_modules.head[slot].done = 1u;
        aotx_tool_modules.head[slot].length = (mode == 6u) ? 0u : AOTX_TOOL_RESULT_BYTES;
        aotx_tool_modules.head[slot].status = (mode == 8u) ? AOTX_TOOL_STATUS_ERROR
                                                                        : AOTX_TOOL_STATUS_OK;
        for (unsigned int i = 0u; i < aotx_tool_modules.head[slot].length; ++i) {
            aotx_tool_modules.text[slot * AOTX_TOOL_RESULT_BYTES + i] = (char)('a' + slot % 26u);
        }
    }
    if (mode == 10u) {
        aotx_tool_reply_body body = {};
        body.agent = slot;
        body.request = id;
        body.parts = 1u;
        body.len = 1u;
        body.bytes[0] = 'h';
        unsigned long long seq = aotx_seam_write(AOTX_WRITER_FEEDER,
            AOTX_CLASS_A, AOTX_REC_TOOL_REPLY, 0u, &body, sizeof body);
        aotx_tool_reply_apply(&body, seq);
    }
    if (mode == 11u) aotx_requests.slot[slot].deadline = 0ull;
}

static void aotx_tool_service_shapes(unsigned int count, unsigned int *applied,
                                      unsigned int *failed)
{
    aotx_test_module module;
    char body[3u * AOTX_TOOL_REPLY_BYTES + 8u];
    memset(body, 's', sizeof body - 1u);
    body[sizeof body - 1u] = '\0';
    aotx_test_module_text(&module, AOTX_MODULE_TOOL, "service_module",
        "kind: tool\nname: service_module\nside: device\narguments: text\n"
        "module: service.ptx\nentry: aotx_tool_service\n"
        "sha256: 0000000000000000000000000000000000000000000000000000000000000000\n", NULL);
    aotx_test_import_direct(&module, 901u);
    aotx_test_module_free(&module);
    unsigned int entry = aotx_test_catalog_entry("service_module", AOTX_MODULE_TOOL);
    aotx_seam_set_replaying(0);
    for (unsigned int mode = 0u; mode < 12u; ++mode) {
        aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
        if (mode == 0u || mode == 2u) {
            aotx_test_module_text(&module, AOTX_MODULE_SKILL, "service_skill",
                "kind: skill\nname: service_skill\ndescription: A tool result check.\n",
                (mode == 0u) ? body : "one skill result");
            aotx_test_import_direct(&module, 902u + mode);
            aotx_test_module_free(&module);
        }
        unsigned int nodes = (mode == 9u) ? 0u : 1u;
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_modules, &nodes, sizeof nodes,
                            offsetof(aotx_tool_module_state, nodes)), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_modules, &entry, sizeof entry,
                            offsetof(aotx_tool_module_state, entry)), "cudaMemcpyToSymbol");
        aotx_catalog_state *catalog = aotx_test_catalog_read();
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_modules, &catalog->count.device_gen,
                            sizeof(unsigned int), offsetof(aotx_tool_module_state, gen)),
                            "cudaMemcpyToSymbol");
        free(catalog);
        aotx_tool_call *calls = (aotx_tool_call *)aotx_tool_test_take(count * sizeof *calls);
        aotx_tool_service_complete<<<1, AOTX_SLOTS>>>(count, mode, entry, calls);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        unsigned long long before, after;
        aotx_seam_set_replaying(mode == 6u);
        unsigned long long hash = aotx_tool_test_hash(&before);
        aotx_tool_step<<<1, AOTX_SLOTS>>>(51ull);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        unsigned int wrong = aotx_tool_service_records(count,
            (mode >= 10u) ? AOTX_CLASS_A : AOTX_CLASS_B);
        unsigned long long next_hash = aotx_tool_test_hash(&after);
        if (mode != 11u) wrong += (hash != next_hash);
        if (mode == 10u) wrong += (before != after);
        aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
        aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                           "cudaMemcpyFromSymbol");
        unsigned int status = (mode == 1u || mode == 4u || mode == 5u
                              || mode == 8u || mode == 9u) ? AOTX_TOOL_ERROR
                            : (mode == 3u) ? AOTX_TOOL_REFUSED
                            : (mode == 11u) ? AOTX_TOOL_LATE : AOTX_TOOL_OK;
        unsigned int expected = 0u;
        for (unsigned int i = 0u; i < count; ++i) {
            const aotx_request *hold = &table->slot[i];
            wrong += (hold->status != status);
            if (mode == 0u) wrong += (hold->result_len != sizeof body - 1u
                                      || memcmp(hold->result, body, sizeof body - 1u));
            if (mode == 2u) wrong += (hold->result_len != 16u
                                      || memcmp(hold->result, "one skill result", 16u));
            if (mode == 6u) wrong += (hold->result_len != 0u);
            if (mode == 7u || mode == 8u) wrong += (hold->result_len != AOTX_TOOL_RESULT_BYTES);
            unsigned int parts = (hold->result_len + AOTX_TOOL_REPLY_BYTES - 1u)
                               / AOTX_TOOL_REPLY_BYTES;
            expected += (parts != 0u) ? parts : 1u;
        }
        if (mode != 10u) wrong += (after - before != expected);
        aotx_tool_step<<<1, AOTX_SLOTS>>>(52ull);
        aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        aotx_tool_test_hash(&before);
        wrong += (before != after);
        char label[64];
        snprintf(label, sizeof label, "completion shape %u", mode);
        aotx_tool_service_check(wrong, label, count, applied, failed);
        free(table);
        cudaFree(calls);
    }
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
    unsigned int zero = 0u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_modules, &zero, sizeof zero,
                        offsetof(aotx_tool_module_state, nodes)), "cudaMemcpyToSymbol");
}

/* The real planner must complete a request whose page allowance is exhausted. */
__global__ void aotx_tool_service_no_pages(unsigned int count, aotx_tool_call *calls)
{
    unsigned int slot = threadIdx.x;
    if (slot >= count) return;
    aotx_tool_call *call = &calls[slot];
    call->tool = AOTX_TOOL_MEMORY_WRITE;
    call->entry = aotx_catalog_built_entry(call->tool);
    call->arg_len = 1u;
    call->arg[0] = 'x';
    aotx_tool_request(slot, call, 0u, 50ull);
    aotx_kv_release(slot);
    aotx_tool_gear.count[slot] = 1u;
    aotx_tool_embed.starved[slot] = AOTX_TOOL_ASK_LIMIT;
}

static void aotx_tool_service_starved(unsigned int count, unsigned int *applied,
                                       unsigned int *failed)
{
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
    aotx_tool_call *calls = (aotx_tool_call *)aotx_tool_test_take(count * sizeof *calls);
    aotx_tool_service_no_pages<<<1, AOTX_SLOTS>>>(count, calls);
    aotx_tool_plan<<<1, AOTX_SLOTS>>>(50ull);
    aotx_tool_step<<<1, AOTX_SLOTS>>>(50ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int wrong = aotx_tool_service_records(count, AOTX_CLASS_B);
    aotx_request_table *table = (aotx_request_table *)calloc(1, sizeof *table);
    aotx_check_runtime(cudaMemcpyFromSymbol(table, aotx_requests, sizeof *table),
                       "cudaMemcpyFromSymbol");
    for (unsigned int i = 0u; i < count; ++i) {
        static const char reason[] = "the cache gave no page for the text of the tool";
        wrong += (table->slot[i].status != AOTX_TOOL_ERROR
                  || table->slot[i].result_len != sizeof reason - 1u
                  || memcmp(table->slot[i].result, reason, sizeof reason - 1u));
    }
    aotx_tool_service_check(wrong, "page refusal", count, applied, failed);
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(1u);
    free(table);
    cudaFree(calls);
}

/* A failed reopen and a close must not leave the previous service ready. */
static void aotx_tool_service_closed(unsigned int reopen, unsigned int *applied,
                                      unsigned int *failed)
{
    unsigned int wrong = 0u;
    unsigned int ready = 1u;
    unsigned int width = 1u;
    unsigned int role = AOTX_MODEL_EMBEDDING;
    if (reopen != 0u) {
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_embed, &ready, sizeof ready,
                            offsetof(aotx_tool_embed_batch, ready)), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_embed, &width, sizeof width,
                            offsetof(aotx_tool_embed_batch, width)), "cudaMemcpyToSymbol");
        aotx_check_runtime(cudaMemcpyToSymbol(aotx_tool_embed, &role, sizeof role,
                            offsetof(aotx_tool_embed_batch, role)), "cudaMemcpyToSymbol");
        wrong += (aotx_tool_open() == 0);
    }
    aotx_check_runtime(cudaMemcpyFromSymbol(&ready, aotx_tool_embed, sizeof ready,
                        offsetof(aotx_tool_embed_batch, ready)), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&width, aotx_tool_embed, sizeof width,
                        offsetof(aotx_tool_embed_batch, width)), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&role, aotx_tool_embed, sizeof role,
                        offsetof(aotx_tool_embed_batch, role)), "cudaMemcpyFromSymbol");
    wrong += (ready != 0u || width != 0u || role != 0u);
    aotx_tool_service_check(wrong, "closed memory service", 1u, applied, failed);
}

#endif
