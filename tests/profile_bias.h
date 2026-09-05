/* Purpose: Check required bias names, types and shapes through the file descriptor reader.
 * Owns: Small model files with independent tensor names and grouped bias widths.
 * Launch shape: Host calls only; no device context.
 * Lifetime: One profile test run. */
#ifndef AOTX_TEST_PROFILE_BIAS_H
#define AOTX_TEST_PROFILE_BIAS_H

static int aotx_profile_bias_open(aotx_modelfile **model, unsigned int fault)
{
    const char *names[] = { "token_embd.weight", "output_norm.weight",
        "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
        "blk.0.attn_v.weight", "blk.0.attn_output.weight", "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
        "blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias" };
    const uint64_t shapes[][2] = { {64,32}, {64,0}, {64,0}, {64,64}, {64,32},
        {64,32}, {64,64}, {64,0}, {64,128}, {64,128}, {128,64}, {64,0}, {32,0}, {32,0} };
    unsigned int changed = fault == 0u || fault > 12u ? ~0u : 11u + (fault - 1u) % 3u;
    unsigned int error = fault == 0u || fault > 12u ? 0u : 1u + (fault - 1u) / 3u;
    aotx_profile_fixture file = {};
    aotx_profile_fixture_raw(&file, "GGUF", 4u);
    aotx_profile_fixture_number(&file, AOTX_GGUF_VERSION, 4u);
    aotx_profile_fixture_number(&file, fault == 15u ? 11u : (error == 1u ? 13u : 14u), 8u);
    aotx_profile_fixture_number(&file, fault >= 13u ? 10u : 9u, 8u);
    aotx_profile_fixture_text(&file, "general.architecture");
    aotx_profile_fixture_number(&file, AOTX_GGUF_STRING, 4u);
    aotx_profile_fixture_text(&file, "qwen2");
    const char *keys[] = { "qwen2.block_count", "qwen2.embedding_length",
        "qwen2.context_length", "qwen2.feed_forward_length", "qwen2.attention.head_count",
        "qwen2.attention.head_count_kv" };
    const unsigned int values[] = { 1u, 64u, 128u, 128u, 2u, 1u };
    for (unsigned int i = 0u; i < 6u; ++i)
        aotx_profile_fixture_u32(&file, keys[i], values[i]);
    aotx_profile_fixture_f32(&file, "qwen2.rope.freq_base", 1000000.0f);
    aotx_profile_fixture_f32(&file, "qwen2.attention.layer_norm_rms_epsilon", 1e-6f);
    if (fault == 13u)
        aotx_profile_fixture_f32(&file, "qwen2.attention.key_length", 32.0f);
    if (fault >= 14u)
        aotx_profile_fixture_u32(&file, "qwen2.attention.key_length", 0u);
    uint64_t payload = 0u;
    for (unsigned int i = 0u; i < 14u; ++i) {
        if (fault == 15u && i >= 11u) continue;
        if (i == changed && error == 1u) continue;
        uint64_t width = shapes[i][0], rows = shapes[i][1];
        unsigned int type = AOTX_TENSOR_F32;
        if (i == changed && error == 2u) type = AOTX_TENSOR_F16;
        if (i == changed && error == 3u) width += 1u;
        if (i == changed && error == 4u) rows = 1u;
        aotx_profile_fixture_text(&file, names[i]);
        aotx_profile_fixture_number(&file, rows ? 2u : 1u, 4u);
        aotx_profile_fixture_number(&file, width, 8u);
        if (rows) aotx_profile_fixture_number(&file, rows, 8u);
        aotx_profile_fixture_number(&file, type, 4u);
        aotx_profile_fixture_number(&file, payload, 8u);
        uint64_t bytes = width * (rows ? rows : 1u) * (type == AOTX_TENSOR_F16 ? 2u : 4u);
        payload += (bytes + 31u) & ~31ull;
    }
    while ((file.used & 31u) != 0u) aotx_profile_fixture_number(&file, 0u, 1u);
    unsigned char zero[128] = {};
    while (payload != 0u) {
        size_t count = payload < sizeof zero ? (size_t)payload : sizeof zero;
        aotx_profile_fixture_raw(&file, zero, count);
        payload -= count;
    }
    char path[] = "/tmp/aotx-bias-model-XXXXXX";
    int fd = mkstemp(path);
    FILE *out = fd >= 0 ? fdopen(fd, "wb") : NULL;
    int bad = file.bad != 0 || out == NULL;
    if (out != NULL) {
        bad |= fwrite(file.data, 1u, file.used, out) != file.used;
        bad |= fclose(out) != 0;
    } else if (fd >= 0) {
        close(fd);
    }
    if (bad == 0) bad = aotx_modelfile_open(path, model) != 0;
    unlink(path);
    return bad;
}

static void aotx_profile_test_bias(void)
{
    for (unsigned int fault = 0u; fault <= 15u; ++fault) {
        aotx_modelfile *file = NULL;
        if (aotx_profile_bias_open(&file, fault) != 0 || file == NULL) {
            aotx_profile_test_check(0, "the bias model file opens");
            continue;
        }
        aotx_model_binding binding[AOTX_DESC_WHOLE + AOTX_LAYER_TENSOR_SLOTS] = {};
        aotx_model_desc desc = {};
        unsigned int count = 0u;
        char reason[192] = {};
        int bad = aotx_model_desc_file(file, AOTX_MODEL_LANGUAGE, &desc, binding,
            sizeof binding / sizeof binding[0], &count, reason, sizeof reason);
        if (fault == 0u) {
            aotx_profile_test_check(bad == 0 && reason[0] == '\0'
                && desc.kind[0] == AOTX_LAYER_KIND_ATTENTION_BIAS && desc.head_dim == 32u
                && desc.heads == 2u && desc.kv_heads == 1u,
                "the bias file binds with a derived head width and grouped heads");
            const char *names[] = { "blk.0.attn_q.bias", "blk.0.attn_k.bias", "blk.0.attn_v.bias" };
            const unsigned int slots[] = { 5u, 6u, 11u };
            for (unsigned int b = 0u; b < 3u; ++b) {
                unsigned int found = 0u;
                for (unsigned int i = 0u; i < count; ++i) {
                    if (strcmp(binding[i].name, names[b]) == 0)
                        found += binding[i].needed == 1u
                              && binding[i].slot == AOTX_DESC_WHOLE + slots[b];
                }
                aotx_profile_test_check(found == 1u,
                    "the complete bias name binds once to its required descriptor slot");
            }
        } else {
            printf("profile: bias refusal %u: %s\n", fault, reason);
            if (fault <= 12u)
                aotx_profile_test_check(bad != 0 && strstr(reason, "bias") != NULL,
                    "missing biases and invalid bias types, widths and ranks are refused");
            else
                aotx_profile_test_check(bad != 0 && (strstr(reason, "head") != NULL
                    || strstr(reason, "key_length") != NULL),
                    "present head widths must have the required type and a positive value");
        }
        aotx_modelfile_close(file);
    }
}

#endif
