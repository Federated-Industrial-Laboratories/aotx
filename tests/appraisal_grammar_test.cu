/* Purpose: Verify constrained appraisal output and independent full admission.
 * Owns: Distinct source responses, malformed values and exact quote expectations.
 * Launch shape: N=1 and N=64 with four token boundary splits per response.
 * Lifetime: One maintained test process without model weights. */
#include "appraisal_model_fixture.h"
#include "appraisal/token.cuh"
#include "text/text.cuh"

static void aotx_appraisal_grammar_responses(unsigned n) {
    aotx_appraisal_model_device d(n);
    const std::string unknown = "4294967295,4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295";
    const std::string supported = "4294967295,4294967295,4294967295,0,400000,4294967295,4294967295,4294967295,4294967295";
    for (unsigned mode = 0; mode < 33; ++mode) {
        std::vector<std::string> responses;
        for (unsigned i = 0; i < n; ++i) {
            auto name = "I helped with task " + std::to_string(i) + ".";
            auto task = "task " + std::to_string(i);
            auto mixed = std::to_string(700000 + i) + "," + std::to_string(800000 - i) + ",4294967295,3,400000,900000,200000,600000,100000";
            std::vector<std::string> cases = {
                aotx_appraisal_response("", unknown),
                aotx_appraisal_response(name, mixed, task, "I promise to finish.", "1"),
                aotx_appraisal_response("Ren\\u00e9", supported),
                aotx_appraisal_response("Soup \\ud83c\\udf72.", supported),
                aotx_appraisal_response("Ren\xc3\xa9 says \\\"check\\\".", supported),
                aotx_appraisal_response("repeat", supported), aotx_appraisal_response("absent", supported),
                aotx_appraisal_response(name, "1000001,4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295"),
                aotx_appraisal_response(name, "4294967294,4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295"),
                aotx_appraisal_response(name, "4294967296,4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295"),
                aotx_appraisal_response(name, "-1,4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295"),
                aotx_appraisal_response(name, "0.5,4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295"),
                aotx_appraisal_response(name, "00,4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295"),
                aotx_appraisal_response(name, "0,4294967295,4294967295,5,4294967295,4294967295,4294967295,4294967295,4294967295"),
                aotx_appraisal_response("", mixed, task), aotx_appraisal_response("", unknown, task),
                aotx_appraisal_response("", unknown, "", "I promise to finish."), aotx_appraisal_response("", unknown, "", "", "1"),
                aotx_appraisal_response(name, mixed), aotx_appraisal_response(name, supported, "", "", "2"),
                aotx_appraisal_response(name, supported, "", "", "01"), aotx_appraisal_response(name, supported) + " extra",
                aotx_appraisal_response("\\ud800", supported), aotx_appraisal_response("\\u0000", supported),
                aotx_appraisal_response(name, "4 294967295,4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295"),
                aotx_appraisal_response(name, supported, "", "", "0,\"actor\":\"actor999\""),
                aotx_appraisal_response(name, "\"1\",4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295"),
                aotx_appraisal_response(name, "1000000,0,1000000,4,0,1000000,0,0,1000000", task),
                " \n" + aotx_appraisal_response(name, supported) + "\t\r\n",
                aotx_appraisal_response(name, "0,0,0,1,0,0,0,4294967295,4294967295"),
                aotx_appraisal_response(name, mixed, "task"),
                aotx_appraisal_response(name, mixed, "I promise to finish."),
                aotx_appraisal_response(name, mixed, task + ".")};
            responses.push_back(cases[mode]);
        }
        d.responses(responses);
        bool expected = mode < 5 || (mode >= 27 && mode <= 29);
        for (unsigned split : {1u, 2u, 7u, AOTX_INTAKE_REPLY}) {
            aotx_appraisal_model_parse<<<1,64>>>(d.reply, d.lengths, d.out, n, split); AOTX_CUDA(cudaDeviceSynchronize());
            for (unsigned i = 0; i < n; ++i) {
                aotx_check(d.out[i * 16] == expected, "incremental appraisal grammar follows the independent response expectation");
                aotx_check((d.out[i * 16 + 1] == AOTX_COG_OK) == expected, "complete appraisal parser independently checks each response");
                if (mode == 1) {
                    aotx_check(d.out[i * 16 + 2] == 700000 + i && d.out[i * 16 + 3] == 800000 - i,
                        "mixed benefit and harm preserve distinct row values");
                    aotx_check(d.out[i * 16 + 11] == 0 && d.out[i * 16 + 12] == std::string("I helped with task " + std::to_string(i) + ".").size(),
                        "exact supporting source span belongs to the current row");
                    aotx_check(d.out[i * 16 + 13] == std::string("task " + std::to_string(i)).size() && d.out[i * 16 + 14] == 20 && d.out[i * 16 + 15] == 1,
                        "task, promise and correction fields retain their separate meanings");
                }
            }
        }
    }
}
__global__ void aotx_appraisal_grammar_tokens(unsigned n, unsigned *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    aotx_seqs.slot[i].role = AOTX_MODEL_LANGUAGE; aotx_seqs.slot[i].stop = 99;
    aotx_intake.rows[i].bytes = 0; aotx_intake.rows[i].status = 0; aotx_appraisal.rows[i].prefix = {};
    out[i * 16] = aotx_appraisal_allows(i, 0);
    out[i * 16 + 1] = aotx_appraisal_allows(i, 1);
    out[i * 16 + 2] = aotx_appraisal_allows(i, 99);
    aotx_appraisal.rows[i].prefix.stage = 12;
    out[i * 16 + 3] = aotx_appraisal_allows(i, 99);
    out[i * 16 + 4] = aotx_appraisal_allows(i, 1);
}
static void aotx_appraisal_grammar_token_test(unsigned n) {
    aotx_appraisal_model_device d(n);
    unsigned char *raw; unsigned long long *offset;
    AOTX_CUDA(cudaMallocManaged(&raw, 2)); AOTX_CUDA(cudaMallocManaged(&offset, 3 * sizeof(unsigned long long)));
    memcpy(raw, "Z{", 2); for (unsigned i = 0; i < 3; ++i) offset[i] = i;
    aotx_text_vocab v = {}; v.tokens = 2; v.token_bytes = raw; v.token_at = offset;
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_text_vocab_table, &v, sizeof(v)));
    aotx_appraisal_grammar_tokens<<<1,64>>>(n, d.out); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) for (unsigned j = 0; j < 5; ++j)
        aotx_check(d.out[i * 16 + j] == (j == 1 || j == 3), "vocabulary admission permits start and stop only at complete grammar boundaries");
    cudaFree(raw); cudaFree(offset);
}
__global__ void aotx_appraisal_grammar_prefixes(const unsigned char *bytes, const unsigned *lengths,
    unsigned *out, unsigned n, unsigned split) {
    unsigned i = threadIdx.x; if (i >= n) return;
    const aotx_intake_index_row *s = aotx_intake_index_rows + i;
    unsigned task_bytes = 0;
    const unsigned char *task = aotx_appraisal_task_index(i, s, &task_bytes);
    aotx_appraisal_prefix prefix = {};
    unsigned accepted = 0;
    bool valid = true;
    for (unsigned start = 0; valid && start < lengths[i]; start += split)
        for (unsigned j = start; valid && j < min(start + split, lengths[i]); ++j) {
            valid = aotx_appraisal_prefix_byte(s, &prefix, bytes[i * AOTX_INTAKE_REPLY + j], task, task_bytes);
            if (valid) ++accepted;
        }
    out[i * 16] = valid; out[i * 16 + 1] = accepted; out[i * 16 + 2] = prefix.stage;
    out[i * 16 + 3] = out[i * 16 + 4] = out[i * 16 + 5] = 0;
    if (!valid) return;
    for (unsigned c = 0; c < 256; ++c) {
        auto candidate = prefix;
        if (!aotx_appraisal_prefix_byte(s, &candidate, c, task, task_bytes)) continue;
        ++out[i * 16 + 3];
        out[i * 16 + 5] = c;
        if (c == '"') out[i * 16 + 4] = candidate.stage;
    }
}
static void aotx_appraisal_grammar_prefix_test(unsigned n) {
    aotx_appraisal_model_device d(n);
    const std::string unknown = "4294967295,4294967295,4294967295,0,4294967295,4294967295,4294967295,4294967295,4294967295";
    for (unsigned boundary = 0; boundary < 3; ++boundary) {
        std::vector<std::string> prefixes, complete;
        for (unsigned i = 0; i < n; ++i) {
            auto task = "task " + std::to_string(i);
            auto evidence = "I helped with " + task + ".";
            auto text = boundary < 2 ? aotx_appraisal_response("", unknown) :
                aotx_appraisal_response(evidence, "700000,800000,4294967295,3,400000,900000,200000,600000,100000", task);
            size_t end = boundary == 1 ? text.find("\"commitment\":\"") + 14 :
                text.find("\"task\":\"") + 8 + (boundary == 2 ? task.size() : 0);
            prefixes.push_back(text.substr(0, end)); complete.push_back(text);
        }
        d.responses(prefixes);
        for (unsigned split : {1u, 2u, 7u, AOTX_INTAKE_REPLY}) {
            aotx_appraisal_grammar_prefixes<<<1,64>>>(d.reply, d.lengths, d.out, n, split);
            AOTX_CUDA(cudaDeviceSynchronize());
            for (unsigned i = 0; i < n; ++i) {
                aotx_check(d.out[i * 16] && d.out[i * 16 + 1] == prefixes[i].size(),
                    "each incomplete quote prefix is admitted before its required closing byte");
                aotx_check(d.out[i * 16 + 2] == (boundary == 1 ? 9u : 6u) && d.out[i * 16 + 3] == 1 &&
                    d.out[i * 16 + 4] == (boundary == 1 ? 10u : 7u),
                    "all 256 next bytes admit only quote closure at the exact required boundary");
            }
        }
        for (const std::string suffix : {"x", " ", "\\", "\\u", "\\u002e", "\\ud83c"}) {
            std::vector<std::string> responses;
            for (const auto &prefix : prefixes) responses.push_back(prefix + suffix);
            d.responses(responses);
            aotx_appraisal_grammar_prefixes<<<1,64>>>(d.reply, d.lengths, d.out, n, 7);
            AOTX_CUDA(cudaDeviceSynchronize());
            for (unsigned i = 0; i < n; ++i)
                aotx_check(!d.out[i * 16] && d.out[i * 16 + 1] == prefixes[i].size(),
                    "truncated forbidden text and escape prefixes fail at their first byte");
        }
        d.responses(complete);
        aotx_appraisal_model_parse<<<1,64>>>(d.reply, d.lengths, d.out, n, 7);
        AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i)
            aotx_check(d.out[i * 16] && d.out[i * 16 + 1] == AOTX_COG_OK,
                "each admitted incomplete boundary has a fully parsed valid completion");
    }
}
static std::string aotx_appraisal_key_case(unsigned i, unsigned mode) {
    auto task = "task " + std::to_string(i);
    auto quote = "I helped with " + task + ".";
    auto benefit = std::to_string(700000 + i);
    auto numbers = benefit + ",800000,4294967295,3,400000,900000,200000,600000,100000";
    auto text = aotx_appraisal_response(quote, numbers, task);
    auto at = text.find("\"benefit\"");
    switch (mode) {
    case 1: text.insert(text.size() - 1, ","); break;
    case 2: text.erase(at, text.find(',', at) - at + 1); break;
    case 3: text.replace(text.find("\"harm\""), 6, "\"benefit\""); break;
    case 4:
        text.replace(text.find("\"harm\""), 6, "\"benefit\""); text.replace(at, 9, "\"harm\""); break;
    case 5: text.replace(at, 9, "\"ben\\u0065fit\""); break;
    case 6: text.replace(at, 9, "\"actor\""); break;
    case 7: text.replace(at, 9, "\"Benefit\""); break;
    case 8: text.erase(at + 9, 1); break;
    case 9: text.erase(at + 8, 1); break;
    case 10: text.replace(at + 10, benefit.size(), "true"); break;
    case 11: text.replace(at, 9, "\"ben efit\""); break;
    case 12: text = "[\"" + quote + "\"," + numbers + ",\"" + task + "\",\"\",0]"; break;
    case 13: text.insert(text.size() - 1, ",\"correction\":0"); break;
    case 14: text.replace(at + 9, 1, " \n\t :\r "); text = " \n" + text + "\r\t"; break;
    case 15: text.replace(at + 10, benefit.size(), "{\"value\":" + benefit + "}"); break;
    }
    return text;
}
static void aotx_appraisal_grammar_key_test(unsigned n) {
    aotx_appraisal_model_device d(n);
    for (unsigned mode = 0; mode < 16; ++mode) {
        std::vector<std::string> responses;
        for (unsigned i = 0; i < n; ++i) responses.push_back(aotx_appraisal_key_case(i, mode));
        d.responses(responses);
        for (unsigned split : {1u, 2u, 7u, AOTX_INTAKE_REPLY}) {
            aotx_appraisal_model_parse<<<1,64>>>(d.reply, d.lengths, d.out, n, split);
            AOTX_CUDA(cudaDeviceSynchronize());
            for (unsigned i = 0; i < n; ++i) {
                aotx_check(d.out[i * 16] == (mode == 0 || mode == 14),
                    "incremental grammar requires exact ordered keys, scalar values and delimiters");
                aotx_check((d.out[i * 16 + 1] == AOTX_COG_OK) == (mode == 0 || mode == 14),
                    "independent complete parser rejects missing, duplicate, reordered and escaped keys");
            }
        }
    }
    std::vector<std::string> prefixes;
    for (unsigned i = 0; i < n; ++i) {
        auto text = aotx_appraisal_key_case(i, 0);
        prefixes.push_back(text.substr(0, text.find("\"regard_gain\"") + 8));
    }
    d.responses(prefixes);
    aotx_appraisal_grammar_prefixes<<<1,64>>>(d.reply, d.lengths, d.out, n, 7);
    AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i)
        aotx_check(d.out[i * 16] && d.out[i * 16 + 1] == prefixes[i].size() &&
            d.out[i * 16 + 3] == 1 && d.out[i * 16 + 5] == 'g',
            "a partial field name admits only its exact next byte across all 256 candidates");
    for (auto &text : prefixes) text += 'l';
    d.responses(prefixes);
    aotx_appraisal_grammar_prefixes<<<1,64>>>(d.reply, d.lengths, d.out, n, 7);
    AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i)
        aotx_check(!d.out[i * 16] && d.out[i * 16 + 1] + 1 == prefixes[i].size(),
            "an incomplete key for the wrong dimension fails at its first differing byte");
}
static void aotx_appraisal_grammar_evidence_test(unsigned n) {
    aotx_appraisal_model_device d(n);
    for (unsigned field = 0; field <= 9; ++field) {
        std::vector<std::string> valid, invalid, prefixes;
        for (unsigned i = 0; i < n; ++i) {
            std::string values;
            for (unsigned j = 0; j < 9; ++j) {
                unsigned value = j == 3 ? (field == j ? 1 : 0) : field == j ? 0 : UINT32_MAX;
                if (j) values += ',';
                values += std::to_string(value);
            }
            auto quote = "I helped with task " + std::to_string(i) + ".";
            auto task = field == 7 || field == 8 ? "task " + std::to_string(i) : "";
            valid.push_back(aotx_appraisal_response(field < 9 ? quote : "", values, task));
            invalid.push_back(aotx_appraisal_response(field < 9 ? "" : quote, values, task));
            prefixes.push_back(valid.back().substr(0, valid.back().find("\"evidence\":\"") + 12));
        }
        for (unsigned mode = 0; mode < 2; ++mode) {
            d.responses(mode ? invalid : valid);
            for (unsigned split : {1u, 2u, 7u, AOTX_INTAKE_REPLY}) {
                aotx_appraisal_model_parse<<<1,64>>>(d.reply, d.lengths, d.out, n, split);
                AOTX_CUDA(cudaDeviceSynchronize());
                for (unsigned i = 0; i < n; ++i) {
                    aotx_check(d.out[i * 16] == !mode, "dimension support and evidence presence agree at each token split");
                    aotx_check((d.out[i * 16 + 1] == AOTX_COG_OK) == !mode,
                        "final admission independently requires evidence exactly when an interpretation is known");
                }
            }
        }
        d.responses(prefixes);
        aotx_appraisal_grammar_prefixes<<<1,64>>>(d.reply, d.lengths, d.out, n, 7);
        AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(d.out[i * 16] && d.out[i * 16 + 1] == prefixes[i].size(),
                "dimensions are complete before evidence byte admission starts");
            aotx_check(field < 9 ? d.out[i * 16 + 3] > 0 && !d.out[i * 16 + 4] :
                d.out[i * 16 + 3] == 1 && d.out[i * 16 + 5] == '"',
                "all 256 candidate bytes preserve required evidence or force empty evidence without numeric changes");
        }
        for (auto &prefix : prefixes) prefix += field < 9 ? '"' : 'x';
        d.responses(prefixes);
        aotx_appraisal_grammar_prefixes<<<1,64>>>(d.reply, d.lengths, d.out, n, 7);
        AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i)
            aotx_check(!d.out[i * 16] && d.out[i * 16 + 1] + 1 == prefixes[i].size(),
                "a forbidden empty or nonempty evidence prefix fails on its first byte");
    }
}
int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        aotx_appraisal_grammar_responses(n); aotx_appraisal_grammar_token_test(n);
        aotx_appraisal_grammar_prefix_test(n); aotx_appraisal_grammar_key_test(n);
        aotx_appraisal_grammar_evidence_test(n);
    }
    printf("appraisal grammar: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
