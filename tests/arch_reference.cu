/* Purpose: Generate CPU reference logits for fixed token positions.
 * Owns: The CPU model, context, input lists and output files.
 * Threading: One sequence per context; the reference uses 14 CPU threads.
 * Lifetime: One model check. */
// SPDX-License-Identifier: Apache-2.0
#include "llama.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string read_text(const fs::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + path.string());
    std::ostringstream text; text << in.rdbuf();
    if (in.bad()) throw std::runtime_error("read error " + path.string());
    return text.str();
}

static std::vector<std::string> columns(const std::string &line) {
    std::vector<std::string> result;
    std::stringstream stream(line); std::string value;
    while (std::getline(stream, value, '\t')) result.push_back(value);
    return result;
}

static std::vector<llama_token> id_row(const std::string &text, const std::string &key, int vocab) {
    std::istringstream input(text); std::string line;
    std::vector<llama_token> result; bool found = false;
    while (std::getline(input, line)) {
        if (line.compare(0, key.size()+1, key+" ") != 0) continue;
        if (found) throw std::runtime_error("duplicate token row");
        found = true; std::istringstream row(line.substr(key.size()+1));
        while (row >> std::ws && !row.eof()) {
            int64_t token;
            if (!(row >> token) || token < 0 || token >= vocab) throw std::runtime_error("invalid token");
            result.push_back(static_cast<llama_token>(token));
        }
    }
    if (!found || result.empty()) throw std::runtime_error("missing token row " + key);
    return result;
}

static void ids(std::ostream &out, const char *label, const std::vector<llama_token> &values) {
    out << label;
    for (auto value : values) out << ' ' << value;
    out << '\n';
}

int main(int argc, char **argv) {
    try {
        bool fixed_teacher = argc == 5 && std::strcmp(argv[4], "--fixed-teacher") == 0;
        if (argc != 4 && !fixed_teacher)
            throw std::runtime_error("usage: reference MODEL JOBS NEW_OUTPUT [--fixed-teacher]");
        const std::string job_text = read_text(argv[2]);
        if (job_text.empty()) throw std::runtime_error("no reference jobs");
        uint16_t endian = 1;
        if (*reinterpret_cast<uint8_t *>(&endian) != 1 || sizeof(float) != 4)
            throw std::runtime_error("little-endian float32 host required");
        fs::path output = argv[3];
        if (fs::exists(output)) throw std::runtime_error("output already exists");
        fs::create_directories(output);
        ggml_backend_load_all();
        auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) throw std::runtime_error("CPU backend missing");
        for (size_t i=0; i<ggml_backend_dev_count(); ++i)
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) != GGML_BACKEND_DEVICE_TYPE_CPU)
                throw std::runtime_error("unexpected non-CPU backend in pinned reference");
        llama_backend_init();
        auto mp = llama_model_default_params();
        ggml_backend_dev_t devices[] = {cpu,nullptr};
        mp.devices = devices; mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1],mp),llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        auto vocabulary = llama_model_get_vocab(model.get());
        int vocab = llama_vocab_n_tokens(vocabulary);
        auto cp = llama_context_default_params();
        cp.n_ctx=32768; cp.n_batch=2048; cp.n_ubatch=512; cp.n_seq_max=1;
        cp.n_threads=14; cp.n_threads_batch=14;
        cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_AUTO; cp.kv_unified=false;
        cp.offload_kqv=false; cp.op_offload=false;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(),cp),llama_free);
        if (!context) throw std::runtime_error("context creation failed");
        std::ofstream metadata(output/"run.txt");
        metadata << "backend cpu\nmodel " << argv[1] << "\nvocab " << vocab
                 << "\nn_ctx 32768\nn_batch 2048\nn_ubatch 512\nn_seq_max 1\nthreads 14"
                 << "\nthreads_batch 14\nflash_attn auto\nkv_unified false\noffload_kqv false"
                 << "\nop_offload false\nn_gpu_layers 0\ncallback_registered false\n"
                 << "warmup false\nsampling raw_argmax_lowest_id_tie\nparse_special true\nadd_special false\n"
                 << "cache_k f16\ncache_v f16\nfixed_teacher " << (fixed_teacher ? "true" : "false") << '\n'
                 << "system " << llama_print_system_info() << '\n';
        metadata.close();
        if (!metadata) throw std::runtime_error("metadata write failed");
        std::istringstream jobs(job_text); std::string line;
        unsigned total_rows=0, total_sequences=0;
        std::set<std::string> seen;
        while (std::getline(jobs,line)) {
            auto fields=columns(line);
            if (fields.size()!=5) throw std::runtime_error("bad job row");
            const std::string &id=fields[0];
            if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-_")!=std::string::npos
                || !seen.insert(id).second) throw std::runtime_error("bad or duplicate sequence id");
            unsigned steps=static_cast<unsigned>(std::stoul(fields[1]));
            unsigned prefix=static_cast<unsigned>(std::stoul(fields[2]));
            if (!steps || steps>64) throw std::runtime_error("bad row count");
            if (fixed_teacher && (fields[4]=="-" || fields[3]!="-" || prefix))
                throw std::runtime_error("fixed teacher mode requires complete token inputs");
            std::vector<llama_token> prefill,teacher;
            if (fields[4]!="-") {
                auto text=read_text(fields[4]);
                prefill=id_row(text,"prefill",vocab); teacher=id_row(text,"generated",vocab);
                if (teacher.size()!=steps) throw std::runtime_error("retained continuation length mismatch");
            } else {
                auto text=read_text(fields[3]);
                int count=llama_tokenize(vocabulary,text.data(),static_cast<int32_t>(text.size()),nullptr,0,false,true);
                if (count>=0) throw std::runtime_error("unexpected tokenizer sizing result");
                prefill.resize(-count);
                int written=llama_tokenize(vocabulary,text.data(),static_cast<int32_t>(text.size()),prefill.data(),-count,false,true);
                if (written!=-count) throw std::runtime_error("tokenizer size mismatch");
                if (prefix) {
                    if (prefill.size()<prefix) throw std::runtime_error("raw prefix too short");
                    prefill.resize(prefix);
                }
            }
            if (prefill.empty() || prefill.size()>512) throw std::runtime_error("bad prefill length");
            fs::path directory=output/id; fs::create_directory(directory);
            std::ofstream rows(directory/"rows.f32",std::ios::binary);
            std::ofstream tokens(directory/"tokens.txt");
            if (!rows || !tokens) throw std::runtime_error("cannot open sequence output");
            auto memory=llama_get_memory(context.get());
            if (!memory) throw std::runtime_error("model memory missing");
            llama_memory_clear(memory,true);
            std::vector<llama_token> input=prefill, argmax, feed;
            std::vector<llama_pos> positions(input.size());
            std::vector<int8_t> outputs(input.size(),0);
            for (size_t i=0;i<input.size();++i) positions[i]=static_cast<llama_pos>(i);
            outputs.back()=1;
            for (unsigned step=0;step<steps;++step) {
                llama_batch batch{};
                batch.n_tokens=static_cast<int32_t>(input.size());batch.token=input.data();
                batch.pos=positions.data();batch.logits=outputs.data();
                int rc=llama_decode(context.get(),batch);
                if (rc) throw std::runtime_error(id+" decode step "+std::to_string(step)+" rc "+std::to_string(rc));
                const float *row=llama_get_logits_ith(context.get(),-1);
                if (!row) throw std::runtime_error("no logit row");
                int best=0;
                for (int i=0;i<vocab;++i) {
                    if (!std::isfinite(row[i])) throw std::runtime_error("non-finite logit");
                    if (row[i]>row[best]) best=i;
                }
                argmax.push_back(best);
                rows.write(reinterpret_cast<const char *>(row),static_cast<std::streamsize>(vocab)*4);
                if (!rows) throw std::runtime_error("row write failed");
                llama_token chosen=teacher.empty()?best:teacher[step];
                feed.push_back(chosen);
                if (step+1<steps) {
                    input.assign(1,chosen);positions.assign(1,static_cast<llama_pos>(prefill.size()+step));
                    outputs.assign(1,1);
                }
            }
            ids(tokens,"prefill",prefill);ids(tokens,"teacher",feed);ids(tokens,"argmax",argmax);
            rows.close();tokens.close();
            if (!rows || !tokens || fs::file_size(directory/"rows.f32")!=static_cast<uint64_t>(vocab)*steps*4)
                throw std::runtime_error("incomplete sequence files");
            if (fixed_teacher && feed!=teacher) throw std::runtime_error("fixed teacher inputs changed: "+id);
            if (!fixed_teacher && !teacher.empty() && teacher!=argmax)
                throw std::runtime_error("retained CPU continuation changed: "+id);
            total_rows+=steps;total_sequences++;
            std::cout << "complete " << id << " prefill=" << prefill.size() << " rows=" << steps << std::endl;
        }
        std::cout << "total sequences=" << total_sequences << " rows=" << total_rows << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "reference failed: " << error.what() << std::endl;
        return 1;
    }
}
