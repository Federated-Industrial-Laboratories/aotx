// Purpose: Declare parsers for each shipped replica line schema.
// Owns: Nothing; callers own input text and parsed records.
// Launch shape: One call parses one complete line.
// Lifetime: Parsed values remain in caller-owned records.
#ifndef AOTX_CTRL_REPLICA_SCHEMA_HPP
#define AOTX_CTRL_REPLICA_SCHEMA_HPP

#include "replica/replica.hpp"

#include <string>

namespace aotx::ctrl::replica::schema {

bool transcript(const std::string &line, TranscriptEvent &out);
bool note(const std::string &line, Note &out);
bool request(const std::string &line, Request &out);
bool pending_request(const std::string &text, PendingRequest &out);
bool agent_state(const std::string &text, AgentState &out);
bool phase(const std::string &line, std::string &word);
bool module(const std::string &line, Module &out);
bool model_catalog(const std::string &line, Model &out);
bool model_store(const std::string &line, Model &out);
bool model_manifest(const std::string &line, Model &out);
bool language_model(const std::string &line, const std::string &role, std::string &name);
bool model_load(const std::string &text, const std::string &role, std::string &file);
bool setting_result(const std::string &text, std::string &key, std::string &value);
bool import_result(const std::string &text, std::string &name, std::string &kind);
bool fetch_result(const std::string &text, std::string &name, std::uint64_t &done,
                  std::uint64_t &total, std::string &result);
bool action_result(const std::string &text);
bool token_stat(const std::string &line, TokenStat &out);
bool page_stat(const std::string &line, PageStat &out);
bool model_parameters(const std::string &line, ModelParameters &out);
bool steer_vector(const std::string &line, SteerVector &out);

} // namespace aotx::ctrl::replica::schema

#endif
