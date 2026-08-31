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
bool module(const std::string &line, Module &out);
bool language_model(const std::string &line, const std::string &role, std::string &name);
bool model_load(const std::string &text, const std::string &role, std::string &file);

} // namespace aotx::ctrl::replica::schema

#endif
