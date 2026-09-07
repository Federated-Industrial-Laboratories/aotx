// Purpose: Define the JSON value reader for replica line files.
// Owns: Parsed scalar, array, and object values.
// Launch shape: One caller parses one complete line.
// Lifetime: A value remains valid while its owner keeps it.
#ifndef AOTX_CTRL_REPLICA_JSON_HPP
#define AOTX_CTRL_REPLICA_JSON_HPP

#include <string>
#include <utility>
#include <vector>

namespace aotx::ctrl::replica::json {

enum class Kind { null_value, boolean, number, string, object, array };

struct Value {
    Kind kind = Kind::null_value;
    bool boolean = false;
    std::string text;
    std::vector<std::pair<std::string, Value>> members;
    std::vector<Value> elements;

    const Value *get(const char *name) const;
};

bool parse(const std::string &line, Value &out);
bool text(const Value &object, const char *name, std::string &out);
bool number(const Value &object, const char *name, unsigned long long &out);

} // namespace aotx::ctrl::replica::json

#endif
