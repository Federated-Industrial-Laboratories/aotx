// Purpose: Validate the model inspector's complete text report.
// Owns: No process or model compatibility rules.
// Threading: The interface thread parses one bounded completed report.
// Lifetime: Parsed facts belong to the caller.
#ifndef AOTX_CTRL_MODEL_INSPECT_PARSE_HPP
#define AOTX_CTRL_MODEL_INSPECT_PARSE_HPP

#include "model/inspect.hpp"

namespace aotx::ctrl::model {

std::optional<InspectHeader> aotx_parse_inspection(const std::string &output,
                                                 const std::string &source);

} // namespace aotx::ctrl::model
#endif
