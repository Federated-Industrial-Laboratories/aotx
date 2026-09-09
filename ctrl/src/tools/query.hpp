// Purpose: Coalesce tool status queries across frames and connection changes.
// Owns: The last query identity, connection state, and result text.
// Launch shape: One interface thread updates each open conversation view.
// Lifetime: The query state lasts as long as its conversation view.
#ifndef AOTX_CTRL_TOOLS_QUERY_HPP
#define AOTX_CTRL_TOOLS_QUERY_HPP
#include "client/client.hpp"
#include <string>
namespace aotx::ctrl::tools {
struct Query {
    std::string identity;
    std::string result;
    bool attempted = false;
    bool connected = false;
};
bool query_due(Query &query, const std::string &identity, bool connected, bool refresh);
void request_status(Query &query, client::Client &client, const std::string &identity,
                    const std::string &command, bool refresh);
}
#endif
