// Purpose: Send one tool query when connected, refreshed, or reconnected.
// Owns: No socket state; the client owns its connection and output queue.
// Launch shape: One interface thread updates each open conversation view.
// Lifetime: One query; a failed request waits for a connection change or refresh.
#include "tools/query.hpp"
#include <cstring>
namespace aotx::ctrl::tools {
bool query_due(Query &query, const std::string &identity, bool connected, bool refresh)
{
    if (query.identity != identity) {
        query.identity = identity;
        query.attempted = false;
        query.connected = false;
        query.result.clear();
    }
    bool reconnect = connected && !query.connected;
    bool lost = !connected && query.connected;
    query.connected = connected;
    if (!connected) {
        if (!query.attempted || lost || refresh)
            query.result = "The connection is not ready. Reconnect or use Refresh to request tool status.";
        query.attempted = true;
        return false;
    }
    if (query.attempted && !reconnect && !refresh) return false;
    query.attempted = true;
    return true;
}

void request_status(Query &query, client::Client &client, const std::string &identity,
                    const std::string &command, bool refresh)
{
    if (!query_due(query, identity, std::strcmp(client.connection(), "connected") == 0, refresh)) return;
    query.result = client.send_line(command) ? "The tool status request was sent."
        : "The tool status request failed. Reconnect or use Refresh to try again.";
}
}
