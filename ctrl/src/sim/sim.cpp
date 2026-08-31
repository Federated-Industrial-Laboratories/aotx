// Purpose: Advance all simulated control data without a live system.
// Owns: Initial data, fetch progress, and streamed reply generation.
// Launch shape: One frame advances all due simulated events.
// Lifetime: Generated events remain in the state transcript.
#include "sim/sim.hpp"

#include <algorithm>
#include <utility>

namespace aotx::ctrl::sim {

State::State()
{
    instances = {
        {"Local system", InstanceState::running, {{"Card 0", 6120, 12282}}},
        {"Test system", InstanceState::attaching, {{"Card 1", 1880, 12282}}},
        {"Stored system", InstanceState::stopped, {{"Card 0", 0, 12282}}}};
    models = {{"Language model", "active", 1.0f},
              {"Embedding model", "fetching", 0.18f},
              {"Rerank model", "on disk", 1.0f}};
    modules = {{"Conductor", "role"}, {"File tools", "tools"}, {"Clock", "device tool"}};
    settings = {{"reply.tokens", "512"},
                {"reply.continue", "off"},
                {"journal.keep", "on"},
                {"display.rate", "60"}};
    transcript = {
        {EventKind::message, Role::system, "The simulated system is ready.", "", false},
        {EventKind::message, Role::user, "Read the project summary.", "", false},
        {EventKind::tool_call, Role::agent, "conductor calls fs_read README.md",
         "{\"tool\":\"fs_read\",\"path\":\"README.md\"}", false},
        {EventKind::message, Role::agent,
         "The project summary identifies a local inference operating system.", "", false},
        {EventKind::reply_bound, Role::system, "The reply reached the set bound.", "", false}};
}

void State::tick(double now)
{
    if (now >= next_fetch_tick_) {
        next_fetch_tick_ = now + 0.12;
        for (Model &model : models) {
            if (model.state == "fetching") {
                model.fetch_progress = std::min(1.0f, model.fetch_progress + 0.004f);
                if (model.fetch_progress >= 1.0f) {
                    model.state = "on disk";
                }
            }
        }
    }

    if (reply_source_.empty() || now < next_reply_tick_) {
        return;
    }
    next_reply_tick_ = now + 0.025;
    const std::size_t amount = std::min<std::size_t>(3, reply_source_.size() - reply_offset_);
    transcript.back().stated.append(reply_source_, reply_offset_, amount);
    reply_offset_ += amount;
    if (reply_offset_ == reply_source_.size()) {
        transcript.back().streaming = false;
        reply_source_.clear();
        reply_offset_ = 0;
    }
}

void State::start_reply(double now, std::string reply)
{
    reply_source_ = std::move(reply);
    reply_offset_ = 0;
    next_reply_tick_ = now;
    transcript.push_back({EventKind::message, Role::agent, "", "", true});
}

void State::send(std::string text, double now)
{
    if (text.empty()) {
        return;
    }
    transcript.push_back({EventKind::message, Role::user, std::move(text), "", false});
    start_reply(now, "The simulated system received the text. No live system is attached.");
}

void State::continue_reply(double now)
{
    start_reply(now, "The simulated reply continues after the set bound.");
}

void State::set_instance_state(std::size_t index, InstanceState state)
{
    if (index < instances.size()) {
        instances[index].state = state;
    }
}

const char *state_name(InstanceState state)
{
    switch (state) {
    case InstanceState::running: return "running";
    case InstanceState::attaching: return "attaching";
    case InstanceState::stopped: return "stopped";
    }
    return "stopped";
}

} // namespace aotx::ctrl::sim
