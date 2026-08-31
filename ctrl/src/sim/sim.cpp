// Purpose: Advance all simulated control data without a live system.
// Owns: Initial data, fetch progress, and streamed reply generation.
// Launch shape: One frame advances all due simulated events.
// Lifetime: Generated events remain in the state transcript.
#include "sim/sim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>

namespace aotx::ctrl::sim {

State::State()
{
    instances = {
        {"Local system", InstanceState::running, {{"Card 0", 6120, 12282}}},
        {"Test system", InstanceState::attaching, {{"Card 1", 1880, 12282}}},
        {"Stored system", InstanceState::stopped, {{"Card 0", 0, 12282}}}};
    models = {{"Qwen language 4B", "active", 1.0f, "conductor", "", ""},
              {"Qwen embedding 0.6B", "fetching", 0.18f, "", "worker", ""},
              {"Qwen reranker 0.6B", "on disk", 1.0f, "", "", "verifier"},
              {"Compact language 0.6B", "catalog", 0.0f, "", "", ""}};
    modules = {{"Project summary", "skill"}, {"Conductor", "role"},
               {"Worker", "role"}, {"Verifier", "role"},
               {"File read", "tool"}, {"Memory recall", "tool"},
               {"Clock", "tool"}};
    settings = {{"tick.period_ms", "10", "10", "1 to 1000", true},
                {"decode.budget_ms", "120", "120", "10 to 10000", true},
                {"decode.reply_limit", "256", "256", "1 to 8191", false},
                {"decode.auto_continue", "0", "0", "0 or 1", true},
                {"sample.temperature", "0.7", "0.7", "0 to 2", false},
                {"agent.pages", "0", "0", "0 to 4096", false},
                {"mirror.hz", "30", "30", "1 to 120", true}};
    authorizations = {{41, "worker 2", "fs_read", "README.md", AuthorizationState::pending},
                      {42, "worker 4", "memory_write", "project status",
                       AuthorizationState::pending}};
    agents = {{"agent 0", "conductor", "run", 96},
              {"agent 2", "worker", "tool", 42},
              {"agent 4", "worker", "idle", 16},
              {"agent 7", "verifier", "prompt", 28}};
    transcript = {
        {EventKind::message, Role::system, "The simulated system is ready.", "", false},
        {EventKind::message, Role::user, "Read the project summary.", "", false},
        {EventKind::tool_call, Role::agent, "conductor calls fs_read README.md",
         "{\"tool\":\"fs_read\",\"path\":\"README.md\"}", false},
        {EventKind::message, Role::agent,
         "The project summary identifies a local inference operating system.", "", false},
        {EventKind::reply_bound, Role::system, "The reply reached the set bound.", "", false}};
    past_runs = {
        {"Run 104", "The project summary was read.", transcript},
        {"Run 103", "The model catalog was inspected.",
         {{EventKind::message, Role::system, "The stored run is available.", "", false},
          {EventKind::message, Role::user, "List the model catalog.", "", false},
          {EventKind::message, Role::agent,
           "The catalog contains language, embedding, and rerank models.", "", false}}},
        {"Run 102", "A module directory was imported.",
         {{EventKind::message, Role::system, "The module import completed.", "", false}}}};
}

void State::tick(double now)
{
    tick_rate_hz = 99.4f + 0.6f * static_cast<float>(std::sin(now * 0.7));
    ring_occupancy = 0.31f + 0.18f * static_cast<float>(std::sin(now * 0.31));
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

bool State::answer_authorization(std::size_t index, AuthorizationState answer)
{
    if (index >= authorizations.size() ||
        authorizations[index].state != AuthorizationState::pending) {
        return false;
    }
    authorizations[index].state = answer;
    return true;
}

bool State::fetch_model(std::size_t index)
{
    if (index >= models.size() || models[index].state != "catalog") {
        return false;
    }
    models[index].state = "fetching";
    models[index].fetch_progress = 0.0f;
    return true;
}

bool State::activate_model(std::size_t index, const std::string &role)
{
    if (index >= models.size() || models[index].state == "catalog" ||
        models[index].state == "fetching") {
        return false;
    }
    if (role == "language") {
        models[index].language_role = "conductor";
    } else if (role == "embedding") {
        models[index].embedding_role = "worker";
    } else if (role == "rerank") {
        models[index].rerank_role = "verifier";
    } else {
        return false;
    }
    models[index].state = "active";
    return true;
}

bool State::import_module(const std::string &directory)
{
    if (directory.empty()) {
        return false;
    }
    modules.push_back({"Imported directory", "skill"});
    return true;
}

bool State::set_value(std::size_t index, const std::string &value)
{
    if (index >= settings.size() || value.empty()) {
        return false;
    }
    char *end = nullptr;
    const double number = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0') {
        return false;
    }
    const std::string &key = settings[index].key;
    bool valid = false;
    if (key == "tick.period_ms") valid = number >= 1.0 && number <= 1000.0;
    else if (key == "decode.budget_ms") valid = number >= 10.0 && number <= 10000.0;
    else if (key == "decode.reply_limit") valid = number >= 1.0 && number <= 8191.0;
    else if (key == "decode.auto_continue") valid = number == 0.0 || number == 1.0;
    else if (key == "sample.temperature") valid = number >= 0.0 && number <= 2.0;
    else if (key == "agent.pages") valid = number >= 0.0 && number <= 4096.0;
    else if (key == "mirror.hz") valid = number >= 1.0 && number <= 120.0;
    if (!valid) {
        return false;
    }
    settings[index].value = value;
    return true;
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
