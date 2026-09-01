// Purpose: Advance all simulated control data without a live system.
// Owns: Initial data, instance creation, and conversation reply generation.
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
    instances = {{"Local system", InstanceState::running, {{"Card 0", 6120, 12282}}}};
    models = {{"Qwen language 4B", "active", 1.0f, "language", "", ""},
              {"Qwen embedding 0.6B", "active", 1.0f, "", "embedding", ""},
              {"Qwen reranker 0.6B", "active", 1.0f, "", "", "reranker"},
              {"Compact language 0.6B", "catalog", 0.0f, "", "", ""}};
    modules = {{"Project summary", "skill"}, {"Conductor", "role"},
               {"Worker", "role"}, {"Verifier", "role"},
               {"File read", "tool"}, {"Memory recall", "tool"},
               {"Clock", "tool"}};
    module_directories = {{"/opt/aotx/modules/summary", "Imported summary", "skill"},
                          {"/opt/aotx/modules/planner", "Imported planner", "role"},
                          {"/opt/aotx/modules/search", "Imported search", "tool"}};
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
    std::vector<TranscriptEvent> transcript = {
        {EventKind::message, Role::system, "The simulated system is ready.", "", false},
        {EventKind::message, Role::user, "Read the project summary.", "", false},
        {EventKind::tool_call, Role::agent, "conductor calls fs_read README.md",
         "{\"tool\":\"fs_read\",\"path\":\"README.md\"}", false},
        {EventKind::message, Role::agent,
         "The project summary identifies a local inference operating system.", "", false},
        {EventKind::reply_bound, Role::system, "The reply reached the set bound.", "", false}};
    conversations = {{"First conversation", 0, transcript}};
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
                    results_.push_back(model.name + " fetch completed.");
                }
            }
        }
    }

    // A simulated attach completes after four seconds and states the result.
    for (Instance &instance : instances) {
        if (instance.state == InstanceState::attaching && now - instance.state_since >= 4.0) {
            instance.state = InstanceState::running;
            results_.push_back(instance.name + " is running.");
        }
    }

    for (Conversation &conversation : conversations) {
        if (conversation.reply_source.empty() || now < conversation.next_reply_tick) continue;
        conversation.next_reply_tick = now + 0.025;
        const std::size_t amount = std::min<std::size_t>(
            3, conversation.reply_source.size() - conversation.reply_offset);
        conversation.transcript.back().stated.append(conversation.reply_source,
                                                      conversation.reply_offset, amount);
        conversation.reply_offset += amount;
        if (conversation.reply_offset == conversation.reply_source.size()) {
            conversation.transcript.back().streaming = false;
            conversation.reply_source.clear();
            conversation.reply_offset = 0;
        }
    }
}

void State::start_reply(std::size_t conversation_index, double now, std::string reply)
{
    finish_reply(conversation_index);
    Conversation &conversation = conversations[conversation_index];
    conversation.reply_source = std::move(reply);
    conversation.reply_offset = 0;
    conversation.next_reply_tick = now;
    conversation.transcript.push_back(
        {EventKind::message, Role::agent, "", "", true,
         static_cast<unsigned>(conversation_index)});
}

void State::finish_reply(std::size_t conversation_index)
{
    Conversation &conversation = conversations[conversation_index];
    if (conversation.reply_source.empty()) return;
    conversation.transcript.back().stated.append(
        conversation.reply_source, conversation.reply_offset, std::string::npos);
    conversation.transcript.back().streaming = false;
    conversation.reply_source.clear();
    conversation.reply_offset = 0;
}

void State::send(std::size_t conversation_index, std::string text, double now)
{
    if (conversation_index >= conversations.size() || text.empty()) return;
    finish_reply(conversation_index);
    conversations[conversation_index].transcript.push_back(
        {EventKind::message, Role::user, std::move(text), "", false});
    start_reply(conversation_index, now, "The simulated system received the text.");
}

void State::continue_reply(std::size_t conversation, double now)
{
    if (conversation < conversations.size()) {
        start_reply(conversation, now, "The simulated reply continues after the set bound.");
    }
}

std::size_t State::create_conversation(std::size_t instance)
{
    const std::size_t number = conversations.size();
    std::string name = number == 1 ? "New conversation"
                                   : "New conversation " + std::to_string(number);
    conversations.push_back({std::move(name), instance, {}});
    selected_conversation = conversations.size() - 1;
    conversation_selection_requested = true;
    return selected_conversation;
}

bool State::create_instance(const std::string &name)
{
    refusal_.clear();
    if (name.empty()) {
        refusal_ = "The instance creation was refused because the name is empty.";
        return false;
    }
    instances.push_back({name, InstanceState::stopped, {{"Card 0", 0, 12282}}});
    selected_instance = instances.size() - 1;
    instance_selection_requested = true;
    return true;
}

void State::set_instance_state(std::size_t index, InstanceState state, double now)
{
    if (index < instances.size()) {
        instances[index].state = state;
        instances[index].state_since = now;
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
    refusal_.clear();
    if (index >= models.size()) {
        refusal_ = "The model fetch was refused because the model does not exist.";
        return false;
    }
    if (models[index].state != "catalog") {
        refusal_ = models[index].state == "fetching"
                       ? "The model fetch was refused because a fetch is already in progress."
                       : "The model fetch was refused because the model is already on disk.";
        return false;
    }
    models[index].state = "fetching";
    models[index].fetch_progress = 0.0f;
    return true;
}

bool State::activate_model(std::size_t index, const std::string &role)
{
    refusal_.clear();
    if (index >= models.size() || models[index].state == "catalog" ||
        models[index].state == "fetching") {
        refusal_ = "The model activation was refused because the model is not on disk.";
        return false;
    }
    std::string Model::*assignment = nullptr;
    if (role == "language") {
        assignment = &Model::language_role;
    } else if (role == "embedding") {
        assignment = &Model::embedding_role;
    } else if (role == "reranker") {
        assignment = &Model::reranker_role;
    } else {
        refusal_ = "The model activation was refused because the role is not valid.";
        return false;
    }
    for (Model &model : models) {
        model.*assignment = "";
        if (model.language_role.empty() && model.embedding_role.empty() &&
            model.reranker_role.empty() && model.state == "active") {
            model.state = "on disk";
        }
    }
    models[index].*assignment = role;
    models[index].state = "active";
    return true;
}

bool State::import_module(const std::string &directory)
{
    refusal_.clear();
    if (directory.empty()) {
        refusal_ = "The module import was refused because no directory is selected.";
        return false;
    }
    const auto item = std::find_if(module_directories.begin(), module_directories.end(),
                                   [&directory](const ModuleDirectory &entry) {
                                       return entry.path == directory;
                                   });
    if (item == module_directories.end()) {
        refusal_ = "The module import was refused because the directory is not in the catalog.";
        return false;
    }
    modules.push_back({item->name, item->kind});
    return true;
}

bool State::set_value(std::size_t index, const std::string &value)
{
    refusal_.clear();
    if (index >= settings.size() || value.empty()) {
        refusal_ = "The setting was refused because its value is empty.";
        return false;
    }
    char *end = nullptr;
    const double number = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || *end != '\0') {
        refusal_ = "The setting was refused because its value is not a number.";
        return false;
    }
    const std::string &key = settings[index].key;
    if (key != "sample.temperature" && std::floor(number) != number) {
        refusal_ = key + " was refused because it requires a whole number.";
        return false;
    }
    bool valid = false;
    if (key == "tick.period_ms") valid = number >= 1.0 && number <= 1000.0;
    else if (key == "decode.budget_ms") valid = number >= 10.0 && number <= 10000.0;
    else if (key == "decode.reply_limit") valid = number >= 1.0 && number <= 8191.0;
    else if (key == "decode.auto_continue") valid = number == 0.0 || number == 1.0;
    else if (key == "sample.temperature") valid = number >= 0.0 && number <= 2.0;
    else if (key == "agent.pages") valid = number >= 0.0 && number <= 4096.0;
    else if (key == "mirror.hz") valid = number >= 1.0 && number <= 120.0;
    if (!valid) {
        refusal_ = key + " was refused because its value is outside the permitted range.";
        return false;
    }
    settings[index].value = value;
    return true;
}

std::vector<std::string> State::take_results()
{
    std::vector<std::string> results = std::move(results_);
    results_.clear();
    return results;
}

const std::string &State::refusal() const { return refusal_; }

const char *state_name(InstanceState state)
{
    switch (state) {
    case InstanceState::running: return "running";
    case InstanceState::attaching: return "attaching";
    case InstanceState::stopped: return "stopped";
    }
    return "stopped";
}

bool same_transcript(const std::vector<TranscriptEvent> &left,
                     const std::vector<TranscriptEvent> &right)
{
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const TranscriptEvent &a = left[index];
        const TranscriptEvent &b = right[index];
        if (a.kind != b.kind || a.role != b.role || a.stated != b.stated ||
            a.detail != b.detail || a.streaming != b.streaming ||
            a.agent_index != b.agent_index) return false;
    }
    return true;
}

bool verify_paths()
{
    State state;
    if (state.instances.size() != 1 || state.instances[0].name != "Local system" ||
        state.instances[0].state != InstanceState::running ||
        state.instances[0].cards.size() != 1 || state.conversations.size() != 1 ||
        state.conversations[0].name != "First conversation" ||
        state.conversations[0].instance_index != 0) return false;
    if (state.models[0].language_role != "language" ||
        state.models[1].embedding_role != "embedding" ||
        state.models[2].reranker_role != "reranker") return false;
    if (!state.create_instance("Created system") || state.instances.size() != 2 ||
        state.instances[1].state != InstanceState::stopped ||
        state.instances[1].cards.size() != 1) return false;
    const std::vector<TranscriptEvent> first_transcript = state.conversations[0].transcript;
    const unsigned first_bound = state.conversations[0].reply_bound;
    const std::size_t second = state.create_conversation(1);
    if (second != 1 || state.conversations[second].instance_index != 1) return false;
    state.conversations[second].reply_bound = 1024;
    if (state.conversations[0].reply_bound != first_bound) return false;
    state.send(second, "second", 0.0);
    state.tick(0.0);
    if (!same_transcript(state.conversations[0].transcript, first_transcript) ||
        !state.conversations[second].transcript.back().streaming ||
        state.conversations[second].transcript.back().agent_index != second) return false;
    const std::size_t second_size = state.conversations[second].transcript.size();
    state.continue_reply(second, 0.001);
    if (!same_transcript(state.conversations[0].transcript, first_transcript)) return false;
    state.send(second, "third", 0.002);
    state.tick(0.002);
    state.send(0, "first", 0.001);
    if (state.conversations[second].transcript.size() <= second_size ||
        state.conversations[second].transcript.back().stated != "The") return false;
    const std::vector<TranscriptEvent> second_transcript =
        state.conversations[second].transcript;
    state.continue_reply(0, 0.002);
    if (!same_transcript(state.conversations[second].transcript, second_transcript)) return false;
    if (!state.activate_model(2, "language")) return false;
    unsigned language_assignments = 0;
    for (const Model &model : state.models) {
        if (!model.language_role.empty()) ++language_assignments;
    }
    if (language_assignments != 1 || state.set_value(2, "1.5") ||
        state.refusal().find("whole number") == std::string::npos) return false;
    if (state.set_value(5, "17.5") ||
        state.refusal().find("whole number") == std::string::npos) return false;
    for (const ModuleDirectory &directory : state.module_directories) {
        if (!state.import_module(directory.path) || state.modules.back().kind != directory.kind) {
            return false;
        }
    }
    state.set_instance_state(1, InstanceState::attaching, 50.0);
    state.tick(53.9);
    if (state.instances[1].state != InstanceState::attaching) return false;
    if (!state.fetch_model(3) || state.fetch_model(3) ||
        state.refusal().find("already in progress") == std::string::npos) return false;
    for (unsigned frame = 0; frame < 300; ++frame) {
        state.tick(100.0 + static_cast<double>(frame) * 0.13);
    }
    const std::vector<std::string> results = state.take_results();
    return state.models[3].state == "on disk" && !state.fetch_model(3) &&
           state.refusal().find("already on disk") != std::string::npos &&
           state.instances[1].state == InstanceState::running && results.size() == 2 &&
           results[0] == "Created system is running." &&
           results[1] == "Compact language 0.6B fetch completed.";
}

} // namespace aotx::ctrl::sim
