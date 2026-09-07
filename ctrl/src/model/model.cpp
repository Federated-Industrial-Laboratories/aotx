// Purpose: List model files and control their use.
// Owns: Model action buttons and their result notifications.
// Launch shape: One panel draws all catalog entries each frame.
// Lifetime: Model rows remain in the replica or simulated state.
#include "model/model.hpp"

#include "imgui.h"
#include "replica/store.hpp"
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

namespace aotx::ctrl::model {
namespace {

void activate(sim::State &state, toast::Lane &toasts, std::size_t index,
              const char *role, double now)
{
    if (state.activate_model(index, role)) {
        toasts.add(state.models[index].name + " is active for " + role + ".",
                   toast::Severity::success, now);
    } else {
        toasts.add(state.refusal(), toast::Severity::error, now);
    }
}

const replica::ModelParameters *declared(const replica::State &state,
                                         const std::string &model)
{
    for (const replica::ModelParameters &row : state.model_parameters()) {
        if (row.name == model) return &row;
    }
    return nullptr;
}

const replica::ModelParameter *declared(const replica::ModelParameters *parameters,
                                        const std::string &name)
{
    if (parameters == nullptr) return nullptr;
    for (const replica::ModelParameter &row : parameters->values) {
        if (row.name == name) return &row;
    }
    return nullptr;
}

std::string value_text(double value, bool whole)
{
    std::ostringstream out;
    if (whole) out << static_cast<long long>(std::llround(value));
    else out << std::setprecision(7) << value;
    return out.str();
}

std::string acknowledgment(const replica::State &state, const std::string &key,
                           const std::string &value)
{
    const std::string wanted = "agent: decode." + key + " " + value +
                               " changes at the next turn";
    for (auto at = state.notes().rbegin(); at != state.notes().rend(); ++at) {
        if (at->text == wanted) return at->text;
    }
    return {};
}

void initialize_panel(LivePanelState &panel, const replica::State &state, unsigned agent)
{
    const std::string binding = state.models_directory().string() + ":" +
                                state.language_model() + ":" + std::to_string(agent);
    if (panel.binding == binding) return;
    panel = LivePanelState{};
    panel.binding = binding;
    const replica::ModelParameters *model = declared(state, state.language_model());
    for (const EngineParameter &parameter : engine_parameters()) {
        const replica::ModelParameter *row = declared(model, parameter.name);
        panel.values[parameter.name] = row == nullptr ? parameter.initial : row->initial;
    }
    std::string reason;
    if (!read_presets(state.models_directory(), panel.presets, reason)) panel.result = reason;
}

void draw_parameter_controls(LivePanelState &panel, replica::State &state,
                             client::Client &client, unsigned agent)
{
    const replica::ModelParameters *parameters = declared(state, state.language_model());
    ImGui::SeparatorText("Model controls");
    ImGui::Text("Agent %u uses %s.", agent, state.language_model().c_str());
    for (const EngineParameter &engine : engine_parameters()) {
        const replica::ModelParameter *model = declared(parameters, engine.name);
        const double initial = model == nullptr ? engine.initial : model->initial;
        const double least = model == nullptr ? engine.least : model->least;
        const double most = model == nullptr ? engine.most : model->most;
        ImGui::PushID(engine.name);
        ImGui::SetNextItemWidth(180.0f);
        ImGui::InputDouble(engine.label, &panel.values[engine.name], engine.whole ? 1.0 : 0.01,
                           engine.whole ? 10.0 : 0.1, engine.whole ? "%.0f" : "%.4f");
        ImGui::TextDisabled("Default %s. Range %s to %s.%s",
            value_text(initial, engine.whole).c_str(), value_text(least, engine.whole).c_str(),
            value_text(most, engine.whole).c_str(),
            model == nullptr ? " The model does not declare this parameter; this is the engine default." : "");
        const bool valid = panel.values[engine.name] >= least && panel.values[engine.name] <= most &&
            (!engine.whole || panel.values[engine.name] == std::floor(panel.values[engine.name]));
        if (!valid) ImGui::BeginDisabled();
        if (ImGui::Button("Set")) {
            const std::string value = value_text(panel.values[engine.name], engine.whole);
            const std::string line = "agent " + std::to_string(agent) + " decode." +
                                     engine.name + " " + value;
            if (client.send_line(line)) panel.result = "The " + std::string(engine.label) +
                " set request was sent.";
        }
        if (!valid) ImGui::EndDisabled();
        ImGui::SameLine();
        ImGui::TextDisabled("Set this parameter for the agent at its next turn.");
        const std::string value = value_text(panel.values[engine.name], engine.whole);
        const std::string ack = acknowledgment(state, engine.name, value);
        if (!ack.empty()) ImGui::TextWrapped("Acknowledgment: %s", ack.c_str());
        ImGui::Separator();
        ImGui::PopID();
    }
}

void draw_presets(LivePanelState &panel, replica::State &state, client::Client &client,
                  unsigned agent)
{
    ImGui::SeparatorText("Presets");
    if (panel.presets.empty()) ImGui::TextDisabled("The model store has no preset files.");
    for (const Preset &preset : panel.presets) {
        ImGui::PushID(preset.name.c_str());
        ImGui::TextUnformatted(preset.name.c_str());
        std::string values;
        for (const auto &value : preset.values) {
            if (!values.empty()) values += ", ";
            values += value.first + " " + value.second;
        }
        ImGui::TextWrapped("%s", values.c_str());
        if (ImGui::Button("Apply")) {
            unsigned sent = 0u;
            for (const std::string &line : preset_commands(agent, preset)) {
                if (client.send_line(line)) ++sent;
            }
            panel.result = "The " + preset.name + " preset sent " +
                           std::to_string(sent) + " settings: " + values + ".";
        }
        ImGui::SameLine();
        ImGui::TextDisabled("Apply all parameter values in this preset.");
        unsigned confirmed = 0u;
        for (const auto &value : preset.values) {
            if (!acknowledgment(state, value.first.substr(7u), value.second).empty()) ++confirmed;
        }
        if (confirmed != 0u) ImGui::Text("Acknowledgments: %u of %zu.", confirmed,
                                         preset.values.size());
        ImGui::PopID();
    }
}

void draw_conduct(LivePanelState &panel, replica::State &state, client::Client &client,
                  unsigned agent)
{
    ImGui::SeparatorText("Conduct");
    for (unsigned slot = 0u; slot < 2u; ++slot) {
        ImGui::PushID(static_cast<int>(slot));
        const char *preview = panel.steer[slot] == 0 ? "absent" :
            state.steer_vectors()[static_cast<std::size_t>(panel.steer[slot] - 1)].name.c_str();
        if (ImGui::BeginCombo("Steer vector", preview)) {
            if (ImGui::Selectable("absent", panel.steer[slot] == 0)) panel.steer[slot] = 0;
            for (std::size_t index = 0u; index < state.steer_vectors().size(); ++index) {
                if (ImGui::Selectable(state.steer_vectors()[index].name.c_str(),
                                      panel.steer[slot] == static_cast<int>(index + 1u))) {
                    panel.steer[slot] = static_cast<int>(index + 1u);
                }
            }
            ImGui::EndCombo();
        }
        ImGui::SliderFloat("Strength", &panel.strength[slot], -4.0f, 4.0f, "%.2f");
        if (panel.steer[slot] != 0) {
            const auto &vector = state.steer_vectors()[static_cast<std::size_t>(panel.steer[slot] - 1)];
            ImGui::SameLine();
            ImGui::Text("Potency %.6g nats; dose %.6g nats.", vector.potency_nats,
                        vector.potency_nats * panel.strength[slot]);
        } else ImGui::TextDisabled("No potency applies to an absent vector.");
        if (ImGui::Button("Set")) {
            const std::string value = panel.steer[slot] == 0 ? "absent" :
                std::string(preview) + ":" + value_text(panel.strength[slot], false);
            if (client.send_line("agent " + std::to_string(agent) + " decode.steer" +
                                 std::to_string(slot) + " " + value)) {
                panel.result = "The steer vector set request was sent.";
            }
        }
        ImGui::SameLine();
        ImGui::TextDisabled("Set this steer vector and strength for the agent.");
        const std::string selected = panel.steer[slot] == 0 ? "absent" :
            std::string(preview) + ":" + value_text(panel.strength[slot], false);
        const std::string steer_ack = acknowledgment(
            state, "steer" + std::to_string(slot), selected);
        if (!steer_ack.empty()) ImGui::TextWrapped("Acknowledgment: %s", steer_ack.c_str());
        ImGui::Separator();
        ImGui::PopID();
    }
    const char *voice = panel.voice == 0 ? "absent" :
        state.voice_profiles()[static_cast<std::size_t>(panel.voice - 1)].name.c_str();
    if (ImGui::BeginCombo("Voice bias", voice)) {
        if (ImGui::Selectable("absent", panel.voice == 0)) panel.voice = 0;
        for (std::size_t index = 0u; index < state.voice_profiles().size(); ++index) {
            if (ImGui::Selectable(state.voice_profiles()[index].name.c_str(),
                                  panel.voice == static_cast<int>(index + 1u))) {
                panel.voice = static_cast<int>(index + 1u);
            }
        }
        ImGui::EndCombo();
    }
    if (ImGui::Button("Set")) {
        if (client.send_line("agent " + std::to_string(agent) + " decode.voice " + voice)) {
            panel.result = "The voice bias set request was sent.";
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Set this voice bias profile for the agent.");
    const std::string voice_ack = acknowledgment(state, "voice", voice);
    if (!voice_ack.empty()) ImGui::TextWrapped("Acknowledgment: %s", voice_ack.c_str());
}

} // namespace

void draw(sim::State &state, toast::Lane &toasts, double now, bool *open)
{
    if (!ImGui::Begin("Models", open)) {
        ImGui::End();
        return;
    }
    ImGui::SeparatorText("Role assignments");
    static const char *roles[] = {"language", "embedding", "reranker"};
    for (const char *role : roles) {
        const sim::Model *assigned = nullptr;
        for (const sim::Model &item : state.models) {
            const std::string &value = role == roles[0] ? item.language_role
                                      : role == roles[1] ? item.embedding_role
                                                         : item.reranker_role;
            if (!value.empty()) assigned = &item;
        }
        ImGui::Text("%s: %s", role, assigned == nullptr ? "not assigned" : assigned->name.c_str());
    }
    ImGui::SeparatorText("Catalog");
    for (std::size_t index = 0; index < state.models.size(); ++index) {
        sim::Model &item = state.models[index];
        ImGui::PushID(static_cast<int>(index));
        ImGui::TextUnformatted(item.name.c_str());
        ImGui::TextDisabled("%s", item.state.c_str());
        if (item.state == "fetching") {
            ImGui::ProgressBar(item.fetch_progress, ImVec2(-1.0f, 0.0f));
        } else if (item.state == "catalog") {
            if (ImGui::Button("Fetch")) {
                if (state.fetch_model(index)) {
                    toasts.add(item.name + " fetch started.", toast::Severity::info, now);
                } else {
                    toasts.add(state.refusal(), toast::Severity::error, now);
                }
            }
        }
        if (item.state == "on disk" || item.state == "active") {
            if (ImGui::Button("Use##language")) activate(state, toasts, index, "language", now);
            ImGui::SameLine();
            ImGui::TextDisabled("Use this file for language.");
            if (ImGui::Button("Use##embedding")) activate(state, toasts, index, "embedding", now);
            ImGui::SameLine();
            ImGui::TextDisabled("Use this file for embedding.");
            if (ImGui::Button("Use##reranker")) {
                activate(state, toasts, index, "reranker", now);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("Use this file for reranking.");
        }
        if (!item.language_role.empty()) ImGui::Text("Language: %s", item.language_role.c_str());
        if (!item.embedding_role.empty()) ImGui::Text("Embedding: %s", item.embedding_role.c_str());
        if (!item.reranker_role.empty()) {
            ImGui::Text("Reranker: %s", item.reranker_role.c_str());
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    ImGui::End();
}

void draw(StoreAction &action, LivePanelState &panel, DetailsState &details,
          const std::filesystem::path &build,
          replica::State &state, client::Client &client, toast::Lane &toasts, double now,
          bool *open)
{
    if (!ImGui::Begin("Models", open)) {
        ImGui::End();
        return;
    }
    if (action.running() && !action.progress().empty()) {
        ImGui::TextUnformatted(action.progress().c_str());
        ImGui::Separator();
    }
    ImGui::TextWrapped("Store: %s", state.models_directory().string().c_str());
    ImGui::TextWrapped("%s: %s", state.language_load_seen() ? "Last reported language load"
                                                         : "Manifest language",
                       state.language_model().c_str());
    ImGui::SeparatorText("Active manifest roles");
    ImGui::PushID("Active roles");
    bool assigned = false;
    for (const replica::Model &item : state.models()) {
        if (item.active) {
            ImGui::PushID(replica::store::key(item).c_str());
            ImGui::TextWrapped("%s: %s", item.role.c_str(), item.name.c_str());
            if (item.on_disk && ImGui::Button("Inspect"))
                details.start(build, state.models_directory(), item);
            if (!item.on_disk) ImGui::TextDisabled("The model file is absent or its size differs.");
            ImGui::PopID();
            assigned = true;
        }
    }
    if (!assigned) ImGui::TextDisabled("The manifest has no active roles.");
    ImGui::PopID();
    ImGui::SeparatorText("Model files");
    for (const replica::Model &item : state.models()) {
        ImGui::PushID(replica::store::key(item).c_str());
        ImGui::TextWrapped("%s", item.name.c_str());
        ImGui::TextWrapped("File: %s", item.file.c_str());
        ImGui::TextDisabled("%s  %s  %llu bytes", item.role.empty() ? "No role" : item.role.c_str(), item.quant.c_str(),
                            static_cast<unsigned long long>(item.bytes));
        ImGui::TextWrapped("%s", item.on_disk ? "The file is present with the declared size."
                                              : "The file is absent or its size differs.");
        if (!item.catalogued) ImGui::TextDisabled("Local model entry.");
        if (ImGui::TreeNode("Source")) {
            ImGui::TextWrapped("Source: %s", item.source.c_str());
            ImGui::TextWrapped("Declared SHA-256: %s", item.digest.c_str());
            ImGui::TextWrapped("File presence does not verify its digest or runtime use.");
            ImGui::TreePop();
        }
        if (item.on_disk && ImGui::Button("Inspect"))
            details.start(build, state.models_directory(), item);
        if (item.fetching) {
            const float progress = item.fetch_total == 0u ? 0.0f :
                static_cast<float>(static_cast<double>(item.fetched) /
                                   static_cast<double>(item.fetch_total));
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), item.fetch_result.c_str());
        } else if (!item.on_disk && item.catalogued) {
            if (ImGui::Button("Fetch") &&
                !action.fetch(build, state.models_directory(), item.name)) {
                toasts.add(action.refusal(), toast::Severity::error, now);
            }
        } else if (item.on_disk && !item.active && item.catalogued) {
            ImGui::TextDisabled("The model is on disk but is not active in the manifest.");
            if (ImGui::Button("Activate") &&
                !action.activate(build, state.models_directory(), item.role, item.name)) {
                toasts.add(action.refusal(), toast::Severity::error, now);
            }
        }
        if (!item.catalogued && !item.active) {
            ImGui::TextWrapped("Assign a role in the model store manifest before use.");
        }
        if (item.active && item.on_disk) {
            if (ImGui::Button("Load")) {
                client.send_line("model load " + item.role + " " + item.name);
            }
        }
        if (!item.fetch_result.empty() && !item.fetching) {
            ImGui::TextDisabled("Fetch: %s", item.fetch_result.c_str());
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    if (state.models().empty()) {
        ImGui::TextDisabled("The model sources have no readable entries.");
    }
    if (!state.agents().empty()) {
        const std::size_t selected = state.selected_agent();
        const unsigned agent = state.agents()[selected].id;
        initialize_panel(panel, state, agent);
        draw_parameter_controls(panel, state, client, agent);
        draw_presets(panel, state, client, agent);
        draw_conduct(panel, state, client, agent);
    } else {
        ImGui::TextDisabled("Model controls need an active agent.");
    }
    if (!panel.result.empty()) ImGui::TextWrapped("%s", panel.result.c_str());
    ImGui::End();
}

} // namespace aotx::ctrl::model
