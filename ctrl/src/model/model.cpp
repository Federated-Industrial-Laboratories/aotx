// Purpose: List, fetch, and activate simulated model entries.
// Owns: Model action buttons and their result notifications.
// Launch shape: One panel draws all catalog entries each frame.
// Lifetime: Fetch progress advances in the simulated state.
#include "model/model.hpp"

#include "imgui.h"

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
            if (ImGui::Button("Use for language")) activate(state, toasts, index, "language", now);
            ImGui::SameLine();
            if (ImGui::Button("Use for embedding")) activate(state, toasts, index, "embedding", now);
            ImGui::SameLine();
            if (ImGui::Button("Use for reranker")) {
                activate(state, toasts, index, "reranker", now);
            }
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

void draw(replica::State &state, client::Client &client, toast::Lane &toasts,
          double now, bool *open)
{
    (void)toasts;
    (void)now;
    if (!ImGui::Begin("Models", open)) {
        ImGui::End();
        return;
    }
    ImGui::Text("Store: %s", state.models_directory().string().c_str());
    ImGui::Text("Resident language: %s", state.language_model().c_str());
    ImGui::SeparatorText("Active manifest roles");
    bool assigned = false;
    for (const replica::Model &item : state.models()) {
        if (item.active) {
            ImGui::Text("%s: %s", item.role.c_str(), item.name.c_str());
            assigned = true;
        }
    }
    if (!assigned) ImGui::TextDisabled("The manifest has no active roles.");
    ImGui::SeparatorText("Catalog");
    for (const replica::Model &item : state.models()) {
        ImGui::PushID(item.name.c_str());
        ImGui::TextUnformatted(item.name.c_str());
        ImGui::TextDisabled("%s  %s  %llu bytes", item.role.c_str(), item.quant.c_str(),
                            static_cast<unsigned long long>(item.bytes));
        if (item.fetching) {
            const float progress = item.fetch_total == 0u ? 0.0f :
                static_cast<float>(static_cast<double>(item.fetched) /
                                   static_cast<double>(item.fetch_total));
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f), item.fetch_result.c_str());
        } else if (!item.on_disk) {
            if (ImGui::Button("Fetch")) client.send_line("model fetch " + item.name);
        } else if (!item.active) {
            ImGui::TextDisabled("The model is on disk but is not active in the manifest.");
        }
        if (item.active) {
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
        ImGui::TextDisabled("The catalog has no readable entries.");
    }
    ImGui::End();
}

} // namespace aotx::ctrl::model
