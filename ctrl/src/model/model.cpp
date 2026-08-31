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

} // namespace aotx::ctrl::model
