// Purpose: Guide detect, build, model, activation, start, and first-say actions.
// Owns: Modal sequence controls and result notifications.
// Launch shape: One modal advances through six ordered actions.
// Lifetime: Completion closes the current first-run sequence.
#include "wizard/wizard.hpp"

#include "replica/store.hpp"

#include "imgui.h"

#include <array>
#include <utility>

namespace aotx::ctrl::wizard {

namespace {

std::size_t reply_count(const replica::State *state)
{
    if (state == nullptr) return 0u;
    std::size_t count = 0u;
    for (const replica::Agent &agent : state->agents()) {
        for (const replica::TranscriptEvent &event : agent.transcript) {
            if (event.kind == "reply") ++count;
        }
    }
    return count;
}

void add_phase(LiveState &view, const std::string &phase)
{
    if ((phase != "placing" && phase != "replaying" && phase != "running") ||
        phase == view.last_phase) return;
    view.last_phase = phase;
    if (!view.phase_trace.empty()) view.phase_trace += " -> ";
    view.phase_trace += phase;
}

} // namespace

// Each first-run sequence reads the store that its new instance will use.
void refresh_store(LiveState &view, double now, bool force = false)
{
    const std::string path = view.models_path.data();
    if (!force && path == view.store_path && now < view.store_read_at) return;
    view.store_path = path;
    view.store_read_at = now + 1.0;
    std::string reason;
    std::vector<replica::Model> models;
    const bool readable = replica::store::read(AOTX_CTRL_MODEL_CATALOG, path, models, reason);
    view.selection.refresh(readable, std::move(models), std::move(reason));
}

void draw(LiveState &view, DetectAction &detect, model::StoreAction &model_action,
          instances::Lifecycle &lifecycle, replica::State &state, toast::Lane &toasts,
          double now, bool *open)
{
    (void)state;
    if (*open && !ImGui::IsPopupOpen("First run")) ImGui::OpenPopup("First run");
    bool popup_open = true;
    if (!ImGui::BeginPopupModal("First run", &popup_open,
                                ImGuiWindowFlags_AlwaysAutoResize)) {
        if (!popup_open) *open = false;
        return;
    }
    static const std::array<const char *, 6> titles = {
        "Detect", "Build", "Model", "Activate", "Start", "First say"};
    if (view.page >= titles.size()) view.page = 0u;

    /* Each page runs its own work and reports every frame. The Continue control stays
     * disabled until the page completes, so the order always holds. */
    if (view.entered != view.page) {
        view.entered = view.page;
        view.action.reset();
        if (view.page == static_cast<unsigned>(Page::start)) {
            view.last_phase.clear();
            view.phase_trace.clear();
        }
        if (view.page == static_cast<unsigned>(Page::first_say)) {
            view.reply_count = reply_count(lifecycle.replica(view.instance_index));
        }
    }
    detect.tick();
    model_action.tick();
    ImGui::Text("%u of %zu", view.page + 1u, titles.size());
    ImGui::SeparatorText(titles[view.page]);
    bool ready = false;
    std::string status;

    if (view.page == 0u) {
        if (!detect.ready() && !detect.running() && view.action.begin()) {
            const std::filesystem::path script =
                std::filesystem::path(view.build_path.data()).parent_path() /
                "tools/profile-detect.sh";
            if (!detect.start(script)) view.action.complete(false);
        }
        if (view.action.active() && !detect.running()) view.action.complete(detect.ready());
        GateFacts facts;
        facts.detected = detect.ready();
        ready = gate_open(Page::detect, facts);
        status = detect.running() ? "The card detection runs." : detect.result();
        if (!ready && !detect.running() && ImGui::Button("Retry")) view.action.reset();
        if (!ready && !detect.running()) {
            ImGui::SameLine();
            ImGui::TextDisabled("Run the card detection again.");
        }
    } else if (view.page == 1u) {
        ImGui::InputText("Build directory", view.build_path.data(), view.build_path.size());
        ImGui::InputInt("Card", &view.card);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Enter the zero-based card number for the new instance.");
        const std::filesystem::path build = view.build_path.data();
        GateFacts facts;
        facts.build_ready = view.card >= 0 && std::filesystem::is_regular_file(build / "aotx_boot") &&
                            std::filesystem::is_regular_file(build / "aotx_models");
        ready = gate_open(Page::build, facts);
        status = ready ? "The build directory is ready."
                       : view.card < 0 ? "The card number must be zero or greater."
                                       : "The build directory does not hold the programs.";
    } else if (view.page == 2u) {
        refresh_store(view, now);
        const replica::Model *selected = view.selection.selected();
        const std::string preview = selected == nullptr ? "Select a language model"
            : selected->name + " [" + selected->role + "]";
        ImGui::BeginDisabled(view.action.active() || model_action.running());
        if (ImGui::BeginCombo("Language model", preview.c_str())) {
            for (const auto &row : view.selection.rows()) {
                if (!language_role(row.role)) continue;
                const std::string identity = replica::store::key(row);
                const std::string label = row.name + " [" + row.role + "]";
                ImGui::PushID(identity.c_str());
                if (ImGui::Selectable(label.c_str(), identity == view.selection.identity())) {
                    view.selection.select(identity);
                    view.action.reset();
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        selected = view.selection.selected();
        if (view.action.active() && selected != nullptr && selected->on_disk) {
            view.action.complete(true);
        } else if (view.action.active() && (selected == nullptr ||
                   (model_action.finished() && !model_action.succeeded()))) {
            view.action.complete(false);
        }
        if (selected == nullptr) {
            status = view.selection.refusal();
        } else if (selected->on_disk) {
            status = selected->name + " is on disk.";
        } else if (model_action.running()) {
            status = model_action.progress().empty() ? "The model fetch runs."
                                                     : model_action.progress();
        } else if (!can_fetch(selected)) {
            status = "The local model file is absent. Restore the file in this model directory.";
        } else if (view.action.active() && model_action.succeeded()) {
            status = "The model fetch completed. The catalog refresh is pending.";
        } else if (view.action.failed()) {
            if (ImGui::Button("Retry")) view.action.reset();
            status = model_action.progress().empty() ? "The model fetch did not complete."
                                                     : model_action.progress();
        } else {
            if (ImGui::Button("Fetch") && view.action.begin()) {
                if (!model_action.fetch(view.build_path.data(), view.models_path.data(), selected->name)) {
                    view.action.complete(false);
                    toasts.add(model_action.refusal(), toast::Severity::error, now);
                }
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Fetch the selected model to disk.");
            status = "The model is not on disk. Fetch it to continue.";
        }
        GateFacts facts;
        facts.catalog_read = view.selection.readable();
        facts.model_on_disk = selected != nullptr && selected->on_disk;
        ready = gate_open(Page::model, facts);
    } else if (view.page == 3u) {
        refresh_store(view, now);
        const replica::Model *selected = view.selection.selected();
        if (view.action.active() && (selected == nullptr || !selected->on_disk))
            view.action.complete(false);
        if (selected == nullptr) {
            status = view.selection.refusal();
        } else if (!selected->on_disk) {
            status = "The selected model file is no longer on disk.";
        } else if (selected->active) {
            if (view.action.active()) view.action.complete(true);
            status = selected->name + " is active for " + selected->role + ".";
        } else if (model_action.running()) {
            status = "The model activation runs.";
        } else if (!can_activate(selected)) {
            status = "This local model needs an active manifest entry. Set the entry before startup.";
        } else if (view.action.active() && model_action.finished() && !model_action.succeeded()) {
            view.action.complete(false);
            status = "The model activation did not complete.";
        } else if (view.action.active() && model_action.succeeded()) {
            status = "The model activation completed. The catalog refresh is pending.";
        } else if (view.action.begin()) {
            if (!model_action.activate(view.build_path.data(), view.models_path.data(),
                                       selected->role, selected->name)) {
                view.action.complete(false);
                toasts.add(model_action.refusal(), toast::Severity::error, now);
            }
            status = "The model activation started.";
        } else {
            status = "The activation did not complete.";
            if (view.action.failed() && ImGui::Button("Retry")) view.action.reset();
        }
        const ModelRoles roles = model_roles(view.selection);
        if (selected != nullptr && selected->active && !roles.ready()) status = roles.refusal;
        if (!roles.notice.empty()) ImGui::TextWrapped("%s", roles.notice.c_str());
        GateFacts facts;
        facts.catalog_read = view.selection.readable();
        facts.model_active = roles.ready();
        ready = gate_open(Page::activate, facts);
    } else if (view.page == 4u) {
        ImGui::BeginDisabled(view.instance_created || view.action.active());
        ImGui::InputText("Instance name", view.instance_name.data(), view.instance_name.size());
        if (ImGui::TreeNode("Locations")) {
            ImGui::InputText("Journal directory", view.journal_path.data(),
                             view.journal_path.size());
            ImGui::InputText("Settings file", view.settings_path.data(),
                             view.settings_path.size());
            ImGui::InputText("Model directory", view.models_path.data(),
                             view.models_path.size());
            ImGui::InputText("Tool root", view.tools_path.data(), view.tools_path.size());
            ImGui::TreePop();
        }
        ImGui::EndDisabled();
        if (!view.instance_created) refresh_store(view, now);
        ModelRoles roles = model_roles(view.selection);
        if (!view.instance_created && roles.ready() && !view.action.active() && !view.action.failed()) {
            refresh_store(view, now, true);
            roles = model_roles(view.selection);
        }
        if (!roles.notice.empty()) ImGui::TextWrapped("%s", roles.notice.c_str());
        if (!view.instance_created) {
            ImGui::Text("Card: %d", view.card);
            if (roles.ready()) ImGui::Text("Startup roles: %s", roles.roles.c_str());
            if (!roles.ready()) view.result = roles.refusal;
            else if (view.card < 0) view.result = "The card number must be zero or greater.";
        }
        if (!view.instance_created && roles.ready() && view.card >= 0 && view.action.begin()) {
            instances::Definition definition;
            definition.name = view.instance_name.data();
            definition.journal = view.journal_path.data();
            definition.settings = view.settings_path.data();
            definition.build = view.build_path.data();
            definition.models = view.models_path.data();
            definition.tools = view.tools_path.data();
            definition.roles = roles.roles;
            definition.card = static_cast<unsigned>(view.card);
            view.instance_index = lifecycle.instances().size();
            if (lifecycle.create(std::move(definition))) {
                view.instance_created = true;
            }
            if (view.instance_created && lifecycle.start(view.instance_index)) {
                toasts.add("The first instance starts.", toast::Severity::info, now);
            } else {
                view.action.complete(false);
                view.result = lifecycle.refusal();
                toasts.add(view.result, toast::Severity::error, now);
            }
        }
        if (view.instance_created && !view.action.active() && !view.action.completed() &&
            !view.action.failed() &&
            view.action.begin() && lifecycle.start(view.instance_index)) {
            view.result.clear();
            toasts.add("The first instance starts.", toast::Severity::info, now);
        } else if (view.instance_created && view.action.active()) {
            const std::vector<instances::LiveInstance> started = lifecycle.instances();
            if (view.instance_index < started.size() && started[view.instance_index].process < 0 &&
                !start_gate_open(started[view.instance_index])) {
                view.action.complete(false);
                view.result = lifecycle.refusal().empty()
                    ? started[view.instance_index].result : lifecycle.refusal();
            }
        }
        const std::vector<instances::LiveInstance> items = lifecycle.instances();
        if (view.instance_created && view.instance_index < items.size()) {
            const instances::LiveInstance &item = items[view.instance_index];
            add_phase(view, item.phase);
            ready = start_gate_open(item);
            if (ready && view.action.active()) view.action.complete(true);
            status = ready ? "The first instance is running and its socket answers."
                           : item.result;
            if (item.process < 0 && !ready && view.action.active()) {
                view.action.complete(false);
            }
        } else if (view.result.empty()) {
            status = "The first instance starts on this page.";
        } else {
            status = view.result;
        }
        if (!view.phase_trace.empty()) {
            ImGui::Text("Phases: %s", view.phase_trace.c_str());
        }
        if (!ready && view.action.failed() && ImGui::Button("Retry")) {
            view.result.clear();
            view.action.reset();
        }
    } else {
        replica::State *started = lifecycle.replica(view.instance_index);
        const std::size_t replies = reply_count(started);
        if (replies > view.reply_count && view.action.active()) view.action.complete(true);
        if (!view.action.completed() && !view.action.active() && view.action.begin()) {
            if (!lifecycle.send(view.instance_index, "say Hello.")) {
                view.action.complete(false);
                view.result = lifecycle.refusal();
            }
        }
        GateFacts facts;
        facts.reply_received = replies > view.reply_count;
        ready = gate_open(Page::first_say, facts);
        status = ready ? "The first reply arrived."
                       : view.action.active() ? "The first reply is pending."
                                              : view.result;
        if (!ready && view.action.failed() && ImGui::Button("Retry")) {
            view.reply_count = replies;
            view.result.clear();
            view.action.reset();
        }
    }

    if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());

    const bool last = view.page + 1u == titles.size();
    ImGui::BeginDisabled(!ready);
    if (ImGui::Button(last ? "Finish" : "Continue")) {
        if (last) {
            toasts.add("The first reply arrived.", toast::Severity::success, now);
            view.page = 0u;
            view.entered = 0xffffffffu;
            *open = false;
            ImGui::CloseCurrentPopup();
        } else {
            ++view.page;
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (view.page > 0u && view.page <= 4u && !view.instance_created &&
        !view.action.active() && !model_action.running()) {
        if (ImGui::Button("Back")) --view.page;
        ImGui::SameLine();
    }
    if (ImGui::Button("Cancel")) {
        view.page = 0u;
        view.entered = 0xffffffffu;
        *open = false;
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
    if (!popup_open) *open = false;
}

} // namespace aotx::ctrl::wizard
