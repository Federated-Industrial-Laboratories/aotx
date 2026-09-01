// Purpose: Draw timed notifications with severity colors.
// Owns: Notification placement and automatic dismissal.
// Launch shape: One lane draws at the upper right of the main viewport.
// Lifetime: Each notification has a fixed display interval.
#include "toast/toast.hpp"

#include "imgui.h"
#include "theme/theme.hpp"
#include "voice/voice.hpp"

#include <algorithm>

namespace aotx::ctrl::toast {
namespace {

ImVec4 severity_color(Severity severity)
{
    const theme::Palette &colors = theme::palette();
    switch (severity) {
    case Severity::success: return colors.severity_success;
    case Severity::warning: return colors.severity_warning;
    case Severity::error: return colors.severity_error;
    case Severity::info: return colors.severity_info;
    }
    return colors.severity_info;
}

} // namespace

Lane::Lane(voice::Queue *speech) : speech_(speech) {}

void Lane::add(std::string text, Severity severity, double now, double seconds)
{
    /* Routine notices stay visual; only a warning or an error takes the voice channel. */
    if (speech_ != nullptr && (severity == Severity::warning || severity == Severity::error)) {
        speech_->speak(voice::Source::system(), text);
    }
    notices_.push_back({std::move(text), severity, now + seconds});
}

void Lane::draw(double now)
{
    notices_.erase(std::remove_if(notices_.begin(), notices_.end(),
                                  [now](const Notice &notice) {
                                      return notice.expires_at <= now;
                                  }),
                   notices_.end());
    if (notices_.empty()) {
        return;
    }

    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - 18.0f,
                                   viewport->WorkPos.y + 18.0f),
                            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.96f);
    ImGui::Begin("Notifications", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                 ImGuiWindowFlags_NoFocusOnAppearing);
    for (const Notice &notice : notices_) {
        ImGui::TextColored(severity_color(notice.severity), "%s", notice.text.c_str());
    }
    ImGui::End();
}

} // namespace aotx::ctrl::toast
