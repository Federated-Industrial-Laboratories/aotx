// Purpose: Define affect setting dials and their live figure state.
// Owns: Editable values for the thirteen affect and quality settings.
// Launch shape: One interface thread draws one settings window.
// Lifetime: Values remain until the selected instance changes.
#ifndef AOTX_CTRL_AFFECT_DIALS_HPP
#define AOTX_CTRL_AFFECT_DIALS_HPP

#include <array>

namespace aotx::ctrl::client { class Client; }
namespace aotx::ctrl::replica { class State; }
namespace aotx::ctrl::toast { class Lane; }

namespace aotx::ctrl::affect::dials {

enum class Setting : unsigned {
    affect_on,
    quality_on,
    probe_gain,
    decay_fast,
    decay_slow,
    gain_fast,
    gain_slow,
    cap_valence,
    cap_arousal,
    temperature_gain,
    voice_gain,
    steer_gain,
    budget
};

enum class WindowState { off, no_figures, calibration_only, trace_only, ready };

struct FigureAvailability {
    bool calibration = false;
    bool probes = false;
    bool trace = false;
    bool budget_spent = false;
    bool entropy_shift = false;
    bool class_shift = false;
};

struct State {
    std::array<float, 13> values{};
    bool initialized = false;
};

void draw(State &view, replica::State &state, client::Client &client,
          toast::Lane &toasts, double now, bool *open);
WindowState window_state(bool affect_on, bool quality_on,
                         const FigureAvailability &figures);
const char *window_sentence(WindowState state);
bool control_enabled(Setting setting, const FigureAvailability &figures);

} // namespace aotx::ctrl::affect::dials

#endif
