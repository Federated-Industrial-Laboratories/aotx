// Purpose: Classify Dials window states and figure requirements.
// Owns: No state outside each classification call.
// Launch shape: One host call classifies one window or control.
// Lifetime: Returned values have static storage or value lifetime.
#include "affect/dials.hpp"

namespace aotx::ctrl::affect::dials {

WindowState window_state(bool affect_on, bool quality_on,
                         const FigureAvailability &figures)
{
    if (!affect_on && !quality_on) return WindowState::off;
    if (figures.calibration && figures.trace) return WindowState::ready;
    if (figures.calibration) return WindowState::calibration_only;
    if (figures.trace) return WindowState::trace_only;
    return WindowState::no_figures;
}

const char *window_sentence(WindowState state)
{
    switch (state) {
    case WindowState::off:
        return "The affect substrate and the quality stream are off. Set affect.on or quality.on to 1 to start one.";
    case WindowState::no_figures:
        return "No calibration or trace figures are loaded, so the related controls are gray.";
    case WindowState::calibration_only:
        return "Calibration figures are loaded, but trace figures are absent, so trace controls are gray.";
    case WindowState::trace_only:
        return "Trace figures are loaded, but calibration figures are absent, so calibration controls are gray.";
    case WindowState::ready:
        return "Calibration and trace figures are loaded beside their controls.";
    }
    return "No figure state is available.";
}

bool control_enabled(Setting setting, const FigureAvailability &figures)
{
    switch (setting) {
    case Setting::affect_on:
    case Setting::quality_on:
        return true;
    case Setting::probe_gain:
        return figures.probes;
    case Setting::decay_fast:
    case Setting::decay_slow:
    case Setting::gain_fast:
    case Setting::gain_slow:
    case Setting::cap_valence:
    case Setting::cap_arousal:
        return figures.trace;
    case Setting::temperature_gain:
        return figures.entropy_shift;
    case Setting::voice_gain:
        return figures.class_shift;
    case Setting::steer_gain:
        return figures.calibration;
    case Setting::budget:
        return figures.budget_spent;
    }
    return false;
}

} // namespace aotx::ctrl::affect::dials
