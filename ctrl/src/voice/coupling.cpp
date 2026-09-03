// Purpose: Map an effective state to bounded speech-engine controls.
// Owns: The fixed gains and caps of the spoken voice coupling.
// Launch shape: One host call maps one utterance state.
// Lifetime: No value remains after the caller releases its result.
#include "voice/coupling.hpp"

#include <algorithm>

namespace aotx::ctrl::voice {

CouplingValues spoken_voice_values(double valence, double arousal)
{
    const double held_valence = std::clamp(valence, -1.0, 1.0);
    const double held_arousal = std::clamp(arousal, -1.0, 1.0);
    CouplingValues values;
    values.length = std::clamp(1.0 - 0.25 * held_arousal, 0.75, 1.25);
    values.noise = std::clamp(0.667 + 0.167 * held_arousal, 0.5, 0.834);
    values.width = std::clamp(0.8 + 0.1 * held_valence, 0.7, 0.9);
    return values;
}

} // namespace aotx::ctrl::voice
