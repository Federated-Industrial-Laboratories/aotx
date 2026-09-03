// Purpose: Define the spoken voice coupling values for one utterance.
// Owns: Fixed gains, neutral values, and hard output caps.
// Launch shape: One host call maps one agent state to three values.
// Lifetime: Returned values live in the caller.
#ifndef AOTX_CTRL_VOICE_COUPLING_HPP
#define AOTX_CTRL_VOICE_COUPLING_HPP

namespace aotx::ctrl::voice {

struct CouplingValues {
    double length = 1.0;
    double noise = 0.667;
    double width = 0.8;
};

CouplingValues spoken_voice_values(double valence, double arousal);

} // namespace aotx::ctrl::voice

#endif
