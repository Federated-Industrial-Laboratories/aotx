// Purpose: Declare the spoken voice coupling check.
// Owns: No test state outside one call.
// Launch shape: One host thread checks neutral and capped states.
// Lifetime: All values end before the call returns.
#ifndef AOTX_CTRL_TESTS_VOICE_FIX_HPP
#define AOTX_CTRL_TESTS_VOICE_FIX_HPP

void aotx_ctrl_voice_fix(int &applied, int &failed);

#endif
