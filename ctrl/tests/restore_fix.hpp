// Purpose: Declare checks for fresh and restored instance launch arguments.
// Owns: No state outside the check call.
// Threading: One host thread runs batches of 1 and 64 instance fixtures.
// Lifetime: All fixture resources end before return.
#ifndef AOTX_CTRL_TESTS_RESTORE_FIX_HPP
#define AOTX_CTRL_TESTS_RESTORE_FIX_HPP
void aotx_ctrl_restore_fix(int &applied, int &failed);
#endif
