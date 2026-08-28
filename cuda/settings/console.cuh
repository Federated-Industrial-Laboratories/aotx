/* Purpose: Give the console the two commands that read and change the settings table.
 * Owns: Nothing; the table and the console buffer hold the state.
 * Launch shape: One thread; the apply step calls these in slot order.
 * Lifetime: The whole run. */
#ifndef AOTX_SETTINGS_CONSOLE_CUH
#define AOTX_SETTINGS_CONSOLE_CUH

#include "cli/cli.cuh"
#include "settings/settings.cuh"

/* Act on a set line. The key and the value are the two words after the command word. The
 * command builds a setting record and publishes it as a class A record of the console. It
 * then folds the record into the state hash and applies it. A key or a value the table
 * refuses writes the reason and changes nothing.
 *
 * A replay of the journal makes the change from the record and not from the line. A
 * restored run therefore folds each setting body one time. */
__device__ void aotx_settings_set_command(aotx_cli_out *out, const char *key,
                                          unsigned int key_len, const char *value,
                                          unsigned int value_len, unsigned long long tick);

/* Write one line for each setting the device reads: the key, the value and the effect. */
__device__ void aotx_settings_show_command(aotx_cli_out *out);

#endif
