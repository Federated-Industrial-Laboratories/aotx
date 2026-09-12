/* Purpose: Load the vocabulary descriptors of a model set.
 * Owns: The host interface to the vocabulary stores.
 * Launch shape: Host glue only; text checks run on the device.
 * Lifetime: From model load to the end of the run. */
#ifndef AOTX_BOOT_VOCAB_HOST_H
#define AOTX_BOOT_VOCAB_HOST_H
extern "C" {
#include "disk/modelfile/modelfile.h"
}
int aotx_boot_vocab_family(const aotx_modelfile *file, const char *name, unsigned int *row);
int aotx_boot_vocab_take(const aotx_modelfile *file, const char *name, int build, int embedding);
void aotx_boot_vocab_finish(int audio_default);
void aotx_boot_vocab_release(void);
#endif
