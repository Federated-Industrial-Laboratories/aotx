/* Purpose: Give the parts of the check program the counts and the two arms they share.
 * Owns: Nothing; the counts stand in the main file of the program.
 * Threading: One thread; the check program is the only caller.
 * Lifetime: One run of the check program. */
#ifndef AOTX_CATALOG_CHECK_HOST_H
#define AOTX_CATALOG_CHECK_HOST_H

#include "catalog/check.cuh"

/* The checks applied and the checks that failed. The program gives the exit status from
 * the second one. */
extern unsigned int aotx_check_applied;
extern unsigned int aotx_check_failed;

/* State one check with its figure and count it. */
void aotx_check_say(int ok, const char *what, unsigned long long figure);

/* The device arm: load the module by its digest, read its figures and run the batches. */
void aotx_check_device(const char *dir, unsigned int entry, unsigned int rows,
                       aotx_check_entry *held, aotx_check_entry *on);

/* The host arm: run the program of the module over the example line under the timeout. */
int aotx_check_program(const char *dir, const aotx_check_entry *entry, const char *name,
                       unsigned int *applied, unsigned int *failed);

/* Import the manifest of a directory through the reader of the device and read the entry
 * back. The head carries the digest the caller gives, as the head of the feeder does. */
unsigned int aotx_check_import_dir(const char *dir, aotx_check_entry *out,
                                   aotx_check_entry *on, const unsigned char *digest,
                                   unsigned int number);

#endif
