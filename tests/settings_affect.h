/* Purpose: Give the settings reader the optional key rows and refusal rows.
 * Owns: Nothing; the settings check includes the rows in its fixture tables.
 * Threading: One thread; constants only.
 * Lifetime: The run of the settings check. */
#ifndef AOTX_TEST_SETTINGS_AFFECT_H
#define AOTX_TEST_SETTINGS_AFFECT_H

#ifdef AOTX_AFFECT
#define AOTX_TEST_AFFECT_WANT_NUMBERS \
    { "affect.on", 0, 0, 1, 1, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "quality.on", 0, 0, 1, 1, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.probe_gain", 0, 0, 10000, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.decay_fast", 5000, 0, 9900, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.decay_slow", 9000, 0, 9900, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.gain_fast", 5000, 0, 20000, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.gain_slow", 1000, 0, 20000, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.cap_valence", 10000, 0, 10000, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.cap_arousal", 10000, 0, 10000, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.temperature_gain", 0, -10000, 10000, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.voice_gain", 0, -10000, 10000, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.steer_gain", 0, 0, 10000, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }, \
    { "affect.budget", 2500, 0, 40000, 10000, AOTX_SETTING_SIDE_DEVICE, AOTX_SETTING_AT_SEQUENCE }

#define AOTX_TEST_AFFECT_REFUSALS \
    { "affect.on = 2", "the value 2 is not in the range 0 to 1" }, \
    { "quality.on = 2", "the value 2 is not in the range 0 to 1" }, \
    { "affect.probe_gain = 2", "the value 2 is not in the range 0 to 1" }, \
    { "affect.decay_fast = 1", "the value 1 is not in the range 0 to 0.99" }, \
    { "affect.decay_slow = 1", "the value 1 is not in the range 0 to 0.99" }, \
    { "affect.gain_fast = 3", "the value 3 is not in the range 0 to 2" }, \
    { "affect.gain_slow = 3", "the value 3 is not in the range 0 to 2" }, \
    { "affect.cap_valence = 2", "the value 2 is not in the range 0 to 1" }, \
    { "affect.cap_arousal = 2", "the value 2 is not in the range 0 to 1" }, \
    { "affect.temperature_gain = 2", "the value 2 is not in the range -1 to 1" }, \
    { "affect.voice_gain = 2", "the value 2 is not in the range -1 to 1" }, \
    { "affect.steer_gain = 2", "the value 2 is not in the range 0 to 1" }, \
    { "affect.budget = 5", "the value 5 is not in the range 0 to 4" }
#else
#define AOTX_TEST_AFFECT_WANT_NUMBERS
#define AOTX_TEST_AFFECT_REFUSALS \
    { "affect.on = 1", "the key is not known" }, \
    { "quality.on = 1", "the key is not known" }, \
    { "affect.probe_gain = 0", "the key is not known" }, \
    { "affect.decay_fast = 0.5", "the key is not known" }, \
    { "affect.decay_slow = 0.9", "the key is not known" }, \
    { "affect.gain_fast = 0.5", "the key is not known" }, \
    { "affect.gain_slow = 0.1", "the key is not known" }, \
    { "affect.cap_valence = 1", "the key is not known" }, \
    { "affect.cap_arousal = 1", "the key is not known" }, \
    { "affect.temperature_gain = 0", "the key is not known" }, \
    { "affect.voice_gain = 0", "the key is not known" }, \
    { "affect.steer_gain = 0", "the key is not known" }, \
    { "affect.budget = 0.25", "the key is not known" }
#endif

#endif
