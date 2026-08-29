# The disk-side programs and library. C only; no CUDA dependency.

# The library that every host side program links. It holds the ring, the record and the
# journal segment. It also holds the one writer and the one reader of a derived line.
add_library(aotx_disk STATIC
    wire/args.c
    wire/clock.c
    wire/crc32c.c
    wire/jsonread.c
    wire/jsontext.c
    wire/map.c
    wire/record.c
    wire/request_line.c
    wire/ring.c
    wire/segment.c
    wire/sha256.c)
target_include_directories(aotx_disk PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_definitions(aotx_disk PUBLIC _GNU_SOURCE)
set(AOTX_DISK_WARNINGS -Wall -Wextra)
target_compile_options(aotx_disk PRIVATE ${AOTX_DISK_WARNINGS})

# The checksum has a second path that needs one more instruction set. The file that holds
# that path is the only file built with the instruction set, so no other code can take it.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86")
    target_sources(aotx_disk PRIVATE wire/crc32c_hw.c)
    set_source_files_properties(wire/crc32c_hw.c PROPERTIES COMPILE_OPTIONS "-msse4.2")
    target_compile_definitions(aotx_disk PRIVATE AOTX_CRC32C_HW)
endif()

# The settings reader. The host glue links it at boot, and the feeder links it to publish
# the settings that the device applies.
add_library(aotx_settings STATIC settings/settings.c)
target_include_directories(aotx_settings PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_definitions(aotx_settings PUBLIC _GNU_SOURCE)
target_compile_options(aotx_settings PRIVATE ${AOTX_DISK_WARNINGS})

# The model file reader. The host glue links it to load the weights of a model.
add_library(aotx_modelfile STATIC
    modelfile/gguf.c
    modelfile/meta.c
    modelfile/manifest.c)
target_link_libraries(aotx_modelfile PUBLIC aotx_disk)
target_compile_options(aotx_modelfile PRIVATE ${AOTX_DISK_WARNINGS})

add_executable(aotx_drain drain/drain.c drain/derive.c drain/derive_bus.c
                          drain/derive_manifest.c drain/transcript.c drain/bulk.c)
add_executable(aotx_feed feed/feed.c feed/fs_tool.c feed/fs_path.c feed/fs_write.c
                         feed/run_tool.c feed/modules.c feed/import.c feed/attach.c feed/line.c)
add_executable(aotx_restore restore/restore.c restore/scan.c)
add_executable(aotx_journal journal/journal.c journal/chain.c drain/transcript.c
                            restore/scan.c)
add_executable(aotx_manifest manifest/manifest.c)

foreach(program aotx_drain aotx_feed aotx_restore aotx_journal)
    target_link_libraries(${program} PRIVATE aotx_disk)
    target_compile_options(${program} PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()

# The feeder publishes the settings of the run, the drain and the journal reader print
# them, so each of the three links the settings reader.
target_link_libraries(aotx_feed PRIVATE aotx_settings)
target_link_libraries(aotx_drain PRIVATE aotx_settings)
target_link_libraries(aotx_journal PRIVATE aotx_settings)

target_link_libraries(aotx_manifest PRIVATE aotx_modelfile)
target_compile_options(aotx_manifest PRIVATE ${AOTX_DISK_WARNINGS})

# The terminal program. C only, with no CUDA symbol: it reads the mirror over a descriptor
# the feeder sends and reads the disk directly.
add_library(aotx_tui_core STATIC
    tui/term.c
    tui/keys.c
    tui/raster.c
    tui/frame.c
    tui/splash.c
    tui/session.c
    tui/screens.c
    tui/screen_agents.c
    tui/screen_store.c
    tui/screen_settings.c
    tui/screen_system.c)
target_sources(aotx_tui_core PRIVATE tui/screen_session.c)
target_link_libraries(aotx_tui_core PUBLIC aotx_disk aotx_settings)
target_compile_definitions(aotx_tui_core PUBLIC AOTX_VERSION="${PROJECT_VERSION}")
target_compile_options(aotx_tui_core PRIVATE ${AOTX_DISK_WARNINGS})

add_executable(aotx_tui tui/tui.c)
target_link_libraries(aotx_tui PRIVATE aotx_tui_core)
target_compile_options(aotx_tui PRIVATE ${AOTX_DISK_WARNINGS})

# A bus line validator, when one is given, examines the file that the drain derives.
set(AOTX_BUS_LINT "" CACHE STRING "Program that validates a bus line file.")

foreach(case crc32c segment hostring inbound drain derive bulk feed restore journal)
    add_executable(aotx_${case}_test ../tests/${case}_test.c)
    target_link_libraries(aotx_${case}_test PRIVATE aotx_disk)
    target_compile_options(aotx_${case}_test PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()

# The check of the import calls the feeder part directly, so it builds that file with the
# check and needs no child process.
add_executable(aotx_import_test ../tests/import_test.c feed/import.c feed/fs_tool.c
                                feed/fs_path.c feed/fs_write.c feed/run_tool.c
                                feed/modules.c)
target_link_libraries(aotx_import_test PRIVATE aotx_disk)
target_compile_options(aotx_import_test PRIVATE ${AOTX_DISK_WARNINGS})
add_test(NAME disk_import COMMAND aotx_import_test)

# The checks of the host tools drive the real feeder over a fixture tree, so they need the
# feeder and no other program.
add_executable(aotx_fs_tools_test ../tests/fs_tools_test.c)
target_link_libraries(aotx_fs_tools_test PRIVATE aotx_disk)
target_compile_options(aotx_fs_tools_test PRIVATE ${AOTX_DISK_WARNINGS})
add_test(NAME disk_fs_tools COMMAND aotx_fs_tools_test $<TARGET_FILE:aotx_feed>)

add_executable(aotx_run_tool_test ../tests/run_tool_test.c)
target_link_libraries(aotx_run_tool_test PRIVATE aotx_disk)
target_compile_options(aotx_run_tool_test PRIVATE ${AOTX_DISK_WARNINGS})
add_test(NAME disk_run_tool COMMAND aotx_run_tool_test $<TARGET_FILE:aotx_feed>)

# The settings check reads and writes files of its own and needs no other program.
add_executable(aotx_settings_test ../tests/settings_test.c)
target_link_libraries(aotx_settings_test PRIVATE aotx_settings aotx_disk)
target_compile_options(aotx_settings_test PRIVATE ${AOTX_DISK_WARNINGS})
add_test(NAME disk_settings COMMAND aotx_settings_test)

# The checks of the feeder, the drain and the journal reader give the text of a scaled
# value. Each of them links the settings reader.
target_link_libraries(aotx_feed_test PRIVATE aotx_settings)
target_link_libraries(aotx_derive_test PRIVATE aotx_settings)
target_link_libraries(aotx_journal_test PRIVATE aotx_settings)

add_test(NAME disk_crc32c COMMAND aotx_crc32c_test)
add_test(NAME disk_segment COMMAND aotx_segment_test)
add_test(NAME disk_hostring COMMAND aotx_hostring_test)
add_test(NAME disk_inbound COMMAND aotx_inbound_test)
add_test(NAME disk_drain COMMAND aotx_drain_test $<TARGET_FILE:aotx_drain> ${AOTX_BUS_LINT})
add_test(NAME disk_derive COMMAND aotx_derive_test $<TARGET_FILE:aotx_drain> ${AOTX_BUS_LINT})
add_test(NAME disk_bulk COMMAND aotx_bulk_test $<TARGET_FILE:aotx_drain>)
add_test(NAME disk_feed COMMAND aotx_feed_test $<TARGET_FILE:aotx_feed>
                                             $<TARGET_FILE:aotx_drain>
                                             $<TARGET_FILE:aotx_journal> ${AOTX_BUS_LINT})
add_test(NAME disk_restore COMMAND aotx_restore_test $<TARGET_FILE:aotx_restore>)
add_test(NAME disk_journal COMMAND aotx_journal_test $<TARGET_FILE:aotx_journal> $<TARGET_FILE:aotx_drain>)

# The checks of the socket and of the terminal program. Each one builds against the
# library of the terminal, so the check calls the same code the program runs.
add_executable(aotx_attach_test ../tests/attach_test.c feed/attach.c feed/line.c)
target_link_libraries(aotx_attach_test PRIVATE aotx_disk)
target_compile_options(aotx_attach_test PRIVATE ${AOTX_DISK_WARNINGS})
add_test(NAME disk_attach COMMAND aotx_attach_test)

add_executable(aotx_parts_test ../tests/parts_test.c feed/line.c)
target_link_libraries(aotx_parts_test PRIVATE aotx_disk Threads::Threads)
target_compile_options(aotx_parts_test PRIVATE ${AOTX_DISK_WARNINGS})
add_test(NAME disk_parts COMMAND aotx_parts_test)

add_executable(aotx_transcript_test ../tests/transcript_test.c)
target_link_libraries(aotx_transcript_test PRIVATE aotx_disk)
target_compile_options(aotx_transcript_test PRIVATE ${AOTX_DISK_WARNINGS})
add_test(NAME disk_transcript COMMAND aotx_transcript_test $<TARGET_FILE:aotx_drain>
                                               $<TARGET_FILE:aotx_journal>)

foreach(case keys raster splash screens)
    add_executable(aotx_${case}_tui_test ../tests/${case}_test.c)
    target_link_libraries(aotx_${case}_tui_test PRIVATE aotx_tui_core)
    target_compile_options(aotx_${case}_tui_test PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()

# The check of the read of the mirror drives a thread that publishes frames beside it.
find_package(Threads REQUIRED)
target_link_libraries(aotx_raster_tui_test PRIVATE Threads::Threads)

add_test(NAME disk_keys COMMAND aotx_keys_tui_test)
add_test(NAME disk_raster_tui COMMAND aotx_raster_tui_test)
add_test(NAME disk_splash COMMAND aotx_splash_tui_test ${CMAKE_SOURCE_DIR}/share/splash)
add_test(NAME disk_screens COMMAND aotx_screens_tui_test)

# The model file tests read the model files when they are there, and report a skip when
# they are not. The directory of the model files is given to each test.
set(AOTX_MODELS_DIR ${CMAKE_SOURCE_DIR}/models CACHE STRING "Directory that holds the model files.")

foreach(case sha256 gguf manifest modelfile)
    add_executable(aotx_${case}_test ../tests/${case}_test.c)
    target_link_libraries(aotx_${case}_test PRIVATE aotx_modelfile)
    target_compile_options(aotx_${case}_test PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()

add_test(NAME disk_sha256 COMMAND aotx_sha256_test ${AOTX_MODELS_DIR})
add_test(NAME disk_gguf COMMAND aotx_gguf_test ${AOTX_MODELS_DIR})
add_test(NAME disk_manifest COMMAND aotx_manifest_test $<TARGET_FILE:aotx_manifest> ${AOTX_MODELS_DIR})
add_test(NAME disk_modelfile COMMAND aotx_modelfile_test ${AOTX_MODELS_DIR})
