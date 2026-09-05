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
    modelfile/manifest.c
    modelfile/manifest_json.c
    modelfile/wrap.c)
target_link_libraries(aotx_modelfile PUBLIC aotx_disk)
target_compile_options(aotx_modelfile PRIVATE ${AOTX_DISK_WARNINGS})

add_executable(aotx_wrap_test ../tests/wrap_test.c)
target_link_libraries(aotx_wrap_test PRIVATE aotx_modelfile)
target_compile_options(aotx_wrap_test PRIVATE ${AOTX_DISK_WARNINGS})
add_test(NAME wrap_disk COMMAND aotx_wrap_test)

# Model checks and the store program use this directory when no command line names one.
set(AOTX_MODELS_DIR ${CMAKE_SOURCE_DIR}/models CACHE STRING "Directory that holds the model files.")

# The model store library reads the repository catalog and the local store. Its output
# name is aotx_models, beside the program of that name.
add_library(aotx_models_store STATIC
    models/catalog.c
    models/store.c
    models/parameters.c
    models/inspect.c
    models/inspect_http.c
    models/fetch.c)
set_target_properties(aotx_models_store PROPERTIES OUTPUT_NAME aotx_models)
target_link_libraries(aotx_models_store PUBLIC aotx_modelfile)
target_compile_definitions(aotx_models_store PUBLIC
    AOTX_MODELS_CATALOG="${CMAKE_SOURCE_DIR}/share/models/catalog.jsonl"
    AOTX_MODELS_DIRECTORY="${AOTX_MODELS_DIR}"
    AOTX_VERSION="${PROJECT_VERSION}")
if(AOTX_FETCH)
    target_compile_definitions(aotx_models_store PRIVATE AOTX_FETCH)
    target_link_libraries(aotx_models_store PRIVATE CURL::libcurl)
endif()
target_compile_options(aotx_models_store PRIVATE ${AOTX_DISK_WARNINGS})

add_executable(aotx_models models/main.c)
target_link_libraries(aotx_models PRIVATE aotx_models_store)
target_compile_options(aotx_models PRIVATE ${AOTX_DISK_WARNINGS})

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME disk_inspect COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_SOURCE_DIR}/tests/inspect_test.py $<TARGET_FILE:aotx_models> ${AOTX_FETCH})
set_tests_properties(disk_inspect PROPERTIES TIMEOUT 180)

add_executable(aotx_drain drain/drain.c drain/derive.c drain/derive_mask.c drain/derive_bus.c
                          drain/derive_manifest.c drain/transcript.c drain/token_stats.c
                          drain/page_stats.c
                          drain/bulk.c)
if(AOTX_AFFECT)
    target_sources(aotx_drain PRIVATE drain/affect_derive.c drain/affect_stream.c
                                      drain/quality_stream.c)
endif()
add_executable(aotx_feed feed/feed.c feed/fs_tool.c feed/fs_path.c feed/fs_write.c
                         feed/run_tool.c feed/modules.c feed/import.c feed/attach.c feed/line.c
                         feed/model_fetch.c)
add_executable(aotx_restore restore/restore.c restore/scan.c feed/line.c)
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
    tui/screen_models.c
    tui/screen_store.c
    tui/screen_settings.c
    tui/screen_system.c)
target_sources(aotx_tui_core PRIVATE tui/screen_session.c)
target_link_libraries(aotx_tui_core PUBLIC aotx_disk aotx_settings)
target_link_libraries(aotx_tui_core PUBLIC aotx_models_store)
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

add_executable(aotx_feed_models_test ../tests/feed_models_test.c)
target_link_libraries(aotx_feed_models_test PRIVATE aotx_disk)
target_compile_options(aotx_feed_models_test PRIVATE ${AOTX_DISK_WARNINGS})

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

add_test(NAME terminal_path COMMAND ${CMAKE_SOURCE_DIR}/tests/model_test.sh
         ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_attach_test> --terminal-path
         $<TARGET_FILE:aotx_boot> $<TARGET_FILE:aotx_tui> ${AOTX_MODELS_DIR}
         ${CMAKE_SOURCE_DIR}/modules/roles)
set_tests_properties(terminal_path PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 480)

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
add_dependencies(aotx_screens_tui_test aotx_models)

# The check of the read of the mirror drives a thread that publishes frames beside it.
find_package(Threads REQUIRED)
target_link_libraries(aotx_raster_tui_test PRIVATE Threads::Threads)

add_test(NAME disk_keys COMMAND aotx_keys_tui_test)
add_test(NAME disk_raster_tui COMMAND aotx_raster_tui_test)
add_test(NAME disk_splash COMMAND aotx_splash_tui_test ${CMAKE_SOURCE_DIR}/share/splash)
add_test(NAME disk_screens COMMAND ${CMAKE_SOURCE_DIR}/tests/model_test.sh
         ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_screens_tui_test>)
set_tests_properties(disk_screens PROPERTIES SKIP_RETURN_CODE 77)

# The model file tests read the model files when they are there, and report a skip when
# they are not. The directory of the model files is given to each test.

foreach(case sha256 gguf manifest modelfile)
    add_executable(aotx_${case}_test ../tests/${case}_test.c)
    target_link_libraries(aotx_${case}_test PRIVATE aotx_modelfile)
    target_compile_options(aotx_${case}_test PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()

add_test(NAME disk_sha256 COMMAND ${CMAKE_SOURCE_DIR}/tests/model_test.sh
         ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_sha256_test> ${AOTX_MODELS_DIR})
add_test(NAME disk_gguf COMMAND aotx_gguf_test ${AOTX_MODELS_DIR})
add_test(NAME disk_manifest COMMAND ${CMAKE_SOURCE_DIR}/tests/model_test.sh
         ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_manifest_test>
         $<TARGET_FILE:aotx_manifest> ${AOTX_MODELS_DIR})
add_test(NAME disk_modelfile COMMAND ${CMAKE_SOURCE_DIR}/tests/model_test.sh
         ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_modelfile_test> ${AOTX_MODELS_DIR})
set_tests_properties(disk_sha256 disk_manifest disk_modelfile
                     PROPERTIES SKIP_RETURN_CODE 77)

# The model store checks link the store library. The fetch check runs only loopback
# servers and reports a skip when this build has no fetch operation.
foreach(case catalog store fetch models)
    add_executable(aotx_${case}_test ../tests/${case}_test.c)
    target_link_libraries(aotx_${case}_test PRIVATE aotx_models_store)
    target_compile_options(aotx_${case}_test PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()
if(AOTX_FETCH)
    target_compile_definitions(aotx_fetch_test PRIVATE AOTX_FETCH_TEST)
    target_compile_definitions(aotx_models_test PRIVATE AOTX_FETCH_TEST)
endif()
