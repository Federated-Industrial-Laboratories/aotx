# The disk-side programs and library. C only; no CUDA dependency.

add_library(aotx_disk STATIC
    wire/clock.c
    wire/crc32c.c
    wire/jsonread.c
    wire/map.c
    wire/record.c
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

# The model file reader. The host glue links it to load the weights of a model.
add_library(aotx_modelfile STATIC
    modelfile/gguf.c
    modelfile/meta.c
    modelfile/manifest.c)
target_link_libraries(aotx_modelfile PUBLIC aotx_disk)
target_compile_options(aotx_modelfile PRIVATE ${AOTX_DISK_WARNINGS})

add_executable(aotx_drain drain/drain.c drain/derive.c drain/derive_bus.c
                          drain/derive_manifest.c drain/jsontext.c drain/bulk.c)
add_executable(aotx_feed feed/feed.c feed/fs_tool.c)
add_executable(aotx_restore restore/restore.c restore/scan.c)
add_executable(aotx_journal journal/journal.c journal/chain.c restore/scan.c)
add_executable(aotx_manifest manifest/manifest.c)

foreach(program aotx_drain aotx_feed aotx_restore aotx_journal)
    target_link_libraries(${program} PRIVATE aotx_disk)
    target_compile_options(${program} PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()

target_link_libraries(aotx_manifest PRIVATE aotx_modelfile)
target_compile_options(aotx_manifest PRIVATE ${AOTX_DISK_WARNINGS})

# A bus line validator, when one is given, examines the file that the drain derives.
set(AOTX_BUS_LINT "" CACHE STRING "Program that validates a bus line file.")

foreach(case crc32c segment hostring inbound drain derive bulk feed restore journal)
    add_executable(aotx_${case}_test ../tests/${case}_test.c)
    target_link_libraries(aotx_${case}_test PRIVATE aotx_disk)
    target_compile_options(aotx_${case}_test PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()

add_test(NAME disk_crc32c COMMAND aotx_crc32c_test)
add_test(NAME disk_segment COMMAND aotx_segment_test)
add_test(NAME disk_hostring COMMAND aotx_hostring_test)
add_test(NAME disk_inbound COMMAND aotx_inbound_test)
add_test(NAME disk_drain COMMAND aotx_drain_test $<TARGET_FILE:aotx_drain> ${AOTX_BUS_LINT})
add_test(NAME disk_derive COMMAND aotx_derive_test $<TARGET_FILE:aotx_drain> ${AOTX_BUS_LINT})
add_test(NAME disk_bulk COMMAND aotx_bulk_test $<TARGET_FILE:aotx_drain>)
add_test(NAME disk_feed COMMAND aotx_feed_test $<TARGET_FILE:aotx_feed> $<TARGET_FILE:aotx_drain>)
add_test(NAME disk_restore COMMAND aotx_restore_test $<TARGET_FILE:aotx_restore>)
add_test(NAME disk_journal COMMAND aotx_journal_test $<TARGET_FILE:aotx_journal> $<TARGET_FILE:aotx_drain>)

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
