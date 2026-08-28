# The disk-side programs and library. C only; no CUDA dependency.

add_library(aotx_disk STATIC
    wire/clock.c
    wire/crc32c.c
    wire/map.c
    wire/record.c
    wire/ring.c
    wire/segment.c)
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

add_executable(aotx_drain drain/drain.c drain/derive.c)
add_executable(aotx_feed feed/feed.c)
add_executable(aotx_restore restore/restore.c restore/scan.c)

foreach(program aotx_drain aotx_feed aotx_restore)
    target_link_libraries(${program} PRIVATE aotx_disk)
    target_compile_options(${program} PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()

# A bus line validator, when one is given, examines the file that the drain derives.
set(AOTX_BUS_LINT "" CACHE STRING "Program that validates a bus line file.")

foreach(case crc32c segment hostring inbound drain feed restore)
    add_executable(aotx_${case}_test ../tests/${case}_test.c)
    target_link_libraries(aotx_${case}_test PRIVATE aotx_disk)
    target_compile_options(aotx_${case}_test PRIVATE ${AOTX_DISK_WARNINGS})
endforeach()

add_test(NAME disk_crc32c COMMAND aotx_crc32c_test)
add_test(NAME disk_segment COMMAND aotx_segment_test)
add_test(NAME disk_hostring COMMAND aotx_hostring_test)
add_test(NAME disk_inbound COMMAND aotx_inbound_test)
add_test(NAME disk_drain COMMAND aotx_drain_test $<TARGET_FILE:aotx_drain> ${AOTX_BUS_LINT})
add_test(NAME disk_feed COMMAND aotx_feed_test $<TARGET_FILE:aotx_feed>)
add_test(NAME disk_restore COMMAND aotx_restore_test $<TARGET_FILE:aotx_restore>)
