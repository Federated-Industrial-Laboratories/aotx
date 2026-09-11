add_executable(aotx_context_cache_test context_cache_test.cu)
add_executable(aotx_runtime_state_test runtime_state_test.cu)
target_link_libraries(aotx_runtime_state_test PRIVATE aotx_core aotx_cognitive_disk aotx_modelfile)
set_target_properties(aotx_runtime_state_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME runtime_state COMMAND aotx_runtime_state_test)
add_executable(aotx_runtime_wait_test runtime_wait_test.cu)
target_link_libraries(aotx_runtime_wait_test PRIVATE aotx_core aotx_cognitive_disk aotx_modelfile)
target_link_options(aotx_runtime_wait_test PRIVATE -Wl,--wrap=clock_gettime
    -Wl,--wrap=_Z14aotx_pump_tickP9aotx_pump -Wl,--wrap=_Z14aotx_pump_paceP9aotx_pump
    -Wl,--wrap=_Z14aotx_seam_polliPiS_)
set_target_properties(aotx_runtime_wait_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME runtime_wait COMMAND aotx_runtime_wait_test)
set_tests_properties(runtime_wait PROPERTIES TIMEOUT 30)
target_link_libraries(aotx_context_cache_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_context_cache_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME context_cache COMMAND aotx_context_cache_test)
set_tests_properties(context_cache PROPERTIES TIMEOUT 240)

add_executable(aotx_context_live_test context_live_test.cu)
target_link_libraries(aotx_context_live_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_context_live_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME context_live COMMAND aotx_context_live_test)
set_tests_properties(context_live PROPERTIES TIMEOUT 240)

add_executable(aotx_maintenance_test maintenance_test.cu)
target_link_libraries(aotx_maintenance_test PRIVATE aotx_core aotx_cognitive_disk aotx_modelfile)
set_target_properties(aotx_maintenance_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME maintenance COMMAND aotx_maintenance_test)
set_tests_properties(maintenance PROPERTIES TIMEOUT 600)

add_executable(aotx_maintenance_guards_test maintenance_guards_test.cu)
target_link_libraries(aotx_maintenance_guards_test PRIVATE aotx_core aotx_cognitive_disk aotx_modelfile)
set_target_properties(aotx_maintenance_guards_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME maintenance_guards COMMAND aotx_maintenance_guards_test)
set_tests_properties(maintenance_guards PROPERTIES TIMEOUT 240)

add_executable(aotx_checkpoint_test checkpoint_test.cu checkpoint_fault.c)
add_executable(aotx_runtime_offline runtime_offline.c)
target_compile_options(aotx_runtime_offline PRIVATE -Wall -Wextra -Werror)
target_link_libraries(aotx_checkpoint_test PRIVATE aotx_core aotx_cognitive_disk aotx_modelfile)
target_link_options(aotx_checkpoint_test PRIVATE -Wl,--wrap=pwrite -Wl,--wrap=fsync)
set_target_properties(aotx_checkpoint_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME checkpoint COMMAND aotx_checkpoint_test)
set_tests_properties(checkpoint PROPERTIES TIMEOUT 240)

add_executable(aotx_checkpoint_replay_test checkpoint_replay_test.cu)
target_link_libraries(aotx_checkpoint_replay_test PRIVATE aotx_core aotx_cognitive_disk aotx_modelfile)
set_target_properties(aotx_checkpoint_replay_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME checkpoint_replay COMMAND aotx_checkpoint_replay_test)
set_tests_properties(checkpoint_replay PROPERTIES TIMEOUT 240)

add_executable(aotx_context_memory_test context_memory_test.cu)
target_link_libraries(aotx_context_memory_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_context_memory_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME context_memory COMMAND aotx_context_memory_test)
set_tests_properties(context_memory PROPERTIES TIMEOUT 240)

add_executable(aotx_auto_memory_test auto_memory_test.cu)
target_link_libraries(aotx_auto_memory_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_auto_memory_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME auto_memory COMMAND aotx_auto_memory_test)
set_tests_properties(auto_memory PROPERTIES TIMEOUT 240)

add_test(NAME memory_config COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/memory_config_test.py
    ${CMAKE_SOURCE_DIR} ${CMAKE_COMMAND} ${CMAKE_CUDA_COMPILER})
set_tests_properties(memory_config PROPERTIES TIMEOUT 120)

add_executable(aotx_memory_capacity_test memory_capacity_test.cu)
target_link_libraries(aotx_memory_capacity_test PRIVATE aotx_cognitive)
target_compile_definitions(aotx_memory_capacity_test PRIVATE AOTX_PROFILE_HEADER="profile/${AOTX_PROFILE}.cuh")
set_target_properties(aotx_memory_capacity_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME memory_capacity COMMAND aotx_memory_capacity_test)
set_tests_properties(memory_capacity PROPERTIES TIMEOUT 180)

add_executable(aotx_retain_memory_test retain_memory_test.cu)
target_link_libraries(aotx_retain_memory_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_retain_memory_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME retain_memory COMMAND aotx_retain_memory_test)
set_tests_properties(retain_memory PROPERTIES TIMEOUT 180)

add_executable(aotx_text_memory_test text_memory_test.cu)
target_link_libraries(aotx_text_memory_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_text_memory_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME text_memory COMMAND aotx_text_memory_test)
set_tests_properties(text_memory PROPERTIES TIMEOUT 180)
add_executable(aotx_text_paging_test text_paging_test.cu)
target_link_libraries(aotx_text_paging_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_text_paging_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME text_paging COMMAND aotx_text_paging_test)
set_tests_properties(text_paging PROPERTIES TIMEOUT 180)

add_executable(aotx_live_memory_test live_memory_test.cu)
target_link_libraries(aotx_live_memory_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_live_memory_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME live_memory COMMAND aotx_live_memory_test)
set_tests_properties(live_memory PROPERTIES TIMEOUT 180)
# The model directory is an input of each check that reads model files.
set(AOTX_MODELS_DIR ${CMAKE_SOURCE_DIR}/models CACHE STRING "Directory that holds the model files.")
add_executable(aotx_recall_test recall_test.cu)
target_link_libraries(aotx_recall_test PRIVATE aotx_cognitive)
set_target_properties(aotx_recall_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME recall COMMAND aotx_recall_test)
set_tests_properties(recall PROPERTIES TIMEOUT 120)
add_executable(aotx_recall_cli_fixture recall_cli_fixture.c)
target_link_libraries(aotx_recall_cli_fixture PRIVATE aotx_ccir_store)
target_include_directories(aotx_recall_cli_fixture PRIVATE ${CMAKE_SOURCE_DIR}/cuda ${CMAKE_SOURCE_DIR}/disk)
add_test(NAME recall_cli COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/recall_cli_test.py
    $<TARGET_FILE:aotx_ccir_recall> $<TARGET_FILE:aotx_recall_cli_fixture>)
set_tests_properties(recall_cli PROPERTIES TIMEOUT 180)
add_executable(aotx_cognitive_test cognitive_test.cu)
target_link_libraries(aotx_cognitive_test PRIVATE aotx_cognitive)
set_target_properties(aotx_cognitive_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME cognitive COMMAND aotx_cognitive_test)
add_executable(aotx_cognitive_file_test cognitive_file_test.cu)
target_link_libraries(aotx_cognitive_file_test PRIVATE aotx_cognitive aotx_cognitive_disk)
set_target_properties(aotx_cognitive_file_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME cognitive_file COMMAND aotx_cognitive_file_test $<TARGET_FILE:aotx_ccir_state>)
set_tests_properties(cognitive cognitive_file PROPERTIES TIMEOUT 120)
add_test(NAME replay_config COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/replay_config_test.py
         ${CMAKE_CURRENT_SOURCE_DIR}/replay_affect.sh)
add_test(NAME replay_status COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/replay_status_test.py
         ${CMAKE_CURRENT_SOURCE_DIR}/replay_test.sh)
set_tests_properties(replay_status PROPERTIES TIMEOUT 60)

add_test(NAME size_gate_boundary COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/size_gate_test.py
         ${CMAKE_SOURCE_DIR}/tools/size-gate.py)
add_test(NAME arch_process_status COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arch_process_test.py)
add_executable(aotx_fault_status_test fault_status_test.c)
add_test(NAME fault_status COMMAND aotx_fault_status_test)

# The test programs. A test is a host program that launches kernels. A device test target
# carries a device mark, because the disk side has a test of the same subject.
foreach(name rng mem seam bus bulk kvcache sched model sample settings load expert bias delta gated vocab_select)
    add_executable(aotx_${name}_device_test ${name}_test.cu)
    target_link_libraries(aotx_${name}_device_test PRIVATE aotx_core Threads::Threads)
    set_target_properties(aotx_${name}_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    if(name STREQUAL "load")
        add_test(NAME ${name} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh
                 ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_${name}_device_test>
                 ${AOTX_MODELS_DIR})
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
    else()
        add_test(NAME ${name} COMMAND aotx_${name}_device_test)
    endif()
endforeach()

# The profile check runs the rule that states the memory a profile needs. It opens no
# context and needs no card, so it runs on every build of every profile.
add_executable(aotx_profile_test profile_test.cu)
target_link_libraries(aotx_profile_test PRIVATE aotx_core)
set_target_properties(aotx_profile_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME profile COMMAND aotx_profile_test)

add_executable(aotx_hybrid_file_test hybrid_file_test.cu)
target_link_libraries(aotx_hybrid_file_test PRIVATE aotx_core)
set_target_properties(aotx_hybrid_file_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME hybrid_file COMMAND aotx_hybrid_file_test)

# The tokenizer check reads the fixture lists and the tokenizer of a model file. The model
# files are large and are not in the repository, so the check reports a skip for a file
# which is not there. The directory is the one the disk side checks read.
add_executable(aotx_text_device_test text_test.cu)
target_link_libraries(aotx_text_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_text_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME text COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_text_device_test>
         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer ${AOTX_MODELS_DIR})
set_tests_properties(text PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)

# The matrix check reads the model files for the real weight rows. A file which is not
# there gives a skip that the check reports as a figure.
add_executable(aotx_matrix_device_test matrix_test.cu)
target_link_libraries(aotx_matrix_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_matrix_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME matrix COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_matrix_device_test> ${AOTX_MODELS_DIR})
set_tests_properties(matrix PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
add_test(NAME matrix_blocks COMMAND aotx_matrix_device_test --blocks-only)
set_tests_properties(matrix_blocks PROPERTIES TIMEOUT 600)

# The placement check reads a store and places every file. It reports the block type tally
# of each file and the placed count against the tensor count. The store is the default one
# unless the check is run by hand with another directory.
add_executable(aotx_place_device_test place_test.cu)
target_link_libraries(aotx_place_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_place_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME place COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_place_device_test> ${AOTX_MODELS_DIR})
set_tests_properties(place PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)

# The architecture check loads each language file of a store and prints the checks of one
# architecture: bind, wrap, shapes, state, and cache rebuild. It compares greedy token lists
# and distinct batches. Optional process-check arguments add restore and console checks.
add_executable(aotx_arch_device_test arch_test.cu)
target_link_libraries(aotx_arch_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_arch_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(aotx_arch_logits arch_logits.cu)
target_link_libraries(aotx_arch_logits PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_arch_logits PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME arch_logits_input COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arch_logits_input_test.py
         $<TARGET_FILE:aotx_arch_logits>)
add_test(NAME arch_accuracy_metrics COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arch_accuracy_test.py)
add_test(NAME arch_accuracy_inputs COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arch_accuracy_io_test.py)
add_test(NAME arch_accuracy_reference COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arch_accuracy_reference_test.py)
add_test(NAME arch_accuracy_bundle COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arch_accuracy_bundle_test.py)
add_test(NAME arch_reference_inputs COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arch_reference_test.py)

add_executable(aotx_wrap_prompt_test wrap_prompt_test.cu)
target_include_directories(aotx_wrap_prompt_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(aotx_wrap_prompt_test PRIVATE aotx_core)
set_target_properties(aotx_wrap_prompt_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME wrap_prompt COMMAND aotx_wrap_prompt_test)

add_executable(aotx_call_prompt_test call_prompt_test.cu)
target_include_directories(aotx_call_prompt_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(aotx_call_prompt_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_call_prompt_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME call_prompt COMMAND aotx_call_prompt_test)

add_executable(aotx_call_schema_test call_schema_test.cu)
target_include_directories(aotx_call_schema_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(aotx_call_schema_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_call_schema_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME call_schema COMMAND aotx_call_schema_test
         ${CMAKE_SOURCE_DIR}/modules/roles/conductor/module.manifest)

add_executable(aotx_result_bounds_test result_bounds_test.cu)
target_include_directories(aotx_result_bounds_test PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(aotx_result_bounds_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_result_bounds_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME result_bounds COMMAND aotx_result_bounds_test)

add_executable(aotx_wrap_load_test wrap_load_test.cu)
target_link_libraries(aotx_wrap_load_test PRIVATE aotx_core)
set_target_properties(aotx_wrap_load_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME wrap_load COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh
         ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_wrap_load_test> ${AOTX_MODELS_DIR} language-q4)
set_tests_properties(wrap_load PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)

# The model gate reads the reference lists and the model files. The model files are large
# and are not in the repository, so the gate reports a skip for a directory which is empty.
add_executable(aotx_model_gate_device_test model_gate_test.cu)
target_link_libraries(aotx_model_gate_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_model_gate_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME model_gate COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh
         ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_model_gate_device_test>
         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model ${AOTX_MODELS_DIR})
set_tests_properties(model_gate PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)

# The decode check runs the tick graph with a language model. The model files are large and
# are not in the repository, so the check reports a skip for a directory which is empty.
add_executable(aotx_decode_device_test decode_test.cu)
target_link_libraries(aotx_decode_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_decode_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME decode COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_decode_device_test> ${AOTX_MODELS_DIR}
         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer)
set_tests_properties(decode PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)

# The tool check runs the parser with no model and the two memory tools with the embedding
# model. The model files are large and are not in the repository, so the check reports a
# skip for a directory which is empty.
add_executable(aotx_tool_device_test tool_test.cu)
target_link_libraries(aotx_tool_device_test PRIVATE aotx_core aotx_modelfile Threads::Threads)
set_target_properties(aotx_tool_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME tool COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_tool_device_test> ${AOTX_MODELS_DIR})
set_tests_properties(tool PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)

# The agent check runs the turn loop on the real model. It reads the model directory and
# the module directories, because the roles of a run come in as modules.
add_executable(aotx_agent_device_test agent_test.cu)
target_link_libraries(aotx_agent_device_test PRIVATE aotx_core aotx_modelfile Threads::Threads)
set_target_properties(aotx_agent_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME agent COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_agent_device_test> ${AOTX_MODELS_DIR}
         ${CMAKE_SOURCE_DIR}/modules/roles)
set_tests_properties(agent PROPERTIES TIMEOUT 2400 SKIP_RETURN_CODE 77)

# The affect check drives the probe loader on a fixture store. It drives the read branch
# of the conduct kernel on a synthetic batch and the turn node on scripted agents. It
# needs no model file, and it exists only in a build with the affect substrate.
if(AOTX_AFFECT)
    add_executable(aotx_affect_device_test affect_test.cu)
    target_link_libraries(aotx_affect_device_test PRIVATE aotx_core Threads::Threads)
    set_target_properties(aotx_affect_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_test(NAME affect COMMAND aotx_affect_device_test)
    set_tests_properties(affect PROPERTIES TIMEOUT 300)

    add_executable(aotx_quality_device_test quality_test.cu)
    target_link_libraries(aotx_quality_device_test PRIVATE aotx_core Threads::Threads)
    set_target_properties(aotx_quality_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_test(NAME quality COMMAND aotx_quality_device_test
             ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/quality/refusal-phrases.txt
             ${AOTX_MODELS_DIR})
    set_tests_properties(quality PROPERTIES TIMEOUT 300)
endif()

# The axis check runs the axis mode and the calibrate mode of the steer tool on a private
# copy of the store. It boots the build on that copy to load the probe rows. It reads the
# model files, so it reports a skip for a directory without a model record.
if(AOTX_AFFECT)
    add_test(NAME affect_axis
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
                     ${CMAKE_CURRENT_SOURCE_DIR}/affect_axis.sh ${CMAKE_BINARY_DIR}
                     ${AOTX_MODELS_DIR})
    set_tests_properties(affect_axis PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)
endif()

# The score check runs the task mode of the quality score tool on a private copy of the
# store. The check writes a ten-item task set. It reads the model files, so it reports a
# skip for a directory without a model record.
add_test(NAME quality_score
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
                 ${CMAKE_CURRENT_SOURCE_DIR}/quality_score.sh ${CMAKE_BINARY_DIR}
                 ${AOTX_MODELS_DIR})
set_tests_properties(quality_score PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)

# The conversation check drives line parts through the apply and checks the tier state.
add_executable(aotx_conversation_device_test conversation_test.cu)
target_link_libraries(aotx_conversation_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_conversation_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME conversation COMMAND aotx_conversation_device_test)
set_tests_properties(conversation PROPERTIES TIMEOUT 600)

# The catalog check drives the import path over the inbound ring and the apply node. It
# needs no model file, so it runs on every build.
add_executable(aotx_catalog_device_test catalog_test.cu)
target_link_libraries(aotx_catalog_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_catalog_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME catalog COMMAND aotx_catalog_device_test ${CMAKE_SOURCE_DIR}/modules)
set_tests_properties(catalog PROPERTIES TIMEOUT 900)

# The module check drives the device tool modules. It takes the import, the node of the
# tick graph and the capture that follows an import. It takes the batch at 1 row and at the
# row count, and the refusal of a module file that changed on disk. It reads the module
# directory the setup built, so it needs no model file.
add_executable(aotx_module_device_test module_test.cu)
target_link_libraries(aotx_module_device_test PRIVATE aotx_core aotx_modelfile Threads::Threads)
set_target_properties(aotx_module_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# The example modules and the fixtures of the check program go into the build directory
# and are built there. A check therefore leaves the source tree as it was.
add_test(NAME module_setup
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_setup.sh ${CMAKE_SOURCE_DIR}
                 ${CMAKE_BINARY_DIR} ${AOTX_ARCH})
set_tests_properties(module_setup PROPERTIES FIXTURES_SETUP module_files TIMEOUT 600)

add_test(NAME module COMMAND aotx_module_device_test ${CMAKE_BINARY_DIR}/modules/word_count
         ${CMAKE_BINARY_DIR})
set_tests_properties(module PROPERTIES FIXTURES_REQUIRED module_files TIMEOUT 900)

# The check program over the two example modules. With no row count the check program runs
# the module at one row and at the row count of the profile.
add_test(NAME example_word_count
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_check.sh
                 $<TARGET_FILE:aotx_module_check>
                 ${CMAKE_BINARY_DIR}/modules/word_count pass)
set_tests_properties(example_word_count PROPERTIES FIXTURES_REQUIRED module_files
                     TIMEOUT 300)

add_test(NAME example_echo_upper
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_check.sh
                 $<TARGET_FILE:aotx_module_check>
                 ${CMAKE_BINARY_DIR}/modules/echo_upper pass)
set_tests_properties(example_echo_upper PROPERTIES FIXTURES_REQUIRED module_files
                     TIMEOUT 300)

# Each refusal of the check program is proven by a fixture module that earns it.
foreach(case local_memory wrong_entry over_bound untaken_row unknown_key hang_tool
             bad_digest)
    add_test(NAME module_refuse_${case}
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_check.sh
                     $<TARGET_FILE:aotx_module_check>
                     ${CMAKE_BINARY_DIR}/modules/${case} fail)
    set_tests_properties(module_refuse_${case} PROPERTIES FIXTURES_REQUIRED module_files
                         TIMEOUT 300)
endforeach()

# The program of hang_tool never ends by itself. The check program must end it at the
# timeout of its manifest, which is one second. A short allowance here therefore fails a
# check program that waits for such a program instead of ending it.
set_tests_properties(module_refuse_hang_tool PROPERTIES TIMEOUT 60)

# The seam gate of the build script is proven able to fail: a module that holds a host call
# does not become a module file.
add_test(NAME module_build_host_call
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_build.sh ${CMAKE_SOURCE_DIR}
                 ${CMAKE_BINARY_DIR}/modules/host_call fail ${AOTX_ARCH})
set_tests_properties(module_build_host_call PROPERTIES FIXTURES_REQUIRED module_files
                     TIMEOUT 300)

# The spill gate reads the resource usage the device linker wrote into the built objects.
# It refuses local memory or a stack frame over the allowance in the hot kernels. The
# figures come from cuobjdump, which stands beside the compiler of this build.
get_filename_component(AOTX_CUDA_BIN ${CMAKE_CUDA_COMPILER} DIRECTORY)
add_test(NAME spill
         COMMAND ${CMAKE_SOURCE_DIR}/tools/spill-gate.py
                 --cuobjdump ${AOTX_CUDA_BIN}/cuobjdump ${CMAKE_BINARY_DIR})

# The graph check and the tick check take longer than a unit test, because the tick check
# measures 500 ticks.
set_tests_properties(sched PROPERTIES TIMEOUT 300)

# The replay check runs the system, kills it, restores it, and compares the state hash. The
# second scenario asks the language model for a reply. It reads the model directory and
# loads the weights twice, so the time allowance holds both loads.
add_test(NAME replay
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
                 ${CMAKE_CURRENT_SOURCE_DIR}/replay_test.sh
                 ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/replay-journal ${AOTX_MODELS_DIR})
# Allow file verification for all scenarios, including both affect batch sizes.
set_tests_properties(replay PROPERTIES TIMEOUT 6000 FIXTURES_REQUIRED module_files
                     SKIP_RETURN_CODE 77 ENVIRONMENT "AOTX_REPLAY_SCENARIO=")

# The sanitizer gate runs compute-sanitizer over the checks that drive the tick graph. A
# run takes minutes, because the sanitizer makes every kernel far slower. The gate is
# registered only when a build asks for it: cmake -DAOTX_SANITIZER_TESTS=ON.
option(AOTX_SANITIZER_TESTS "Register the compute-sanitizer gate" OFF)
if(AOTX_SANITIZER_TESTS)
    foreach(tool memcheck racecheck)
        add_test(NAME sanitizer_${tool}
                 COMMAND ${CMAKE_SOURCE_DIR}/tools/sanitizer-gate.sh ${tool}
                         ${CMAKE_BINARY_DIR} ${AOTX_MODELS_DIR}
                         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer)
        set_tests_properties(sanitizer_${tool} PROPERTIES TIMEOUT 7200 RUN_SERIAL TRUE
                             LABELS sanitizer
                             ENVIRONMENT AOTX_SANITIZER_BIN=${AOTX_CUDA_BIN}/compute-sanitizer)
    endforeach()
endif()

# The fault cases write past a guard gap on purpose. The fault kills a context on the display
# device, so they run only when this option is on: cmake -DAOTX_FAULT_TESTS=ON.
option(AOTX_FAULT_TESTS "Run the tests that fault the device on purpose" OFF)
if(AOTX_FAULT_TESTS)
    foreach(name mem kvcache)
        add_test(NAME ${name}_fault COMMAND aotx_${name}_device_test)
        set_tests_properties(${name}_fault PROPERTIES ENVIRONMENT AOTX_FAULT_TESTS=1 RUN_SERIAL TRUE)
    endforeach()
endif()

# The disk model-store targets are declared by disk/CMakeLists.txt after this directory.
add_test(NAME disk_catalog COMMAND $<TARGET_FILE:aotx_catalog_test>
         ${CMAKE_SOURCE_DIR}/share/models/catalog.jsonl)
add_test(NAME disk_store COMMAND $<TARGET_FILE:aotx_store_test>)
add_test(NAME disk_fetch COMMAND $<TARGET_FILE:aotx_fetch_test>)
add_test(NAME disk_models COMMAND $<TARGET_FILE:aotx_models_test> $<TARGET_FILE:aotx_models>)
set_tests_properties(disk_fetch PROPERTIES TIMEOUT 120)
add_test(NAME disk_feed_models COMMAND $<TARGET_FILE:aotx_feed_models_test>
         $<TARGET_FILE:aotx_feed>)
set_tests_properties(disk_feed_models PROPERTIES TIMEOUT 30)

foreach(name request_completion tool_policy prompt_policy)
    add_executable(aotx_${name}_test ${name}_test.cu)
    target_link_libraries(aotx_${name}_test PRIVATE aotx_core)
    set_target_properties(aotx_${name}_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_test(NAME ${name} COMMAND aotx_${name}_test)
    set_tests_properties(${name} PROPERTIES TIMEOUT 60)
endforeach()

add_executable(aotx_live_feed_test live_feed_test.c ../disk/feed/cognitive_io.c ../disk/feed/line.c)
target_include_directories(aotx_live_feed_test PRIVATE ${CMAKE_SOURCE_DIR}/cuda)
target_link_libraries(aotx_live_feed_test PRIVATE aotx_cognitive_disk aotx_disk Threads::Threads)
target_compile_options(aotx_live_feed_test PRIVATE -Wall -Wextra -Werror)
target_link_options(aotx_live_feed_test PRIVATE -Wl,--wrap=read)
add_test(NAME live_feed COMMAND aotx_live_feed_test)
set_tests_properties(live_feed PROPERTIES TIMEOUT 60)
add_executable(aotx_live_transcript_test live_transcript_test.c
    ../disk/drain/transcript.c ../disk/drain/transcript_live.c)
target_include_directories(aotx_live_transcript_test PRIVATE ${CMAKE_SOURCE_DIR}/cuda)
target_link_libraries(aotx_live_transcript_test PRIVATE aotx_disk)
target_compile_options(aotx_live_transcript_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME live_transcript COMMAND aotx_live_transcript_test)
set_tests_properties(live_transcript PROPERTIES TIMEOUT 60)

add_executable(aotx_intake_memory_test intake_memory_test.cu)
target_link_libraries(aotx_intake_memory_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_intake_memory_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME intake_memory COMMAND aotx_intake_memory_test)
set_tests_properties(intake_memory PROPERTIES TIMEOUT 300)

add_executable(aotx_intake_correction_test intake_correction_test.cu)
target_link_libraries(aotx_intake_correction_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_intake_correction_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME intake_correction COMMAND aotx_intake_correction_test)
set_tests_properties(intake_correction PROPERTIES TIMEOUT 300)

add_executable(aotx_intake_span_test intake_span_test.cu)
target_link_libraries(aotx_intake_span_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_intake_span_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME intake_span COMMAND aotx_intake_span_test)
set_tests_properties(intake_span PROPERTIES TIMEOUT 300)

add_executable(aotx_intake_history_test intake_history_test.cu)
target_link_libraries(aotx_intake_history_test PRIVATE aotx_cognitive)
set_target_properties(aotx_intake_history_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME intake_history COMMAND aotx_intake_history_test)
set_tests_properties(intake_history PROPERTIES TIMEOUT 60)

add_executable(aotx_intake_lease_test intake_lease_test.cu)
target_link_libraries(aotx_intake_lease_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_intake_lease_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME intake_lease COMMAND aotx_intake_lease_test)
set_tests_properties(intake_lease PROPERTIES TIMEOUT 300)

add_executable(aotx_intake_grammar_test intake_grammar_test.cu)
target_link_libraries(aotx_intake_grammar_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_intake_grammar_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME intake_grammar COMMAND aotx_intake_grammar_test)
set_tests_properties(intake_grammar PROPERTIES TIMEOUT 300)

add_executable(aotx_intake_capacity_test intake_capacity_test.cu)
target_link_libraries(aotx_intake_capacity_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_intake_capacity_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME intake_capacity COMMAND aotx_intake_capacity_test)
set_tests_properties(intake_capacity PROPERTIES TIMEOUT 300)

add_executable(aotx_intake_paging_test intake_paging_test.cu)
target_link_libraries(aotx_intake_paging_test PRIVATE aotx_core aotx_modelfile)
set_target_properties(aotx_intake_paging_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME intake_paging COMMAND aotx_intake_paging_test)
set_tests_properties(intake_paging PROPERTIES TIMEOUT 300)
