# Native sound preparation and the trained audio encoder execute on CUDA.
add_library(aotx_audio STATIC header.cu state.cu coefficients.cu signal.cu spectral.cu
    product.cu element.cu norm.cu attention.cu audio_host.cu)
target_include_directories(aotx_audio PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/cuda)
target_link_libraries(aotx_audio PUBLIC CUDA::cufft)
