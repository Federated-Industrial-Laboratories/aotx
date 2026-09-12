# Native image encoder. Image arithmetic and job state remain on the device.
add_library(aotx_vision STATIC state.cu resize.cu product.cu element.cu norm.cu
                              attention.cu vision_host.cu)
target_include_directories(aotx_vision PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/cuda)
