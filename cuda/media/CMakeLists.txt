# Build the device image decoder without a host codec dependency.
add_library(aotx_image STATIC image.cu image_host.cu jpeg_pixels.cu hash.cu)
target_include_directories(aotx_image PUBLIC ${PROJECT_SOURCE_DIR}/cuda)
