cmake_minimum_required(VERSION 3.10)
project(rnnoise_demo)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src   
)

# rnnoise library
add_library(rnnoise
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/denoise.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/kiss_fft.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/pitch.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/celt_lpc.c 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/rnn.c 
    # ${CMAKE_CURRENT_SOURCE_DIR}/../src/rnn_data.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/rnn_data_epoch_100.c
)

# Find libsamplerate
find_package(PkgConfig REQUIRED)
pkg_check_modules(SAMPLERATE REQUIRED samplerate)

# demo
add_executable(rnnoise_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_demo.c
)
target_link_libraries(rnnoise_demo rnnoise ${SAMPLERATE_LIBRARIES})
target_include_directories(rnnoise_demo PRIVATE ${SAMPLERATE_INCLUDE_DIRS})
target_compile_options(rnnoise_demo PRIVATE ${SAMPLERATE_CFLAGS_OTHER})
target_link_directories(rnnoise_demo PRIVATE ${SAMPLERATE_LIBRARY_DIRS})