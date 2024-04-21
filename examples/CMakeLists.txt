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
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/rnn_data.c
)

# demo
add_executable(rnnoise_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_demo.c
)
target_link_libraries(rnnoise_demo rnnoise)