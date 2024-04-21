project(rnnoise_demo)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src  
    ${CMAKE_CURRENT_SOURCE_DIR} 
)

# rnnoise library
add_library(rnnoise
${CMAKE_CURRENT_SOURCE_DIR}/../src/denoise.c
${CMAKE_CURRENT_SOURCE_DIR}/../src/kiss_fft.c
${CMAKE_CURRENT_SOURCE_DIR}/../src/nnet_default.c
${CMAKE_CURRENT_SOURCE_DIR}/../src/nnet.c
${CMAKE_CURRENT_SOURCE_DIR}/../src/parse_lpcnet_weights.c
${CMAKE_CURRENT_SOURCE_DIR}/../src/pitch.c
${CMAKE_CURRENT_SOURCE_DIR}/../src/celt_lpc.c 
${CMAKE_CURRENT_SOURCE_DIR}/../src/rnn.c 
${CMAKE_CURRENT_SOURCE_DIR}/../src/rnnoise_data.c
${CMAKE_CURRENT_SOURCE_DIR}/../src/rnnoise_tables.c
)

# demo
add_executable(rnnoise_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_demo.c
    ${CMAKE_CURRENT_SOURCE_DIR}/wav_io.c
)
target_link_libraries(rnnoise_demo rnnoise)