project(training)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/src   
)

# denoise training
add_compile_definitions(TRAINING=1)
add_executable(denoise_training
    ${CMAKE_CURRENT_SOURCE_DIR}/denoise.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kiss_fft.c
    ${CMAKE_CURRENT_SOURCE_DIR}/pitch.c
    ${CMAKE_CURRENT_SOURCE_DIR}/celt_lpc.c 
    ${CMAKE_CURRENT_SOURCE_DIR}/rnn.c 
    ${CMAKE_CURRENT_SOURCE_DIR}/rnn_data.c
)


