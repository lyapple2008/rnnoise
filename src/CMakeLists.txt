project(training)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/src   
)

# todo: support x86 compile

# training: dump_features
add_executable(dump_features
    ${CMAKE_CURRENT_SOURCE_DIR}/dump_features.c
    ${CMAKE_CURRENT_SOURCE_DIR}/denoise.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kiss_fft.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nnet_default.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nnet.c
    ${CMAKE_CURRENT_SOURCE_DIR}/parse_lpcnet_weights.c
    ${CMAKE_CURRENT_SOURCE_DIR}/pitch.c
    ${CMAKE_CURRENT_SOURCE_DIR}/celt_lpc.c 
    ${CMAKE_CURRENT_SOURCE_DIR}/rnn.c 
    ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_data.c
    ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_tables.c
)

# training: dump_rnnoise_tables
add_executable(dump_rnnoise_tables
    ${CMAKE_CURRENT_SOURCE_DIR}/dump_rnnoise_tables.c
    ${CMAKE_CURRENT_SOURCE_DIR}/denoise.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kiss_fft.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nnet_default.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nnet.c
    ${CMAKE_CURRENT_SOURCE_DIR}/parse_lpcnet_weights.c
    ${CMAKE_CURRENT_SOURCE_DIR}/pitch.c
    ${CMAKE_CURRENT_SOURCE_DIR}/celt_lpc.c 
    ${CMAKE_CURRENT_SOURCE_DIR}/rnn.c 
    ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_data.c
    ${CMAKE_CURRENT_SOURCE_DIR}/rnnoise_tables.c
)

