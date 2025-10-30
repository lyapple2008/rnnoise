cmake_minimum_required(VERSION 3.12)
project(RNNoiseONNX C)

# Set C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find ONNX Runtime
find_package(onnxruntime REQUIRED)

# Alternative: Use pkg-config if find_package fails
if(NOT onnxruntime_FOUND)
    pkg_check_modules(ONNXRUNTIME REQUIRED onnxruntime)
    include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
    link_directories(${ONNXRUNTIME_LIBRARY_DIRS})
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARIES})
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Add ONNX Runtime include directory
if(onnxruntime_FOUND)
    include_directories(${onnxruntime_INCLUDE_DIRS})
else()
    # Fallback for manual installation
    include_directories(/opt/homebrew/Cellar/onnxruntime/1.22.2_3/include/onnxruntime)
endif()

# Add RNNoise source files
set(RNNOISE_SOURCES
    ../src/denoise.c
    ../src/rnn.c
    ../src/pitch.c
    ../src/celt_lpc.c
    ../src/kiss_fft.c
    ../src/rnnoise_weights.c
    ../src/rnn_reader.c
    ../src/rnn_data_final.c
)

# Add RNNoise header files
set(RNNOISE_HEADERS
    ../include/rnnoise.h
    ../src/rnn.h
    ../src/pitch.h
    ../src/celt_lpc.h
    ../src/kiss_fft.h
    ../src/rnn_data.h
    ../src/opus_types.h
    ../src/common.h
    ../src/arch.h
    ../src/_kiss_fft_guts.h
    ../src/tansig_table.h
)

# Create RNNoise library
add_library(rnnoise STATIC ${RNNOISE_SOURCES} ${RNNOISE_HEADERS})

# Set library properties
set_target_properties(rnnoise PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)

# Compiler flags
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    target_compile_options(rnnoise PRIVATE -Wall -Wextra -O3)
endif()

# Create the main executable (ONNX version)
add_executable(rnnoise_onnx rnnoise_onnx.c)

# Link libraries (ONNX version needs ONNX Runtime)
if(onnxruntime_FOUND)
    target_link_libraries(rnnoise_onnx 
        rnnoise
        onnxruntime::onnxruntime
    )
else()
    target_link_libraries(rnnoise_onnx 
        rnnoise
        ${ONNXRUNTIME_LIBRARIES}
    )
endif()

# Set executable properties
set_target_properties(rnnoise_onnx PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
)

# Compiler flags for executable
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    target_compile_options(rnnoise_onnx PRIVATE -Wall -Wextra -O3)
endif()

# Installation
install(TARGETS rnnoise_onnx DESTINATION bin)
install(TARGETS rnnoise DESTINATION lib)

# Print configuration info
message(STATUS "RNNoise ONNX Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  ONNX Runtime found: ${onnxruntime_FOUND}")
message(STATUS "  ONNX Runtime version: ${onnxruntime_VERSION}")

# Optional: Create a simple test
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    
    # Test if the executable can be created
    add_test(NAME rnnoise_onnx_help
        COMMAND rnnoise_onnx
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    
    # Set test properties
    set_tests_properties(rnnoise_onnx_help PROPERTIES
        WILL_FAIL TRUE  # Expected to fail with usage message
    )
endif()

# Print usage instructions
message(STATUS "")
message(STATUS "Build instructions:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make")
message(STATUS "")
message(STATUS "Usage:")
message(STATUS "  ./rnnoise_onnx input.wav output.wav")
message(STATUS "")