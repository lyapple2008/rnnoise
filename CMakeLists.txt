cmake_minimum_required(VERSION 3.10)
project(rnnoise)

# Windows specific fixes
if(WIN32 AND MSVC)
    # Define M_PI and other math constants for MSVC
    add_compile_definitions(_USE_MATH_DEFINES)
    # Enable C99 features in MSVC
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

add_subdirectory(src)
add_subdirectory(examples)


