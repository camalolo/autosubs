cmake_minimum_required(VERSION 3.20)
project(autosubs C CXX)

set(CMAKE_C_STANDARD 11)

# ---- whisper.cpp (CUDA accelerated) ----
set(GGML_CUDA              ON  CACHE BOOL "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)
set(BUILD_SHARED_LIBS      ON  CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES ON  CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/whisper.cpp)

# ---- autosubs ----
if(NOT AS_VERSION)
    set(AS_VERSION "1.0.0-dev")
    find_package(Git QUIET)
    if(Git_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE AS_GIT_VER
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE AS_GIT_RC)
        if(AS_GIT_RC EQUAL 0 AND AS_GIT_VER)
            set(AS_VERSION ${AS_GIT_VER})
        endif()
    endif()
endif()
message(STATUS "autosubs version: ${AS_VERSION}")

add_executable(autosubs src/autosubs.c)
target_include_directories(autosubs PRIVATE third_party/whisper.cpp/include)
target_link_libraries(autosubs PRIVATE whisper)
target_compile_definitions(autosubs PRIVATE AS_VERSION="${AS_VERSION}")

if(MSVC)
    target_compile_options(autosubs PRIVATE /W3 /O2)
else()
    target_compile_options(autosubs PRIVATE -O3 -Wall)
endif()
