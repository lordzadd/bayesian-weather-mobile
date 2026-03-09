cmake_minimum_required(VERSION 3.22.1)
project(bma_executorch VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ExecuTorch AAR is extracted to this path by Gradle
set(EXECUTORCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../executorch")

add_library(bma_executorch SHARED bma_executorch.cpp)

target_include_directories(bma_executorch PRIVATE
    ${EXECUTORCH_ROOT}/include
)

target_link_libraries(bma_executorch
    android
    log
    ${EXECUTORCH_ROOT}/libs/${ANDROID_ABI}/libexecutorch.a
    ${EXECUTORCH_ROOT}/libs/${ANDROID_ABI}/libexecutorch_core.a
    ${EXECUTORCH_ROOT}/libs/${ANDROID_ABI}/libvulkan_backend.a
)

# Enable link-time optimization for release builds
target_compile_options(bma_executorch PRIVATE
    $<$<CONFIG:Release>:-O3 -flto>
)
