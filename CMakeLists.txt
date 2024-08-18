cmake_minimum_required(VERSION 3.20)
project(CUDAMatmulPractice LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(${PROJECT_NAME}
        Matmul_kernels.cu
        )

set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} cublas curand)