# Project definition
cmake_minimum_required(VERSION 4.0)
project(pmpp-project LANGUAGES CXX CUDA)

# Source files
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(INCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SOURCES 
	"${SRC_DIR}/main.cu" 
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# Executable definition and properties
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE 
	"${SRC_DIR}"
	"${INCL_DIR}"
	"${LIB_DIR}"
	"${LIB_DIR}/glm"
	${CUDAToolkit_INCLUDE_DIRS}
)

find_package(CUDAToolkit)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)