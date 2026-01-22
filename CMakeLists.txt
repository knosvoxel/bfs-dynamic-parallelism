# Project definition
cmake_minimum_required(VERSION 4.0)
project(pmpp-project)

# Source files
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(INCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(SOURCES 
	"${SRC_DIR}/main.cpp" 
)

# Executable definition and properties
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE "${SRC_DIR}")
set(CMAKE_CXX_STANDARD 20)

# include folder
target_include_directories(${PROJECT_NAME} PRIVATE "${INCL_DIR}")
target_include_directories(${PROJECT_NAME} PRIVATE "${LIB_DIR}")
