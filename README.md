# Breadth-first Search using dynamic parallelism
Implementation of the breadth-first search (BFS) algorithm in six different configurations using CUDA and C++
- **CPU:** Slow implementation of a GPU kernel on the CPU.
- **CPU_QUEUE:** An improved version of the CPU kernel using *std::queue*.
- **GPU:** Basic implementation of BFS using a GPU kernel 
- **GPU_FRONTIER:** Improvement version of the GPU kernel using frontiers
- **GPU_SHARED:** Improvement version of the GPU_FRONTIER kernel using shared memory
- **GPU_DYNAMIC_PARALLEL:** Implementation of the GPU_SHARED kernel using dynamic parallelism

Created for the module *Advanced Programming of Massively Parallel Processors (143112A)* at Hochschule der Medien, Stuttgart

## Configuration:
All configuration options can be found in *main.cu*

- **PRINT_PATH:** Enabled or disabled whether the generated BFS path should be printed to the console and to a noise.ppm file or not. Mainly useful for debugging purposes.
- **ENABLE_OBSTACLES:** Enables or disabled BFS running with a noisy graph representing obstacles between nodes.
- **activeAlgorithm:** Can be one of the possible algorithms found in the BFSAlgorithm enum. Configures which implementation of BFS is used in the BFS execution.
- **GRID_SIZE:** Configures the size of the grid. The grid always has an equal height and width.
- **START_POS:** Configures the starting vertex of path drawn towards the configured TARGET_POS. Can mainly be used for debug purposes in combination with *PRINT_PATH*.
- **TARGET_POS:** Starting vertex of the BFS calculations  representing layer 0 of the generated levels in the graph.
- **numIterations:** Amount of iterations a configured algorithm is run for. GPU implementations additionally include one warm-up run. 

## Requirements:
- CUDA

## Paper:
The paper for this project can be found in the paper folder

Tested on Windows 11 using Visual Studio