# Configuration:
All configuration options can be found in *main.cu*

- **PRINT_PATH:** Enabled or disabled whether the generated BFS path should be printed to the console and to a noise.ppm file or not. Mainly useful for debugging purposes.
- **ENABLE_OBSTACLES: ** Enables or disabled BFS running with a noisy graph representing obstacles between nodes.
- **activeAlgorithm: ** Can be one of the possible algorithms found in the BFSAlgorithm enum. Configures which implementation of BFS is used in the BFS execution
- **GRID_SIZE: ** Configures the size of the grid. The grid always has an equal height and width.
- **START_POS: ** Configures the starting vertex of path drawn towards the configured TARGET_POS. Can mainly be used for debug purposes in combination with *PRINT_PATH*.
- **TARGET_POS: ** Starting vertex of the BFS calculations  representing layer 0 of the generated levels in the graph.
- **numIterations:** Amount of iterations a configured algorithm is run for. GPU implementations additionally include one warm-up run. 

# Requirements:
- CUDA

Tested on Windows 11 using Visual Studio