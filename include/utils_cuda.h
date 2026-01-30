#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#define GPU_ERRCHK(ans) { gpu_assert((ans), __FILE__, __LINE__); }

inline void gpu_assert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "gpu_assert: " << cudaGetErrorString(code) << file << line << std::endl;
        if (abort)
            exit(code);
    }
}