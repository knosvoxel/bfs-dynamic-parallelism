#pragma once

#include <cuda_runtime.h>
#include <iostream>

#include "csr.h"
#include "timer.h"
#include "utils_cuda.h"

__global__
void bfsFrontier(CSRGraph csrGraph, uint32* level, uint32* prevFrontier, uint32* currFrontier, uint32 numPrevFrontier, uint32* numCurrFrontier, uint32 currLevel)
{
	uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numPrevFrontier)
	{
		uint32 vertex = prevFrontier[i];
		for (uint32 edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
		{
			uint32 neighbour = csrGraph.dst[edge];
			if (atomicCAS(&level[neighbour], UINT_MAX, currLevel) == UINT_MAX)
			{
				uint32 currFrontierIdx = atomicAdd(numCurrFrontier, 1);
				currFrontier[currFrontierIdx] = neighbour;
			}
		}
	}
}

void runBFSFrontier(CSRGraph& graphDevice, uint32* levelDevice, uint32* levelHost, uint32 targetNode, uint32 numVertices, uint32& currLevel, Timer& timer)
{
	uint32* currFrontierDevice, * nextFrontierDevice, * nextFrontierCountDevice;

	GPU_ERRCHK(cudaMalloc(&currFrontierDevice, numVertices * sizeof(uint32)));
	GPU_ERRCHK(cudaMalloc(&nextFrontierDevice, numVertices * sizeof(uint32)));
	GPU_ERRCHK(cudaMalloc(&nextFrontierCountDevice, sizeof(uint32)));

	GPU_ERRCHK(cudaMemcpy(currFrontierDevice, &targetNode, sizeof(uint32), cudaMemcpyHostToDevice));

	uint32 numFrontierElements = 1;
	int32 threadsPerBlock = 256;
	dim3 numThreads(threadsPerBlock, 1, 1);

	while (numFrontierElements > 0)
	{
		GPU_ERRCHK(cudaMemset(nextFrontierCountDevice, 0, sizeof(uint32)));

		dim3 currNumThreads;
		dim3 currNumBlocks;

		if (numFrontierElements <= 1024)
		{
			currNumThreads = dim3(numFrontierElements, 1, 1);
			currNumBlocks = dim3(1, 1, 1);
		}
		else {
			currNumThreads = numThreads;
			int32 frontierBlocks = (numFrontierElements + threadsPerBlock - 1) / threadsPerBlock;
			currNumBlocks = dim3(frontierBlocks, 1, 1);

		}

		bfsFrontier << <currNumBlocks, currNumThreads >> > (graphDevice, levelDevice, currFrontierDevice, nextFrontierDevice, numFrontierElements, nextFrontierCountDevice, currLevel);

		GPU_ERRCHK(cudaMemcpy(&numFrontierElements, nextFrontierCountDevice, sizeof(uint32), cudaMemcpyDeviceToHost));

		std::swap(currFrontierDevice, nextFrontierDevice);

		currLevel++;

		if (currLevel > numVertices) break;
	}

	GPU_ERRCHK(cudaDeviceSynchronize());

	std::cout << "Frontier BFS finished after " << currLevel - 1 << " levels." << std::endl;

	std::cout << timer.ToString("Frontier BFS") << std::endl;

	GPU_ERRCHK(cudaMemcpy(levelHost, levelDevice, numVertices * sizeof(uint32), cudaMemcpyDeviceToHost));

	cudaFree(currFrontierDevice);
	cudaFree(nextFrontierDevice);
	cudaFree(nextFrontierCountDevice);
}