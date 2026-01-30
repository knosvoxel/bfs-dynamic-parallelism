#pragma once

#include <cuda_runtime.h>
#include <iostream>

#include "csr.h"
#include "timer.h"
#include "utils_cuda.h"

#define LOCAL_FRONTIER_CAPACITY 1024

__device__
uint32 blocksFinished = 0;

__global__
void bfsDynamicParallel(CSRGraph csrGraph, uint32* level, uint32* prevFrontier, uint32* currFrontier, uint32 numPrevFrontier, uint32* numCurrFrontier, uint32 currLevel, uint32* finalLevel)
{
	__shared__ uint32 currFrontier_s[LOCAL_FRONTIER_CAPACITY];
	__shared__ uint32 numCurrFrontier_s;
	if (threadIdx.x == 0)
	{
		numCurrFrontier_s = 0;
	}
	__syncthreads();

	uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numPrevFrontier)
	{
		uint32 vertex = prevFrontier[i];
		for (uint32 edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
		{
			uint32 neighbour = csrGraph.dst[edge];
			if (atomicCAS(&level[neighbour], UINT_MAX, currLevel) == UINT_MAX)
			{
				uint32 currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
				if (currFrontierIdx_s < LOCAL_FRONTIER_CAPACITY)
				{
					currFrontier_s[currFrontierIdx_s] = neighbour;
				}
				else
				{
					numCurrFrontier_s = LOCAL_FRONTIER_CAPACITY;
					uint32 currFrontierIdx = atomicAdd(numCurrFrontier, 1);
					currFrontier[currFrontierIdx] = neighbour;
				}
			}
		}
	}
	__syncthreads();

	__shared__ uint32 currFrontierStartIdx;
	if (threadIdx.x == 0)
	{
		currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
	}
	__syncthreads();

	for (uint32 currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s; currFrontierIdx_s += blockDim.x)
	{
		uint32 currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
		currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
	}

	__threadfence();

	__shared__ bool isLastBlock;
	if (threadIdx.x == 0) {
		uint32 ticket = atomicAdd(&blocksFinished, 1);
		isLastBlock = (ticket == gridDim.x - 1);
	}

	__syncthreads();

	if (isLastBlock && threadIdx.x == 0) {
		blocksFinished = 0;

		uint32 totalCount = *numCurrFrontier;

		if (totalCount > 0) {
			uint32 nextBlockCount = (totalCount + blockDim.x - 1) / blockDim.x;

			*numCurrFrontier = 0;

			bfsDynamicParallel << <nextBlockCount, blockDim.x, 0, cudaStreamTailLaunch >> > (csrGraph, level, currFrontier, prevFrontier, totalCount, numCurrFrontier, currLevel + 1, finalLevel);
		}
		else {
			*finalLevel = currLevel;
		}
	}
}

void runBFSDynamicParallel(CSRGraph& graphDevice, uint32* levelDevice, uint32* levelHost, uint32 targetNode, uint32 numVertices, uint32& currLevel, Timer& timer)
{
	uint32* currFrontierDevice, * nextFrontierDevice, * nextFrontierCountDevice, * finalLevelDevice;

	GPU_ERRCHK(cudaMalloc(&currFrontierDevice, numVertices * sizeof(uint32)));
	GPU_ERRCHK(cudaMalloc(&nextFrontierDevice, numVertices * sizeof(uint32)));
	GPU_ERRCHK(cudaMalloc(&nextFrontierCountDevice, sizeof(uint32)));
	GPU_ERRCHK(cudaMalloc(&finalLevelDevice, sizeof(uint32)));

	GPU_ERRCHK(cudaMemcpy(currFrontierDevice, &targetNode, sizeof(uint32), cudaMemcpyHostToDevice));

	uint32 numFrontierElements = 1;
	uint32 finalLevelHost = 0;
	int32 threadsPerBlock = 256;

	bfsDynamicParallel << <1, threadsPerBlock >> > (graphDevice, levelDevice, currFrontierDevice, nextFrontierDevice, numFrontierElements, nextFrontierCountDevice, currLevel, finalLevelDevice);

	GPU_ERRCHK(cudaDeviceSynchronize());

	GPU_ERRCHK(cudaMemcpy(&finalLevelHost, finalLevelDevice, sizeof(uint32), cudaMemcpyDeviceToHost));

	std::cout << "Dynamic Parallel BFS finished after " << finalLevelHost << " levels." << std::endl;

	std::cout << timer.ToString("Dynamic Parallel BFS") << std::endl;

	GPU_ERRCHK(cudaMemcpy(levelHost, levelDevice, numVertices * sizeof(uint32), cudaMemcpyDeviceToHost));

	cudaFree(currFrontierDevice);
	cudaFree(nextFrontierDevice);
	cudaFree(nextFrontierCountDevice);
	cudaFree(finalLevelDevice);
}