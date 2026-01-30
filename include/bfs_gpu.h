#pragma once

#include <cuda_runtime.h>
#include <iostream>

#include "csr.h"
#include "timer.h"
#include "utils_cuda.h"

__global__
void bfs(CSRGraph csrGraph, uint32* level, uint32* newVertexVisited, uint32 currLevel)
{
	uint32 vertex = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertex < csrGraph.numVertices)
	{
		if (level[vertex] == currLevel - 1)
		{
			for (uint32 edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge)
			{
				uint32 neighbour = csrGraph.dst[edge];
				if (level[neighbour] == UINT_MAX)
				{
					level[neighbour] = currLevel;
					*newVertexVisited = 1;
				}
			}
		}
	}
}

void runBFSGPU(CSRGraph& graphDevice, uint32* levelDevice, uint32* levelHost, uint32 numVertices, uint32& currLevel, Timer& timer)
{
	uint32* newVertexVisitedDevice;
	GPU_ERRCHK(cudaMalloc(&newVertexVisitedDevice, sizeof(uint32)));

	int32 threadsPerBlock = 256;
	int32 blocksPerGrid = (numVertices + threadsPerBlock - 1) / threadsPerBlock;
	dim3 numBlocks(blocksPerGrid, 1, 1);
	dim3 numThreads(threadsPerBlock, 1, 1);

	uint32 newVertexVisitedHost = 1;

	while (newVertexVisitedHost > 0)
	{
		newVertexVisitedHost = 0;
		GPU_ERRCHK(cudaMemcpy(newVertexVisitedDevice, &newVertexVisitedHost, sizeof(uint32), cudaMemcpyHostToDevice));

		bfs << <numBlocks, numThreads >> > (graphDevice, levelDevice, newVertexVisitedDevice, currLevel);

		GPU_ERRCHK(cudaMemcpy(&newVertexVisitedHost, newVertexVisitedDevice, sizeof(uint32), cudaMemcpyDeviceToHost));

		currLevel++;

		if (currLevel > numVertices) break;
	}

	GPU_ERRCHK(cudaDeviceSynchronize());

	std::cout << "BFS finished after " << currLevel - 1 << " levels." << std::endl;

	std::cout << timer.ToString("BFS GPU") << std::endl;

	GPU_ERRCHK(cudaMemcpy(levelHost, levelDevice, numVertices * sizeof(uint32), cudaMemcpyDeviceToHost));

	cudaFree(newVertexVisitedDevice);
}