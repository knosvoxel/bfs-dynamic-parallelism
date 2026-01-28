#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <fstream>
#include <cstdlib>

#include <sstream>
#include <string>

#include <FastNoiseLite.h>
#include <glm/glm.hpp>

#include "utils_cuda.h"

// grid width & height
#define GRID_SIZE 128

using namespace glm;

typedef struct CSRGraph {
	int32* srcPtrs;
	int32* dst;
	uint32 numVertices = 0;
};

__global__
void bfs(CSRGraph csrGraph, uint32* level, uint32* newVertexVisited, uint32 currLevel)
{
	uint32 vertex = blockIdx.x * blockDim.x + threadIdx.x;
	if (vertex < csrGraph.numVertices) {
		if (level[vertex] == currLevel - 1) {
			for (uint32 edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
				uint32 neighbour = csrGraph.dst[edge];
				if (level[neighbour] == UINT_MAX) {
					level[neighbour] = currLevel;
					*newVertexVisited = 1;
				}
			}
		}
	}
}

void constructCSR(const std::vector<int>& noiseData, CSRGraph& graph) {
	std::vector<int> gridPosToNodeID(noiseData.size());
	int nodeCount = 0;

	for (int i = 0; i < noiseData.size(); i++)
	{
		if (noiseData[i] == 1) {
			gridPosToNodeID[i] = nodeCount;
			nodeCount++;

		}
	}

	graph.numVertices = nodeCount;

	std::cout << "Node Count: " << nodeCount << "\n" << std::endl;

	std::vector<int> tempSrcPtrs;
	std::vector<int> tempDst;

	tempSrcPtrs.push_back(0);

	for (int y = 0; y < GRID_SIZE; y++)
	{
		for (int x = 0; x < GRID_SIZE; x++)
		{
			int idx = y * GRID_SIZE + x;

			if (noiseData[idx] == 1)
			{
				vec2 neighbours[4] = { 
					vec2(0, 1), 
					vec2(0, -1), 
					vec2(1, 0), 
					vec2(-1, 0) 
				};

				for (int i = 0; i < 4; i++)
				{
					vec2 neighbourPos = vec2(x, y) + neighbours[i];

					if (neighbourPos.x >= 0 && neighbourPos.x < GRID_SIZE &&
						neighbourPos.y >= 0 && neighbourPos.y < GRID_SIZE)
					{
						int neighbourIdx = neighbourPos.y * GRID_SIZE + neighbourPos.x;
						if (noiseData[neighbourIdx] == 1)
						{
							tempDst.push_back(gridPosToNodeID[neighbourIdx]);
						}
					}
				}
				tempSrcPtrs.push_back((int)tempDst.size());
			}
		}
	}

	graph.srcPtrs = new int32[tempSrcPtrs.size()];
	graph.dst = new int32[tempDst.size()];

	std::copy(tempSrcPtrs.begin(), tempSrcPtrs.end(), graph.srcPtrs);
	std::copy(tempDst.begin(), tempDst.end(), graph.dst);
}

std::vector<uint32> getPath(uint32 startingNode, const CSRGraph& graph,uint32* levels)
{
	std::vector<uint32> path;

	if (levels[startingNode] == UINT_MAX) {
		std::cout << "No path found!" << std::endl;
		return path;
	}

	uint32 currNode = startingNode;
	path.push_back(currNode);

	std::cout << "Starting level: " << levels[currNode] << std::endl;

	while (levels[currNode] != 0) {
		uint32 currentLevel = levels[currNode];
		bool found = false;

		std::vector<uint32> possibleEdges;

		for (int32 edge = graph.srcPtrs[currNode]; edge < graph.srcPtrs[currNode + 1]; ++edge)
		{
			uint32 neighbour = graph.dst[edge];

			if (levels[neighbour] < currentLevel)
			{
				//currNode = neighbour;
				//path.push_back(currNode);
				possibleEdges.push_back(neighbour);
				//found = true;
				//break;
			}
		}

		int32 edgeRand = rand() % possibleEdges.size();
		uint32 selectedEdge = possibleEdges[edgeRand];
		currNode = selectedEdge;
		path.push_back(selectedEdge);
		found = true;
		//break;

		if (!found) {
			std::cout << "No further path progression possible!" << std::endl;
			return path;
		}
	}

	std::cout << "Path found: ";
	for (uint32 node : path) 
		std::cout << node << " ";
	std::cout << std::endl;

	return path;
}

int main(void)
{
	FastNoiseLite noise;
	noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
	noise.SetFractalType(FastNoiseLite::FractalType_FBm);

	std::vector<int> noiseData(GRID_SIZE * GRID_SIZE);
	int index = 0;

	for (int y = 0; y < GRID_SIZE; y++)
	{
		for (int x = 0; x < GRID_SIZE; x++)
		{
			float noiseValue = (noise.GetNoise((float)x , (float)y) + 1.0) / 2.0;
			noiseValue >= 0.5 ? noiseData[index++] = 1 : noiseData[index++] = 0;
		}
	}

	CSRGraph graphHost, graphDevice;
	constructCSR(noiseData, graphHost);

	graphDevice.numVertices = graphHost.numVertices;

	GPU_ERRCHK(cudaMalloc(&graphDevice.srcPtrs, (graphHost.numVertices + 1) * sizeof(int32)));
	GPU_ERRCHK(cudaMalloc(&graphDevice.dst, graphHost.srcPtrs[graphHost.numVertices] * sizeof(int32)));

	GPU_ERRCHK(cudaMemcpy(graphDevice.srcPtrs, graphHost.srcPtrs, (graphHost.numVertices + 1) * sizeof(int32), cudaMemcpyHostToDevice));
	GPU_ERRCHK(cudaMemcpy(graphDevice.dst, graphHost.dst, graphHost.srcPtrs[graphHost.numVertices] * sizeof(int32), cudaMemcpyHostToDevice));

	uint32* levelHost = new uint32[graphHost.numVertices];

	for (int32 i = 0; i < graphHost.numVertices; i++)
	{
		levelHost[i] = UINT_MAX;
	}

	// target
	levelHost[400] = 0; // TODO: function that turns 3D position into target

	uint32* levelDevice, * newVertexVisitedDevice = nullptr;

	GPU_ERRCHK(cudaMalloc(&levelDevice, graphHost.numVertices * sizeof(uint32)));
	GPU_ERRCHK(cudaMalloc(&newVertexVisitedDevice, sizeof(uint32)));
	GPU_ERRCHK(cudaMemcpy(levelDevice, levelHost, graphHost.numVertices * sizeof(uint32), cudaMemcpyHostToDevice));

	int32 threadsPerBlock = 1024;
	int32 blocksPerGrid = (graphHost.numVertices + threadsPerBlock - 1) / threadsPerBlock;

	dim3 numThreads(threadsPerBlock, 1, 1);
	dim3 numBlocks(blocksPerGrid, 1, 1);

	uint32 newVertexVisitedHost = 1;
	uint32 currLevel = 1;

	while (newVertexVisitedHost > 0) {
		newVertexVisitedHost = 0;
		GPU_ERRCHK(cudaMemcpy(newVertexVisitedDevice, &newVertexVisitedHost, sizeof(uint32), cudaMemcpyHostToDevice));

		bfs << <numBlocks, numThreads >> > (graphDevice, levelDevice, newVertexVisitedDevice, currLevel);

		GPU_ERRCHK(cudaMemcpy(&newVertexVisitedHost, newVertexVisitedDevice, sizeof(uint32), cudaMemcpyDeviceToHost));

		currLevel++;

		if (currLevel > graphHost.numVertices) break;
	}

	std::cout << "BFS finished after " << currLevel - 1 << " levels." << std::endl;

	// Wait for GPU to finish before accessing on host
	GPU_ERRCHK(cudaDeviceSynchronize());

	GPU_ERRCHK(cudaMemcpy(levelHost, levelDevice , graphHost.numVertices * sizeof(uint32), cudaMemcpyDeviceToHost));


	std::vector<uint32> path = getPath(11000, graphHost, levelHost);

	std::vector<int32> nodeToGridIndex;
	nodeToGridIndex.reserve(graphHost.numVertices);

	for (int i = 0; i < noiseData.size(); i++) {
		if (noiseData[i] == 1) {
			nodeToGridIndex.push_back(i);
		}
	}

	std::vector<unsigned char> rgbImage(GRID_SIZE * GRID_SIZE * 3);

	for (int32 i = 0; i < noiseData.size(); i++)
	{
		unsigned char val = (noiseData[i] == 1) ? 255 : 0;
		rgbImage[i * 3] = val;
		rgbImage[i * 3 + 1] = val;
		rgbImage[i * 3 + 2] = val;
	}

	for (int32 nodeID : path) {
		if (nodeID < nodeToGridIndex.size()) {
			int gridIdx = nodeToGridIndex[nodeID];
			rgbImage[gridIdx * 3] = 255;
			rgbImage[gridIdx * 3 + 1] = 0;
			rgbImage[gridIdx * 3 + 2] = 0;
		}
	}

	std::ofstream ofs("noise.ppm", std::ios::binary);

	std::stringstream ss;
	ss << "P6\n" << GRID_SIZE << " " << GRID_SIZE << "\n255\n";
	ofs << ss.str();

	ofs.write(reinterpret_cast<const char*>(rgbImage.data()), rgbImage.size());
	ofs.close();

	// Free memory
	cudaFree(levelDevice);
	cudaFree(newVertexVisitedDevice);
	cudaFree(graphDevice.srcPtrs);
	cudaFree(graphDevice.dst);
	GPU_ERRCHK(cudaGetLastError());

	delete[] graphHost.srcPtrs;
	delete[] graphHost.dst;
	delete[] levelHost;

	return 0;
}