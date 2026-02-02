#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <fstream>
#include <cstdlib>
#include <iostream>

#include <sstream>
#include <string>

#include <FastNoiseLite.h>
#include <glm/glm.hpp>

#include "csr.h"
#include "utils_cuda.h"
#include "timer.h"

#include "bfs_cpu.h"
#include "bfs_cpu_queue.h"
#include "bfs_gpu.h"
#include "bfs_frontier.h"
#include "bfs_shared.h"
#include "bfs_dynamic_parallel.h"

using namespace glm;

#define PRINT_PATH false

enum BFSAlgorithm
{
	ALGO_BFS_CPU,
	ALGO_BFS_CPU_QUEUE,
	ALGO_BFS_GPU,
	ALGO_BFS_FRONTIER,
	ALGO_BFS_SHARED,
	ALGO_BFS_DYNAMIC_PARALLEL
};

BFSAlgorithm activeAlgorithm = BFSAlgorithm::ALGO_BFS_FRONTIER;

// grid width & height
const uint32 GRID_SIZE = 2048;

const ivec2 START_POS = ivec2(1500, 1350);
const ivec2 TARGET_POS = ivec2(0, 0);

int32 numIterations = 100;
float32 totalDuration = 0.0;
std::vector<float32> meshingDurations;
float32 minDuration = UINT_MAX;
float32 maxDuration = 0.0;
float32 median = 0.0;
float32 standardDeviation = 0.0;

std::vector<uint32> getPath(uint32 startingNode, const CSRGraph& graph,uint32* levels)
{
	std::vector<uint32> path;

	if (levels[startingNode] == UINT_MAX) 
	{
		std::cout << "No path found!" << std::endl;
		return path;
	}

	uint32 currNode = startingNode;
	path.push_back(currNode);

	std::cout << "Starting level: " << levels[currNode] << "\n" << std::endl;

	while (levels[currNode] != 0) 
	{
		uint32 currentLevel = levels[currNode];
		bool found = false;

		std::vector<uint32> possibleEdges;

		for (int32 edge = graph.srcPtrs[currNode]; edge < graph.srcPtrs[currNode + 1]; ++edge)
		{
			uint32 neighbour = graph.dst[edge];

			if (levels[neighbour] < currentLevel)
			{
				possibleEdges.push_back(neighbour);
			}
		}

		int32 edgeRand = rand() % possibleEdges.size();
		uint32 selectedEdge = possibleEdges[edgeRand];
		currNode = selectedEdge;
		path.push_back(selectedEdge);
		found = true;

		if (!found) 
		{
			std::cout << "No further path progression possible!" << std::endl;
			return path;
		}
	}

	std::cout << "Calculated path: " << std::endl;
	for (uint32 node : path) 
		std::cout << node << " ";
	std::cout << std::endl;

	return path;
}

int main()
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
			//float noiseValue = (noise.GetNoise((float)x , (float)y) + 1.0) / 2.0;
			//noiseValue >= 0.5 ? noiseData[index++] = 1 : noiseData[index++] = 0;
			noiseData[index++] = 1;
		}
	}

	CSRGraphHost csr;
	CSRGraph graphDevice;
	
	csr.constructCSR(noiseData, GRID_SIZE);

	graphDevice.numVertices = csr.graph.numVertices;

	GPU_ERRCHK(cudaMalloc(&graphDevice.srcPtrs, (csr.graph.numVertices + 1) * sizeof(int32)));
	GPU_ERRCHK(cudaMalloc(&graphDevice.dst, csr.numEdges * sizeof(int32)));

	GPU_ERRCHK(cudaMemcpy(graphDevice.srcPtrs, csr.graph.srcPtrs, (csr.graph.numVertices + 1) * sizeof(int32), cudaMemcpyHostToDevice));
	GPU_ERRCHK(cudaMemcpy(graphDevice.dst, csr.graph.dst, csr.numEdges * sizeof(int32), cudaMemcpyHostToDevice));

	// target
	uint32 targetNode = csr.getNodeIdFromPos(TARGET_POS);

	std::cout << "Target position: (" << TARGET_POS.x << " " << TARGET_POS.y << ")" << "\n" << std::endl;

	uint32* levelHost = new uint32[csr.graph.numVertices];
	uint32* levelDevice;
	GPU_ERRCHK(cudaMalloc(&levelDevice, csr.graph.numVertices * sizeof(uint32)));

	uint32 numVertices = csr.graph.numVertices;

	Timer timer;

	for (int currIteration = 0; currIteration <= numIterations; ++currIteration) {
		for (int32 i = 0; i < csr.graph.numVertices; i++) {
			levelHost[i] = UINT_MAX;
		}
		levelHost[targetNode] = 0;

		GPU_ERRCHK(cudaMemcpy(levelDevice, levelHost, csr.graph.numVertices * sizeof(uint32), cudaMemcpyHostToDevice));

		uint32 currLevel = 1;

		// first iteration is warm up run
		if (currIteration > 0) {
			timer.Reset();
		}

		switch (activeAlgorithm)
		{
		case BFSAlgorithm::ALGO_BFS_CPU:
			runBFSCPU(csr, levelHost, currLevel, timer);
			break;
		case BFSAlgorithm::ALGO_BFS_CPU_QUEUE:
			runBFSCPUQueue(csr, levelHost, targetNode, timer);
			break;
		case BFSAlgorithm::ALGO_BFS_GPU:
			runBFSGPU(graphDevice, levelDevice, levelHost, numVertices, currLevel, timer);
			break;
		case BFSAlgorithm::ALGO_BFS_FRONTIER:
			runBFSFrontier(graphDevice, levelDevice, levelHost, targetNode, numVertices, currLevel, timer);
			break;
		case BFSAlgorithm::ALGO_BFS_SHARED:
			runBFSShared(graphDevice, levelDevice, levelHost, targetNode, numVertices, currLevel, timer);
			break;
		case BFSAlgorithm::ALGO_BFS_DYNAMIC_PARALLEL:
			runBFSDynamicParallel(graphDevice, levelDevice, levelHost, targetNode, numVertices, currLevel, timer);
			break;
		default:
			break;
		}

		if (currIteration > 0)
		{
			float32 elapsed = timer.ElapsedMs();
			meshingDurations.push_back(elapsed);
			totalDuration += elapsed;
			if (elapsed > maxDuration) maxDuration = elapsed;
			if (elapsed < minDuration) minDuration = elapsed;
		}
	}

	float32 mean = totalDuration / numIterations;

	if (numIterations % 2 == 0) {
		median = (meshingDurations[numIterations / 2 + 1] + meshingDurations[numIterations / 2]) / 2.0;
	}
	else {
		median = meshingDurations[numIterations / 2];
	}

	float32 variance = 0.0;

	for(float32 duration : meshingDurations)
	{
		variance += (duration - mean) * (duration - mean);
	}

	variance /= numIterations;
	standardDeviation = sqrt(variance);

	std::cout << "\nBFS duration:" << std::endl;
	std::cout << " Total: " << totalDuration << " ms" << std::endl;
	std::cout << " Average: " << mean << " ms" << std::endl;
	std::cout << " Min: " << minDuration << " ms" << std::endl;
	std::cout << " Max: " << maxDuration << " ms" << std::endl;
	std::cout << " Median: " << median << " ms" << std::endl;
	std::cout << " Standard Deviation: " << standardDeviation << " ms\n" << std::endl;

#if PRINT_PATH
	std::cout << "Calculated path from start pos (" << START_POS.x << " " << START_POS.y << ") to target pos" << std::endl;
	// start node
	uint32 startNode = csr.getNodeIdFromPos(START_POS);
	std::vector<uint32> path = getPath(startNode, csr.graph, levelHost);

	std::vector<int32> nodeToGridIndex;
	nodeToGridIndex.reserve(csr.graph.numVertices);

	for (int i = 0; i < noiseData.size(); i++) 
	{
		if (noiseData[i] == 1) 
		{
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

	for (int32 nodeID : path) 
	{
		if (nodeID < nodeToGridIndex.size()) 
		{
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
#endif // PRINT_PATH

	// Free memory
	cudaFree(levelDevice);
	cudaFree(graphDevice.srcPtrs);
	cudaFree(graphDevice.dst);
	GPU_ERRCHK(cudaGetLastError());

	delete[] levelHost;

	return 0;
}