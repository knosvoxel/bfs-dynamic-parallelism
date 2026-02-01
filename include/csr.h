#pragma once

#include <glm/glm.hpp>

using namespace glm;

typedef struct CSRGraph {
	int32* srcPtrs;
	int32* dst;
	uint32 numVertices = 0;
};

struct CSRGraphHost {
	CSRGraph graph;
	uint32 gridSize = 0;
	uint32 numEdges = 0;
	std::vector<int32> gridPosToNodeID;

	~CSRGraphHost() {
		delete[] graph.srcPtrs;
		delete[] graph.dst;
		graph.srcPtrs = nullptr;
		graph.dst = nullptr;
	}

	void constructCSR(const std::vector<int>& noiseData, const uint32 gridSize) {
		this->gridSize = gridSize;
		gridPosToNodeID = std::vector<int32>(gridSize * gridSize, -1);

		int nodeCount = 0;
		for (int32 i = 0; i < noiseData.size(); i++)
		{
			if (noiseData[i] == 1) {
				gridPosToNodeID[i] = nodeCount++;

			}
		}

		graph.numVertices = nodeCount;

		std::vector<int32> tempSrcPtrs;
		std::vector<int32> tempDst;
		tempSrcPtrs.reserve(nodeCount + 1);
		tempSrcPtrs.push_back(0);

		for (int32 y = 0; y < gridSize; y++)
		{
			for (int32 x = 0; x < gridSize; x++)
			{
				int32 idx = y * gridSize + x;

				if (noiseData[idx] == 1)
				{
					vec2 neighbours[4] = {
						vec2(0, 1),
						vec2(0, -1),
						vec2(1, 0),
						vec2(-1, 0)
					};

					for (int32 i = 0; i < 4; i++)
					{
						vec2 neighbourPos = vec2(x, y) + neighbours[i];

						if (neighbourPos.x >= 0 && neighbourPos.x < gridSize &&
							neighbourPos.y >= 0 && neighbourPos.y < gridSize)
						{
							int32 neighbourIdx = neighbourPos.y * gridSize + neighbourPos.x;
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

		numEdges = graph.srcPtrs[graph.numVertices];
		
		std::cout << "Node Count: " << nodeCount << std::endl;
		std::cout << "Edge Count: " << numEdges << "\n" << std::endl;


	}

	int getNodeIdFromPos(ivec2 pos) {
		if (pos.x < 0 || pos.x >= gridSize || pos.y < 0 || pos.y >= gridSize) {
			return -1;
		}
		int32 gridIdx = pos.y * gridSize + pos.x;
		return gridPosToNodeID[gridIdx];
	}
};