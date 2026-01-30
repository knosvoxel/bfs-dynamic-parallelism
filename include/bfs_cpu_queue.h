#pragma once

#include <iostream>
#include <deque>
#include <algorithm>

#include "csr.h"
#include "timer.h"

void runBFSCPUQueue(CSRGraphHost& csr, uint32* levelHost, uint32 targetNode, Timer& timer)
{
	std::deque<uint32> frontier;

	frontier.push_back(targetNode);

	uint32 lastLevel = 0;

	while (!frontier.empty()) {
		uint32 v = frontier.front();
		frontier.pop_front();

		uint32 neighborLevel = levelHost[v] + 1;

		for (uint32 edge = csr.graph.srcPtrs[v]; edge < csr.graph.srcPtrs[v + 1]; ++edge) {
			uint32 neighbor = csr.graph.dst[edge];

			if (levelHost[neighbor] == UINT_MAX) {
				levelHost[neighbor] = neighborLevel;
				lastLevel = std::max(lastLevel, neighborLevel);

				frontier.push_back(neighbor);
			}
		}
	}
	std::cout << "Optimized CPU BFS finished. Max level: " << lastLevel + 1 << std::endl;

	std::cout << timer.ToString("BFS CPU QUEUE") << std::endl;
}