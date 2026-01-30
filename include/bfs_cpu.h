#pragma once

#include <iostream>

#include "csr.h"
#include "timer.h"

void runBFSCPU(CSRGraphHost& csr, uint32* levelHost, uint32& currLevel, Timer& timer)
{
	bool changed = true;

	while (changed) {
		changed = false;

		for (uint32 v = 0; v < csr.graph.numVertices; v++) {
			if (levelHost[v] == currLevel - 1) {
				for (uint32 edge = csr.graph.srcPtrs[v]; edge < csr.graph.srcPtrs[v + 1]; ++edge) {
					uint32 neighbor = csr.graph.dst[edge];

					if (levelHost[neighbor] == UINT_MAX) {
						levelHost[neighbor] = currLevel;
						changed = true;
					}
				}
			}
		}
		currLevel++;

		if (currLevel > csr.graph.numVertices) break;
	}
	std::cout << "CPU BFS finished after " << currLevel << " levels." << std::endl;

	std::cout << timer.ToString("BFS CPU") << std::endl;
}