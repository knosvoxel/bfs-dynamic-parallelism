#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <fstream>

#include <sstream>
#include <string>

#include <FastNoiseLite.h>
#include <glm/glm.hpp>

#define VERTEX_COUNT 4

using namespace glm;

// Kernel function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{
	int index = threadIdx.x;
	int stride = blockDim.x;
	for(int i = index; i < n; i+= stride)
		y[i] = x[i] + y[i];
}

void constructCSR(const std::vector<int>& noiseData, std::vector<int>& rowPtrs, std::vector<int>& colIndices) {
	std::vector<int> gridPosToNodeID(noiseData.size());
	int currNodeCount = 0;

	for (int i = 0; i < noiseData.size(); i++)
	{
		if (noiseData[i] == 1) {
			gridPosToNodeID[i] = currNodeCount++;
		}
	}

	rowPtrs.push_back(0);

	for (int y = 0; y < VERTEX_COUNT; y++)
	{
		for (int x = 0; x < VERTEX_COUNT; x++)
		{
			int idx = y * VERTEX_COUNT + x;

			std::cout << noiseData[idx] << " ";

			if (noiseData[idx] == 1)
			{
				int dx[] = { 0, 0, 1, -1 };
				int dy[] = { 1, -1, 0, 0 };

				for (int i = 0; i < 4; i++)
				{
					int nx = x + dx[i];
					int ny = y + dy[i];

					if (nx >= 0 && nx < VERTEX_COUNT && ny >= 0 && ny < VERTEX_COUNT)
					{
						int v = ny * VERTEX_COUNT + nx;
						if (noiseData[v] == 1)
						{
							colIndices.push_back(gridPosToNodeID[v]);
						}
					}
				}
				rowPtrs.push_back((int)colIndices.size());
			}
		}
		std::cout << "" << std::endl;
	}

	std::cout << "\n" << std::endl;
}

int main(void)
{
	//FastNoiseLite noise;
	//noise.SetNoiseType(FastNoiseLite::NoiseType_Perlin);
	//noise.SetFractalType(FastNoiseLite::FractalType_FBm);

	//std::vector<int> noiseData(VERTEX_COUNT * VERTEX_COUNT);
	//int index = 0;

	//for (int y = 0; y < VERTEX_COUNT; y++)
	//{
	//	for (int x = 0; x < VERTEX_COUNT; x++)
	//	{
	//		float noiseValue = (noise.GetNoise((float)x, (float)y) + 1.0) / 2.0;
	//		noiseValue >= 0.5 ? noiseData[index++] = 1 : noiseData[index++] = 0;
	//	}
	//}

	//std::ofstream ofs("noise.ppm", std::ios::binary);

	//std::stringstream ss;
	//ss << "P6\n" << VERTEX_COUNT << " " << VERTEX_COUNT << "\n255\n";
	//std::string imageHeader = ss.str();
	//ofs << imageHeader;

	//for (float val : noiseData)
	//{
	//	unsigned char color = (unsigned char)(val * 255.0f);
	//	ofs << color << color << color;
	//}
	//ofs.close();
	std::vector<int> rowPtrs;
	std::vector<int> colIndices;

	std::vector<int> noiseData =
	{
		1, 1, 0, 0,
		1, 1, 1, 1,
		0, 1, 1, 0,
		0, 0, 1, 1
	};

	constructCSR(noiseData, rowPtrs, colIndices);

	std::cout << "rowPtrs: ";
	for (int i = 0; i < rowPtrs.size(); i++)
	{
		std::cout << rowPtrs[i] << " ";
	}

	std::cout << "\n\n" << std::endl;

	std::cout << "colIndices: ";
	for (int i = 0; i < colIndices.size(); i++)
	{
		std::cout << colIndices[i] << " ";
	}

	std::cout << "\n\n" << std::endl;

	int N = 1 << 20;
	float* x, * y;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));

	// initialize x and y arrays on the host
	for (int i = 0; i < N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// Run kernel on 1M elements on the GPU
	add<<<1, 256 >>>(N, x, y);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(y[i] - 3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	// Free memory
	cudaFree(x);
	cudaFree(y);
	return 0;
}