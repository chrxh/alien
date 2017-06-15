#pragma once

struct CudaData
{
	int2 size;

	CudaCell **map1;
	CudaCell **map2;

	ArrayController<CudaCellCluster> clustersAC1;
	ArrayController<CudaCellCluster> clustersAC2;
	ArrayController<CudaCell> cellsAC1;
	ArrayController<CudaCell> cellsAC2;

/*
	CudaData(int2 const &size_)
		: size(size_)
		, clustersAC1(static_cast<int>(NUM_CLUSTERS * 1.1))
		, clustersAC2(static_cast<int>(NUM_CLUSTERS * 1.1))
		, cellsAC1(static_cast<int>(NUM_CLUSTERS * 30 * 30 * 1.1))
		, cellsAC2(static_cast<int>(NUM_CLUSTERS * 30 * 30 * 1.1))
	{
		size_t mapSize = size.x * size.y * sizeof(CudaCell*);
		cudaMallocManaged(&map1, mapSize);
		cudaMallocManaged(&map2, mapSize);
		checkCudaErrors(cudaGetLastError());
		for (int i = 0; i < size.x * size.y; ++i) {
			map1[i] = nullptr;
			map2[i] = nullptr;
		}

	}
*/

/*
	~CudaData()
	{
		cellsAC1.free();
		clustersAC1.free();
		cellsAC2.free();
		clustersAC2.free();

		cudaFree(map1);
		cudaFree(map2);
	}

*/
	void swapData()
	{
		swap(clustersAC1, clustersAC2);
		swap(cellsAC1, cellsAC2);
		swap(map1, map2);
	}

	void prepareTargetData()
	{
		clustersAC2.reset();
		cellsAC2.reset();
	}
};
