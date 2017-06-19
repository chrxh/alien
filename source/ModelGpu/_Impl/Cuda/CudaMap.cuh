#pragma once

#include "device_functions.h"
#include "CudaSimulationData.cuh"

__host__ __device__ void mapPosCorrection(int2 &pos, int2 const &size)
{
	pos = { ((pos.x % size.x) + size.x) % size.x, ((pos.y % size.y) + size.y) % size.y };
}

__host__ __device__ void mapPosCorrection(float2 &pos, int2 const &size)
{
	int2 intPart{ (int)pos.x, (int)pos.y };
	float2 fracPart = { pos.x - intPart.x, pos.y - intPart.y };
	mapPosCorrection(intPart, size);
	pos = { static_cast<float>(intPart.x) + fracPart.x, static_cast<float>(intPart.y) + fracPart.y };
}

__device__ void angleCorrection_Kernel(int &angle)
{
	angle = ((angle % 360) + 360) % 360;
}

__device__ void angleCorrection_Kernel(float &angle)
{
	int intPart = (int)angle;
	float fracPart = angle - intPart;
	angleCorrection_Kernel(intPart);
	angle = (float)intPart + fracPart;
}

__device__ void mapDisplacementCorrection_Kernel(float2 &disp, int2 const &size)
{
}

__device__ bool isCellPresentAtMap_Kernel(int2 posInt, CudaCell * cell, CudaCell ** map, int2 const &size)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	return map[mapEntry] == cell;

}

__host__ __device__ CudaCell* getCellFromMap(int2 posInt, CudaCell ** __restrict__ map, int2 const &size)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	return map[mapEntry];
}

__host__ __device__ void setCellToMap(int2 posInt, CudaCell * __restrict__ cell, CudaCell ** __restrict__ map, int2 const &size)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	map[mapEntry] = cell;
}

__device__ float mapDistanceSquared_Kernel(float2 const &p, float2 const &q, int2 const &size)
{
	float2 d = { p.x - q.x, p.y - q.y };
	mapDisplacementCorrection_Kernel(d, size);
	return d.x*d.x + d.y*d.y;
}

__device__ void clearOldMap_Kernel(CudaSimulation const &data, int clusterIndex)
{
	CudaCellCluster *oldCluster = &data.clustersAC1.getEntireArray()[clusterIndex];

	int startCellIndex;
	int endCellIndex;
	int2 size = data.size;
	int oldNumCells = oldCluster->numCells;
	if (threadIdx.x >= oldNumCells) {
		return;
	}

	calcPartition(oldNumCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		float2 absPos = oldCluster->cells[cellIndex].absPos;
		setCellToMap({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, nullptr, data.map1, size);
	}
}

__global__ void clearOldMap_Kernel(CudaSimulation data)
{
	int blockIndex = blockIdx.x;
	int numClusters = data.clustersAC1.getNumEntries();
	if (blockIndex >= numClusters) {
		return;
	}
	int startClusterIndex;
	int endClusterIndex;
	calcPartition(numClusters, blockIndex, gridDim.x, startClusterIndex, endClusterIndex);

	for (int clusterIndex = startClusterIndex; clusterIndex <= endClusterIndex; ++clusterIndex) {
		clearOldMap_Kernel(data, clusterIndex);
	}
}
