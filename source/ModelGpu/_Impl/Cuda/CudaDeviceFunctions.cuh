#pragma once

#include "CudaShared.cuh"
#include "CudaBase.cuh"
#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#define LAYERS 2

struct CudaData
{
	int2 size;

	CellCuda **map1;
	CellCuda **map2;

	ArrayController<ClusterCuda> clustersAC1;
	ArrayController<ClusterCuda> clustersAC2;
	ArrayController<CellCuda> cellsAC1;
	ArrayController<CellCuda> cellsAC2;

};

__device__ void mapCorrection_Kernel(int2 &pos, int2 const &size)
{
	pos = { ((pos.x % size.x) + size.x) % size.x, ((pos.y % size.y) + size.y) % size.y };
}

__device__ void mapCorrection_Kernel(double2 &pos, int2 const &size)
{
	int2 intPart{ (int)pos.x, (int)pos.y };
	double2 fracPart = { pos.x - intPart.x, pos.y - intPart.y };
	mapCorrection_Kernel(intPart, size);
	pos = { (double)intPart.x + fracPart.x, (double)intPart.y + fracPart.y };
}

__device__ void angleCorrection_Kernel(int &angle)
{
	angle = ((angle % 360) + 360) % 360;
}

__device__ void angleCorrection_Kernel(double &angle)
{
	int intPart = (int)angle;
	double fracPart = angle - intPart;
	angleCorrection_Kernel(intPart);
	angle = (double)intPart + fracPart;
}

__device__ CellCuda* readCellMapEntry_Kernel(int2 posInt, int clusterIndex, CellCuda ** __restrict__ map, int2 const &size)
{
	mapCorrection_Kernel(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	auto slice = size.x*size.y;

	for (int i = 0; i < LAYERS; ++i) {
		auto index = map[mapEntry + i * slice];
		if (index != nullptr) {
			return index;
			/*
			if (cellMapEntryArray[index].clusterIndex != clusterIndex) {
			return index;
			}
			*/
		}
	}
	return nullptr;
}

__device__ CellCuda* readCellMapEntries_Kernel(int2 posInt, int clusterIndex, CellCuda ** __restrict__ map, int2 const &size)
{
	--posInt.x;
	--posInt.y;
	auto index = readCellMapEntry_Kernel(posInt, clusterIndex, map, size);
	if (index) { return index; }

	++posInt.x;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, size);
	if (index) { return index; }

	++posInt.x;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, size);
	if (index) { return index; }

	++posInt.y;
	posInt.x -= 2;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, size);
	if (index) { return index; }

	++posInt.y;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, size);
	if (index) { return index; }

	++posInt.y;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, size);
	if (index) { return index; }

	++posInt.y;
	posInt.x -= 2;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, size);
	if (index) { return index; }

	++posInt.y;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, size);
	if (index) { return index; }

	++posInt.y;
	index = readCellMapEntry_Kernel(posInt, clusterIndex, map, size);
	if (index) { return index; }
	return nullptr;
}

__device__ int ii = 4;

__device__ void movement_Kernel(CudaData &data, int clusterIndex)
{
	ClusterCuda cluster = data.clustersAC1.getEntireArrayKernel()[clusterIndex];
	if (threadIdx.x >= cluster.numCells) {
		return;
	}

	int2 size = data.size;
	int startCellIndex;
	int endCellIndex;
	tiling_Kernel(cluster.numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

	__shared__ double rotMatrix[2][2];
	if (threadIdx.x == 0) {
		double sinAngle = __sinf(cluster.angle*DEG_TO_RAD);
		double cosAngle = __cosf(cluster.angle*DEG_TO_RAD);
		rotMatrix[0][0] = cosAngle;
		rotMatrix[0][1] = sinAngle;
		rotMatrix[1][0] = -sinAngle;
		rotMatrix[1][1] = cosAngle;
	}

	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellCuda* cell = &cluster.cells[cellIndex];
		double2 relPos = cell->relPos;
		relPos = { relPos.x*rotMatrix[0][0] + relPos.y*rotMatrix[0][1], relPos.x*rotMatrix[1][0] + relPos.y*rotMatrix[1][1] };
		double2 absPos = { relPos.x + cluster.pos.x, relPos.y + cluster.pos.y };
		int2 absPosInt = { (int)absPos.x , (int)absPos.y };
		cell->absPos = absPos;

		readCellMapEntries_Kernel(absPosInt, clusterIndex, data.map1, size);
	}

	if (threadIdx.x == 0) {
		cluster.angle += cluster.angularVel;
		angleCorrection_Kernel(cluster.angle);
		cluster.pos = { cluster.pos.x + cluster.vel.x, cluster.pos.y + cluster.vel.y };
		mapCorrection_Kernel(cluster.pos, size);

		*data.clustersAC2.getElementKernel() = cluster;
	}
	__syncthreads();

	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		CellCuda* cell = &cluster.cells[cellIndex];
		double2 relPos = cell->relPos;
		relPos = { relPos.x*rotMatrix[0][0] + relPos.y*rotMatrix[0][1], relPos.x*rotMatrix[1][0] + relPos.y*rotMatrix[1][1] };
		double2 absPos = { relPos.x + cluster.pos.x, relPos.y + cluster.pos.y };
		int2 absPosInt = { (int)absPos.x , (int)absPos.y };
		mapCorrection_Kernel(absPosInt, size);
		//		newMap[absPosInt.x + absPosInt.y * config.size.x] = 1;
	}
	__syncthreads();
}


__global__ void movement_Kernel(CudaData data)
{

	int blockIndex = blockIdx.x;
	int numClusters = data.clustersAC1.getNumEntriesKernel();
	if (blockIndex >= numClusters) {
		return;
	}
	int startClusterIndex;
	int endClusterIndex;
	tiling_Kernel(numClusters, blockIndex, gridDim.x, startClusterIndex, endClusterIndex);

	for (int clusterIndex = startClusterIndex; clusterIndex <= endClusterIndex; ++clusterIndex) {
		movement_Kernel(data, clusterIndex);
	}
}
