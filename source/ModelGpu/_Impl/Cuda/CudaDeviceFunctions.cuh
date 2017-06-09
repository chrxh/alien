#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaShared.cuh"
#include "CudaConstants.cuh"
#include "CudaBase.cuh"
#include "CudaCollision.cuh"

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

__device__ inline void updateCollisionData_Kernel(int2 posInt, ClusterCuda *cluster, CellCuda ** __restrict__ map, int2 const &size
	, CollisionData &collisionData)
{
	mapCorrection_Kernel(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
//	auto slice = size.x*size.y;

//	for (int i = 0; i < LAYERS; ++i) {
		auto cell = map[mapEntry/* + i * slice*/];
		if (cell != nullptr) {
			if (cell->cluster != cluster) {
//				if(cell->absPos)
				atomicAdd(&collisionData.numCollisions, 1);
				calcCollision_Kernel(cluster, cell, collisionData);
			}
		}
//	}
}

__device__ inline bool isCellPresentAtMap_Kernel(int2 posInt, CellCuda *cell, CellCuda ** __restrict__ map, int2 const &size)
{
	mapCorrection_Kernel(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	return map[mapEntry] == cell;

}

__device__ inline void collectCollisionData_Kernel(CudaData const &data, CellCuda *cell, CollisionData &collisionData)
{
	ClusterCuda *cluster = cell->cluster;
	auto absPos = cell->absPos;
	auto map = data.map1;
	auto size = data.size;
	int2 posInt = { (int)absPos.x , (int)absPos.y };
	if (!isCellPresentAtMap_Kernel(posInt, cell, map, size)) {
		return;
	}

	--posInt.x;
	--posInt.y;
	updateCollisionData_Kernel(posInt, cluster, map, size, collisionData);
	++posInt.x;
	updateCollisionData_Kernel(posInt, cluster, map, size, collisionData);
	++posInt.x;
	updateCollisionData_Kernel(posInt, cluster, map, size, collisionData);

	++posInt.y;
	posInt.x -= 2;
	updateCollisionData_Kernel(posInt, cluster, map, size, collisionData);
	posInt.y += 2;
	updateCollisionData_Kernel(posInt, cluster, map, size, collisionData);

	++posInt.y;
	posInt.x -= 2;
	updateCollisionData_Kernel(posInt, cluster, map, size, collisionData);
	++posInt.y;
	updateCollisionData_Kernel(posInt, cluster, map, size, collisionData);
	++posInt.y;
	updateCollisionData_Kernel(posInt, cluster, map, size, collisionData);
}

__device__ void inline setCellToMap_Kernel(int2 const &posInt, CellCuda *cell, CellCuda ** __restrict__ map, int2 const &size)
{
	auto mapEntry = posInt.x + posInt.y * size.x;
	map[mapEntry] = cell;
}

__device__ void inline movement_Kernel(CudaData &data, int clusterIndex)
{
	__shared__ ClusterCuda clusterCopy;
	__shared__ ClusterCuda *newCluster;
	__shared__ CellCuda *newCells;
	__shared__ double rotMatrix[2][2];
	__shared__ CollisionData collisionData;

	ClusterCuda *oldCluster = &data.clustersAC1.getEntireArray_Kernel()[clusterIndex];
	int startCellIndex;
	int endCellIndex;
	int2 size = data.size;
	int oldNumCells = oldCluster->numCells;
	if (threadIdx.x < oldNumCells) {

		tiling_Kernel(oldCluster->numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

		if (threadIdx.x == 0) {
			clusterCopy = *oldCluster;
			double sinAngle = __sinf(clusterCopy.angle*DEG_TO_RAD);
			double cosAngle = __cosf(clusterCopy.angle*DEG_TO_RAD);
			rotMatrix[0][0] = cosAngle;
			rotMatrix[0][1] = sinAngle;
			rotMatrix[1][0] = -sinAngle;
			rotMatrix[1][1] = cosAngle;
			newCells = data.cellsAC2.getArray_Kernel(clusterCopy.numCells);
			newCluster = data.clustersAC2.getElement_Kernel();

			collisionData.init_Kernel();
		}
	}
	
	__syncthreads();

	if (threadIdx.x < oldNumCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellCuda *oldCell = &oldCluster->cells[cellIndex];
			collectCollisionData_Kernel(data, oldCell, collisionData);
		}
	}

	__syncthreads();

	if (threadIdx.x == 0) {

		if (collisionData.numCollisions > 0) {
			double numCollisions = static_cast<double>(collisionData.numCollisions);
			clusterCopy.vel.x += collisionData.velDelta.x / numCollisions;
			clusterCopy.vel.y += collisionData.velDelta.y / numCollisions;
			clusterCopy.angularVel += collisionData.angularVelDelta / numCollisions;
		}

		clusterCopy.angle += clusterCopy.angularVel;
		angleCorrection_Kernel(clusterCopy.angle);
		clusterCopy.pos.x += clusterCopy.vel.x;
		clusterCopy.pos.y += clusterCopy.vel.y;
		mapCorrection_Kernel(clusterCopy.pos, size);
		clusterCopy.cells = newCells;
		*newCluster = clusterCopy;
	}

	__syncthreads();

	if (threadIdx.x < oldNumCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellCuda *oldCell = &oldCluster->cells[cellIndex];
			CellCuda *newCell = &newCells[cellIndex];
			CellCuda cellCopy = *oldCell;
			double2 relPos = cellCopy.relPos;
			relPos = { relPos.x*rotMatrix[0][0] + relPos.y*rotMatrix[0][1], relPos.x*rotMatrix[1][0] + relPos.y*rotMatrix[1][1] };
			double2 absPos = { relPos.x + clusterCopy.pos.x, relPos.y + clusterCopy.pos.y };
			mapCorrection_Kernel(absPos, size);
			cellCopy.absPos = absPos;
			cellCopy.cluster = newCluster;
			*newCell = cellCopy;
			setCellToMap_Kernel({ (int)absPos.x, (int)absPos.y }, newCell, data.map2, size);

			oldCell->nextTimestep = newCell;
		}
	}

	__syncthreads();

	if (threadIdx.x < oldNumCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellCuda* newCell = &newCells[cellIndex];
			int numConnections = newCell->numConnections;
			for (int i = 0; i < numConnections; ++i) {
				newCell->connections[i] = newCell->connections[i]->nextTimestep;
			}
		}
	}

	__syncthreads();
}


__global__ void movement_Kernel(CudaData data)
{
	int blockIndex = blockIdx.x;
	int numClusters = data.clustersAC1.getNumEntries_Kernel();
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

__device__ void clearOldMap_Kernel(CudaData const &data, int clusterIndex)
{
	ClusterCuda *oldCluster = &data.clustersAC1.getEntireArray_Kernel()[clusterIndex];

	int startCellIndex;
	int endCellIndex;
	int2 size = data.size;
	int oldNumCells = oldCluster->numCells;
	if (threadIdx.x >= oldNumCells) {
		return;
	}

	tiling_Kernel(oldNumCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);
	for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
		double2 absPos = oldCluster->cells[cellIndex].absPos;
		setCellToMap_Kernel({ (int)absPos.x, (int)absPos.y }, nullptr, data.map1, size);
	}
}

__global__ void clearOldMap_Kernel(CudaData data)
{
	int blockIndex = blockIdx.x;
	int numClusters = data.clustersAC1.getNumEntries_Kernel();
	if (blockIndex >= numClusters) {
		return;
	}
	int startClusterIndex;
	int endClusterIndex;
	tiling_Kernel(numClusters, blockIndex, gridDim.x, startClusterIndex, endClusterIndex);

	for (int clusterIndex = startClusterIndex; clusterIndex <= endClusterIndex; ++clusterIndex) {
		clearOldMap_Kernel(data, clusterIndex);
	}
}
