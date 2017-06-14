#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaShared.cuh"
#include "CudaConstants.cuh"
#include "CudaBase.cuh"
#include "CudaCollision.cuh"
#include "CudaMap.cuh"

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

__device__ void inline movement_Kernel(CudaData &data, int clusterIndex)
{
	__shared__ ClusterCuda clusterCopy;
	__shared__ ClusterCuda *newCluster;
	__shared__ CellCuda *newCells;
	__shared__ double rotMatrix[2][2];
	__shared__ CollisionData collisionData;

	ClusterCuda *oldCluster = &data.clustersAC1.getEntireArray()[clusterIndex];
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
			rotMatrix[0][1] = -sinAngle;
			rotMatrix[1][0] = sinAngle;
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

/*
			printf("%d, %d; pos: %f, %f \n", clusterIndex, collisionData.numCollisions, clusterCopy.pos.x, clusterCopy.pos.y);
*/
		for (int i = 0; i < collisionData.numEntries; ++i) {
			CollisionEntry* entry = &collisionData.entries[i];
			double numCollisions = static_cast<double>(entry->numCollisions);
			entry->collisionPos.x /= numCollisions;
			entry->collisionPos.y /= numCollisions;
			entry->normalVec.x /= numCollisions;
			entry->normalVec.y /= numCollisions;
			calcCollision_Kernel(&clusterCopy, entry, size);
		}


		clusterCopy.angle += clusterCopy.angularVel;
		angleCorrection_Kernel(clusterCopy.angle);
		clusterCopy.pos = add(clusterCopy.pos, clusterCopy.vel);
		mapPosCorrection(clusterCopy.pos, size);
		clusterCopy.cells = newCells;
		*newCluster = clusterCopy;
	}

	__syncthreads();

	if (threadIdx.x < oldNumCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CellCuda *oldCell = &oldCluster->cells[cellIndex];
			CellCuda *newCell = &newCells[cellIndex];
			CellCuda cellCopy = *oldCell;
			double2 absPos;
			absPos.x = cellCopy.relPos.x*rotMatrix[0][0] + cellCopy.relPos.y*rotMatrix[0][1] + clusterCopy.pos.x;
			absPos.y = cellCopy.relPos.x*rotMatrix[1][0] + cellCopy.relPos.y*rotMatrix[1][1] + clusterCopy.pos.y;
			cellCopy.absPos = absPos;
			cellCopy.cluster = newCluster;
			*newCell = cellCopy;
			setCellToMap({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, newCell, data.map2, size);

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
	int numClusters = data.clustersAC1.getNumEntries();
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
	ClusterCuda *oldCluster = &data.clustersAC1.getEntireArray()[clusterIndex];

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
		setCellToMap({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, nullptr, data.map1, size);
	}
}

__global__ void clearOldMap_Kernel(CudaData data)
{
	int blockIndex = blockIdx.x;
	int numClusters = data.clustersAC1.getNumEntries();
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
