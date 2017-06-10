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
		mapPosCorrection_Kernel(clusterCopy.pos, size);
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
			mapPosCorrection_Kernel(absPos, size);
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
