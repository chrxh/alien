#pragma once

#include "CudaShared.cuh"
#include "CudaBase.cuh"
#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaConstants.cuh"

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

__device__ CellCuda* readCellMapHelper_Kernel(int2 posInt, ClusterCuda *cluster, CellCuda ** __restrict__ map, int2 const &size)
{
	mapCorrection_Kernel(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	auto slice = size.x*size.y;

	for (int i = 0; i < LAYERS; ++i) {
		auto cell = map[mapEntry + i * slice];
		if (cell != nullptr) {
			if (cell->cluster != cluster) {
				return cell;
			}
		}
	}
	return nullptr;
}

__device__ CellCuda* readCellFromMap_Kernel(int2 posInt, ClusterCuda *cluster, CellCuda ** __restrict__ map, int2 const &size)
{
	--posInt.x;
	--posInt.y;
	auto cell = readCellMapHelper_Kernel(posInt, cluster, map, size);
	if (cell) { return cell; }

	++posInt.x;
	cell = readCellMapHelper_Kernel(posInt, cluster, map, size);
	if (cell) { return cell; }

	++posInt.x;
	cell = readCellMapHelper_Kernel(posInt, cluster, map, size);
	if (cell) { return cell; }

	++posInt.y;
	posInt.x -= 2;
	cell = readCellMapHelper_Kernel(posInt, cluster, map, size);
	if (cell) { return cell; }

	++posInt.y;
	cell = readCellMapHelper_Kernel(posInt, cluster, map, size);
	if (cell) { return cell; }

	++posInt.y;
	cell = readCellMapHelper_Kernel(posInt, cluster, map, size);
	if (cell) { return cell; }

	++posInt.y;
	posInt.x -= 2;
	cell = readCellMapHelper_Kernel(posInt, cluster, map, size);
	if (cell) { return cell; }

	++posInt.y;
	cell = readCellMapHelper_Kernel(posInt, cluster, map, size);
	if (cell) { return cell; }

	++posInt.y;
	cell = readCellMapHelper_Kernel(posInt, cluster, map, size);
	if (cell) { return cell; }
	return nullptr;
}

__device__ void setCellToMap_Kernel(int2 const &posInt, CellCuda *cell, CellCuda ** __restrict__ map, int2 const &size)
{
	auto mapEntry = posInt.x + posInt.y * size.x;
	map[mapEntry] = cell;
}

__device__ void rotateQuarterCounterClockwise(double2 &v)
{
	double temp = v.x;
	v.x = v.y;
	v.y = -temp;
}

__device__ void calcCollision(ClusterCuda *cluster, CellCuda *collidingCell, double2 &resultVel, double &resultAngularVel
	, double2 &resultCollidingVel, double &resultCollidingAngularVel)
{
	double2 pos = cluster->pos;
	double2 vel = cluster->vel;
	double angularVel = cluster->angularVel * DEG_TO_RAD;
	double angularMass = 1.0;

	ClusterCuda *collidingCluster = collidingCell->cluster;
	double2 collidingPos = collidingCluster->pos;
	double2 collidingVel = collidingCluster->vel;
	double collidingAngularVel = collidingCluster->angularVel * DEG_TO_RAD;
	double collidingAngularMass = 1.0;

	double2 rAPp = { collidingCell->absPos.x - pos.x, collidingCell->absPos.y - pos.y };
	rotateQuarterCounterClockwise(rAPp);
	double2 rBPp = { collidingCell->absPos.x - collidingPos.x, collidingCell->absPos.y - collidingPos.y };
	rotateQuarterCounterClockwise(rBPp);
	double2 vAB = {
		-(collidingVel.x - rBPp.x*collidingAngularVel) + (vel.x - rAPp.x*angularVel)
		, -(collidingVel.y - rBPp.y*collidingAngularVel) + (vel.y - rAPp.y*angularVel)
	};

	double scale = sqrt(vAB.x*vAB.x + vAB.y*vAB.y);
	double2 n;
	if (scale < FP_PRECISION) {
		n.x = 1.0;
	}
	else {
		n = { -vAB.x / scale, -vAB.y / scale };
	}

	if (angularMass < FP_PRECISION) {
		angularVel = 0.0;
	}
	if (collidingAngularMass < FP_PRECISION) {
		collidingAngularVel = 0.0;
	}

	if (vAB.x * n.x + vAB.y * n.y > 0.0) {
		resultVel = vel;
		resultAngularVel = angularVel;
		resultCollidingVel = collidingVel;
		resultCollidingAngularVel = collidingAngularVel;
	}

}

__device__ void movement_Kernel(CudaData &data, int clusterIndex)
{
	
	ClusterCuda *oldCluster = &data.clustersAC1.getEntireArrayKernel()[clusterIndex];
	__shared__ ClusterCuda clusterCopy;
	__shared__ ClusterCuda *newCluster;
	__shared__ CellCuda *newCells;
	__shared__ double rotMatrix[2][2];

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
			newCells = data.cellsAC2.getArrayKernel(clusterCopy.numCells);
			newCluster = data.clustersAC2.getElementKernel();

			clusterCopy.angle += clusterCopy.angularVel;
			angleCorrection_Kernel(clusterCopy.angle);
			clusterCopy.pos = { clusterCopy.pos.x + clusterCopy.vel.x, clusterCopy.pos.y + clusterCopy.vel.y };
			mapCorrection_Kernel(clusterCopy.pos, size);
			clusterCopy.cells = newCells;

			*newCluster = clusterCopy;
		}
	}
	
	__syncthreads();

	if (threadIdx.x < oldNumCells) {

		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			double2 absPos = clusterCopy.cells[cellIndex].absPos;
			int2 absPosInt = { (int)absPos.x , (int)absPos.y };

			CellCuda* collidingCell = readCellFromMap_Kernel(absPosInt, oldCluster, data.map1, size);
			//		calcCollision(clusterPtr, collidingCell);
		}
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

__device__ void clearOldMap_Kernel(CudaData const &data, int clusterIndex)
{
	ClusterCuda *oldCluster = &data.clustersAC1.getEntireArrayKernel()[clusterIndex];

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
	int numClusters = data.clustersAC1.getNumEntriesKernel();
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
