#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaShared.cuh"
#include "CudaConstants.cuh"
#include "CudaBase.cuh"
#include "CudaCollision.cuh"
#include "CudaMap.cuh"

__device__ void createNewParticle(CudaSimulation &data, float2 &pos)
{
/*
	auto particle = data.particlesAC2.getElement_Kernel();
	particle->pos = pos;
	particle->vel = { data.randomGen.random()-0.5f, data.randomGen.random() - 0.5f };
*/
}

__device__ void cellRadiation(CudaSimulation &data, CudaCell *oldCell)
{
	if (data.randomGen.random() < 0.001) {
		createNewParticle(data, oldCell->absPos);
	}
}

__device__ void clusterMovement(CudaSimulation &data, int clusterIndex)
{
	__shared__ CudaCellCluster clusterCopy;
	__shared__ CudaCellCluster *newCluster;
	__shared__ CudaCell *newCells;
	__shared__ float rotMatrix[2][2];
	__shared__ CollisionData collisionData;

	CudaCellCluster *oldCluster = &data.clustersAC1.getEntireArray()[clusterIndex];
	int startCellIndex;
	int endCellIndex;
	int2 size = data.size;
	int oldNumCells = oldCluster->numCells;
	if (threadIdx.x < oldNumCells) {

		calcPartition(oldCluster->numCells, threadIdx.x, blockDim.x, startCellIndex, endCellIndex);

		if (threadIdx.x == 0) {
			clusterCopy = *oldCluster;
			float sinAngle = __sinf(clusterCopy.angle*DEG_TO_RAD);
			float cosAngle = __cosf(clusterCopy.angle*DEG_TO_RAD);
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
			CudaCell *oldCell = &oldCluster->cells[cellIndex];

			cellRadiation(data, oldCell);
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
			float numCollisions = static_cast<float>(entry->numCollisions);
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
			CudaCell *oldCell = &oldCluster->cells[cellIndex];
			CudaCell *newCell = &newCells[cellIndex];
			CudaCell cellCopy = *oldCell;
			float2 absPos;
			absPos.x = cellCopy.relPos.x*rotMatrix[0][0] + cellCopy.relPos.y*rotMatrix[0][1] + clusterCopy.pos.x;
			absPos.y = cellCopy.relPos.x*rotMatrix[1][0] + cellCopy.relPos.y*rotMatrix[1][1] + clusterCopy.pos.y;
			cellCopy.absPos = absPos;
			cellCopy.cluster = newCluster;
			if (cellCopy.protectionCounter > 0) {
				--cellCopy.protectionCounter;
			}
			if (cellCopy.setProtectionCounterForNextTimestep) {
				cellCopy.protectionCounter = PROTECTION_TIMESTEPS;
				cellCopy.setProtectionCounterForNextTimestep = false;
			}
			*newCell = cellCopy;
			setCellToMap({ static_cast<int>(absPos.x), static_cast<int>(absPos.y) }, newCell, data.map2, size);

			oldCell->nextTimestep = newCell;
		}
	}

	__syncthreads();

	if (threadIdx.x < oldNumCells) {
		for (int cellIndex = startCellIndex; cellIndex <= endCellIndex; ++cellIndex) {
			CudaCell* newCell = &newCells[cellIndex];
			int numConnections = newCell->numConnections;
			for (int i = 0; i < numConnections; ++i) {
				newCell->connections[i] = newCell->connections[i]->nextTimestep;
			}
		}
	}

	__syncthreads();
}

__global__ void clusterMovement(CudaSimulation data)
{
	int indexResource = blockIdx.x;
	int numEntities = data.clustersAC1.getNumEntries();
	if (indexResource >= numEntities) {
		return;
	}

	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, gridDim.x, startIndex, endIndex);
	for (int clusterIndex = startIndex; clusterIndex <= endIndex; ++clusterIndex) {
		clusterMovement(data, clusterIndex);
	}
}

__device__ void particleMovement(CudaSimulation &data, int particleIndex)
{
	CudaEnergyParticle *oldParticle = &data.particlesAC1.getEntireArray()[particleIndex];
	CudaEnergyParticle *newParticle = data.particlesAC2.getElement_Kernel();
	newParticle->pos = add(oldParticle->pos, oldParticle->vel);
	mapPosCorrection(newParticle->pos, data.size);
	newParticle->vel = oldParticle->vel;
}

__global__ void particleMovement(CudaSimulation data)
{
	int indexResource = threadIdx.x + blockIdx.x * blockDim.x;
	int numEntities = data.particlesAC1.getNumEntries();
	if (indexResource >= numEntities) {
		return;
	}
	int startIndex;
	int endIndex;
	calcPartition(numEntities, indexResource, blockDim.x * gridDim.x, startIndex, endIndex);
	for (int particleIndex = startIndex; particleIndex <= endIndex; ++particleIndex) {
		particleMovement(data, particleIndex);
	}
}