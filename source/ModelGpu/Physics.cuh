#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include "CudaInterface.cuh"
#include "Base.cuh"
#include "Constants.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"

struct CollisionEntry
{
	ClusterData* cluster;
	int numCollisions;
	float2 collisionPos;
	float2 normalVec;
};

class CollisionData
{
public:
	
	int numEntries;
	CollisionEntry entries[MAX_COLLIDING_CLUSTERS];

	__device__ void init()
	{
		numEntries = 0;
	}

	__device__ CollisionEntry* getOrCreateEntry(ClusterData* cluster)
	{
		int old;
		int curr;
		do {
			for (int i = 0; i < numEntries; ++i) {
				if (entries[i].cluster = cluster) {
					return &entries[i];
				}
			}
			
			old = numEntries;
			curr = atomicCAS(&numEntries, old, (old + 1) % MAX_COLLIDING_CLUSTERS);
			if (old == curr) {
				auto result = &entries[old];
				result->cluster = cluster;
				result->numCollisions = 0;
				result->collisionPos = { 0, 0 };
				result->normalVec = { 0, 0 };
				return result;
			}

		} while (true);
	}
};

__device__ __inline__ void rotateQuarterCounterClockwise_Kernel(float2 &v)
{
	float temp = v.x;
	v.x = v.y;
	v.y = -temp;
}

__device__ __inline__ float2 calcNormalToCell_Kernel(CellData *cell, float2 outward)
{
	normalize(outward);
	if (cell->numConnections < 2) {
		return outward;
	}

	CellData* minCell = nullptr;
	CellData* maxCell = nullptr;
	float2 minVector;
	float2 maxVector;
	float min_h = 0.0;	//h = angular distance from outward vector
	float max_h = 0.0;

	for (int i = 0; i < cell->numConnections; ++i) {

		//calculate h (angular distance from outward vector)
		float2 u = sub(cell->connections[i]->absPos, cell->absPos);
		normalize(u);
		float h = dot(outward, u);
		if (outward.x*u.y - outward.y*u.x < 0.0) {
			h = -2 - h;
		}

		if (!minCell || h < min_h) {
			minCell = cell->connections[i];
			minVector = u;
			min_h = h;
		}
		if (!maxCell || h > max_h) {
			maxCell = cell->connections[i];
			maxVector = u;
			max_h = h;
		}
	}

	//no adjacent cells?
	if (!minCell && !maxCell) {
		return outward;
	}

	//one adjacent cells?
	if (minCell == maxCell) {
		return sub(cell->absPos, minCell->absPos);
	}

	//calc normal vectors
	float temp = maxVector.x;
	maxVector.x = maxVector.y;
	maxVector.y = -temp;
	temp = minVector.x;
	minVector.x = -minVector.y;
	minVector.y = temp;
	auto result = add(minVector, maxVector);
	normalize(result);
	return result;
}
__device__ __inline__ void calcCollision(ClusterData *clusterA, CollisionEntry *collisionEntry, int2 const& size)
{
	float2 posA = clusterA->pos;
	float2 velA = clusterA->vel;
	float angVelA = clusterA->angularVel * DEG_TO_RAD;
	float angMassA = clusterA->angularMass;

	ClusterData *clusterB = collisionEntry->cluster;
	float2 posB = clusterB->pos;
	float2 velB = clusterB->vel;
	float angVelB = clusterB->angularVel * DEG_TO_RAD;
	float angMassB = clusterB->angularMass;

	float2 rAPp = sub(collisionEntry->collisionPos, posA);
	mapDisplacementCorrection(rAPp, size);
	rotateQuarterCounterClockwise_Kernel(rAPp);
	float2 rBPp = sub(collisionEntry->collisionPos, posB);
	mapDisplacementCorrection(rBPp, size);
	rotateQuarterCounterClockwise_Kernel(rBPp);
	float2 vAB = sub(sub(velA, mul(rAPp, angVelA)), sub(velB, mul(rBPp, angVelB)));

	float2 n = collisionEntry->normalVec;

	if (angMassA < FP_PRECISION) {
		angVelA = 0.0;
	}
	if (angMassB < FP_PRECISION) {
		angVelB = 0.0;
	}

	float vAB_dot_n = dot(vAB, n);
	if (vAB_dot_n > 0.0) {
		return;
	}

	float massA = static_cast<float>(clusterA->numCells);
	float massB = static_cast<float>(clusterB->numCells);
	float rAPp_dot_n = dot(rAPp, n);
	float rBPp_dot_n = dot(rBPp, n);

	if (angMassA > FP_PRECISION && angMassB > FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0/massA + 1.0/massB)
			+ rAPp_dot_n * rAPp_dot_n / angMassA + rBPp_dot_n * rBPp_dot_n / angMassB);


		atomicAdd(&clusterA->angularVel, -(rAPp_dot_n * j / angMassA) * RAD_TO_DEG);
		atomicAdd(&clusterA->vel.x, j / massA * n.x);
		atomicAdd(&clusterA->vel.y, j / massA * n.y);

/*
			atomicAdd(&collisionData.angularVelDelta, (rBPp_dot_n * j / angMassB) * RAD_TO_DEG);
			atomicAdd(&collisionData.velDelta.x, -j / massB * n.x);
			atomicAdd(&collisionData.velDelta.y, -j / massB * n.y);
*/
	}

	if (angMassA <= FP_PRECISION && angMassB > FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			 + rBPp_dot_n * rBPp_dot_n / angMassB);

			atomicAdd(&clusterA->vel.x, j / massA * n.x);
			atomicAdd(&clusterA->vel.y, j / massA * n.y);
/*
			atomicAdd(&collisionData.angularVelDelta, (rBPp_dot_n * j / angMassB) * RAD_TO_DEG);
			atomicAdd(&collisionData.velDelta.x, -j / massB * n.x);
			atomicAdd(&collisionData.velDelta.y, -j / massB * n.y);
*/
	}

	if (angMassA > FP_PRECISION && angMassB <= FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			+ rAPp_dot_n * rAPp_dot_n / angMassA);

		atomicAdd(&clusterA->angularVel, -(rAPp_dot_n * j / angMassA) * RAD_TO_DEG);
		atomicAdd(&clusterA->vel.x, j / massA * n.x);
		atomicAdd(&clusterA->vel.y, j / massA * n.y);
/*
			atomicAdd(&collisionData.velDelta.x, -j / massB * n.x);
			atomicAdd(&collisionData.velDelta.y, -j / massB * n.y);
*/
	}

	if (angMassA <= FP_PRECISION && angMassB <= FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB));

		atomicAdd(&clusterA->vel.x, j / massA * n.x);
		atomicAdd(&clusterA->vel.y, j / massA * n.y);
/*
			atomicAdd(&collisionData.velDelta.x, -j / massB * n.x);
			atomicAdd(&collisionData.velDelta.y, -j / massB * n.y);
*/
	}

}

__device__ __inline__ float2 calcOutwardVector_Kernel(CellData* cellA, CellData* cellB, int2 const &size)
{
	ClusterData* clusterA = cellA->cluster;
	float2 posA = clusterA->pos;
	float2 velA = clusterA->vel;
	float angVelA = clusterA->angularVel * DEG_TO_RAD;

	ClusterData* clusterB = cellB->cluster;
	float2 posB = clusterB->pos;
	float2 velB = clusterB->vel;
	float angVelB = clusterB->angularVel * DEG_TO_RAD;

	float2 rAPp = sub(cellB->absPos, posA);
	mapDisplacementCorrection(rAPp, size);
	rotateQuarterCounterClockwise_Kernel(rAPp);
	float2 rBPp = sub(cellB->absPos, posB);
	mapDisplacementCorrection(rBPp, size);
	rotateQuarterCounterClockwise_Kernel(rBPp);
	return sub(sub(velB, mul(rBPp, angVelB)), sub(velA, mul(rAPp, angVelA)));
}

__device__ __inline__ void updateCollisionData_Kernel(int2 posInt, CellData *cell, CellData ** map
	, int2 const &size, CollisionData &collisionData)
{
	auto mapCell = getFromMap(posInt, map, size);
	if (!mapCell) {
		return;
	}
	if (mapCell->protectionCounter > 0) {
		return;
	}
	ClusterData* mapCluster = mapCell->cluster;
	if (mapCluster != cell->cluster) {
		if (mapDistanceSquared(cell->absPos, mapCell->absPos, size) < CELL_MAX_DISTANCE*CELL_MAX_DISTANCE) {

			CollisionEntry* entry = collisionData.getOrCreateEntry(mapCluster);

			atomicAdd(&entry->numCollisions, 2);
			atomicAdd(&entry->collisionPos.x, mapCell->absPos.x);
			atomicAdd(&entry->collisionPos.y, mapCell->absPos.y);
			atomicAdd(&entry->collisionPos.x, cell->absPos.x);
			atomicAdd(&entry->collisionPos.y, cell->absPos.y);

			float2 outward = calcOutwardVector_Kernel(cell, mapCell, size);
			float2 n = calcNormalToCell_Kernel(mapCell, outward);
			atomicAdd(&entry->normalVec.x, n.x);
			atomicAdd(&entry->normalVec.y, n.y);

			outward = calcOutwardVector_Kernel(mapCell, cell, size);
			n = minus(calcNormalToCell_Kernel(cell, outward));
			atomicAdd(&entry->normalVec.x, n.x);
			atomicAdd(&entry->normalVec.y, n.y);

//			cell->protectionCounter = PROTECTION_TIMESTEPS;
			cell->setProtectionCounterForNextTimestep = true;
		}
	}
}

__device__ __inline__ void collectCollisionData(SimulationData const &data, CellData *cell, CollisionData &collisionData)
{
	if (cell->protectionCounter > 0) {
		return;
	}

	auto absPos = cell->absPos;
	auto map = data.cellMap1;
	auto size = data.size;
	int2 posInt = { static_cast<int>(absPos.x) , static_cast<int>(absPos.y) };
	if (!isCellPresentAtMap(posInt, cell, map, size)) {
		return;
	}

	--posInt.x;
	--posInt.y;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);
	++posInt.x;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);
	++posInt.x;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);

	++posInt.y;
	posInt.x -= 2;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);
	posInt.x += 2;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);

	++posInt.y;
	posInt.x -= 2;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);
	++posInt.x;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);
	++posInt.x;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);
}
