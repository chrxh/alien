#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include "CudaInterface.cuh"
#include "Base.cuh"
#include "Constants.cuh"
#include "Map.cuh"
#include "SimulationDataInternal.cuh"
#include "CollisionData.cuh"


class Physics
{
public:
	__device__ __inline__ static void calcCollision(ClusterData *clusterA, CollisionEntry *collisionEntry, BasicMap const& map);
	__device__ __inline__ static void getCollisionDataForCell(Map<CellData> const &map, CellData *cell
		, CollisionData &collisionData);
	__inline__ __device__ static void calcRotationMatrix(float angle, float(&rotMatrix)[2][2]);
	__inline__ __device__ static void angleCorrection(float &angle);

private:
	__device__ __inline__ static void rotateQuarterCounterClockwise(float2 &v);
	__device__ __inline__ static float2 calcNormalToCell(CellData *cell, float2 outward);
	__device__ __inline__ static float2 calcOutwardVector(CellData* cellA, CellData* cellB, BasicMap const& map);
	__device__ __inline__ static void updateCollisionData(int2 posInt, CellData *cell, Map<CellData> const& cellMap
		, CollisionData &collisionData);
	__inline__ __device__ static void angleCorrection(int &angle);
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__device__ __inline__ void Physics::rotateQuarterCounterClockwise(float2 &v)
{
	float temp = v.x;
	v.x = v.y;
	v.y = -temp;
}

__device__ __inline__ float2 Physics::calcNormalToCell(CellData *cell, float2 outward)
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

__device__ __inline__ void Physics::calcCollision(ClusterData *clusterA, CollisionEntry *collisionEntry, BasicMap const& map)
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
	map.mapDisplacementCorrection(rAPp);
	rotateQuarterCounterClockwise(rAPp);
	float2 rBPp = sub(collisionEntry->collisionPos, posB);
	map.mapDisplacementCorrection(rBPp);
	rotateQuarterCounterClockwise(rBPp);
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
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
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

__device__ __inline__ float2 Physics::calcOutwardVector(CellData* cellA, CellData* cellB, BasicMap const& map)
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
	map.mapDisplacementCorrection(rAPp);
	rotateQuarterCounterClockwise(rAPp);
	float2 rBPp = sub(cellB->absPos, posB);
	map.mapDisplacementCorrection(rBPp);
	rotateQuarterCounterClockwise(rBPp);
	return sub(sub(velB, mul(rBPp, angVelB)), sub(velA, mul(rAPp, angVelA)));
}

__device__ __inline__ void Physics::updateCollisionData(int2 posInt, CellData *cell
	, Map<CellData> const& cellMap, CollisionData &collisionData)
{
	auto mapCell = cellMap.getFromOrigMap(posInt);
	if (!mapCell) {
		return;
	}
	if (mapCell->protectionCounter > 0) {
		return;
	}
	ClusterData* mapCluster = mapCell->cluster;
	if (mapCluster != cell->cluster) {
		if (cellMap.mapDistanceSquared(cell->absPos, mapCell->absPos) < CELL_MAX_DISTANCE*CELL_MAX_DISTANCE) {

			CollisionEntry* entry = collisionData.getOrCreateEntry(mapCluster);

			atomicAdd(&entry->numCollisions, 2);
			atomicAdd(&entry->collisionPos.x, mapCell->absPos.x);
			atomicAdd(&entry->collisionPos.y, mapCell->absPos.y);
			atomicAdd(&entry->collisionPos.x, cell->absPos.x);
			atomicAdd(&entry->collisionPos.y, cell->absPos.y);

			float2 outward = calcOutwardVector(cell, mapCell, cellMap);
			float2 n = calcNormalToCell(mapCell, outward);
			atomicAdd(&entry->normalVec.x, n.x);
			atomicAdd(&entry->normalVec.y, n.y);

			outward = calcOutwardVector(mapCell, cell, cellMap);
			n = minus(calcNormalToCell(cell, outward));
			atomicAdd(&entry->normalVec.x, n.x);
			atomicAdd(&entry->normalVec.y, n.y);

			cell->protectionCounter = PROTECTION_TIMESTEPS;
		}
	}
}

__device__ __inline__ void Physics::getCollisionDataForCell(Map<CellData> const &map, CellData *cell, CollisionData &collisionData)
{
	if (cell->protectionCounter > 0) {
		return;
	}

	auto absPos = cell->absPos;
	int2 posInt = { static_cast<int>(absPos.x) , static_cast<int>(absPos.y) };
	if (!map.isEntityPresentAtOrigMap(posInt, cell)) {
		return;
	}

	--posInt.x;
	--posInt.y;
	updateCollisionData(posInt, cell, map, collisionData);
	++posInt.x;
	updateCollisionData(posInt, cell, map, collisionData);
	++posInt.x;
	updateCollisionData(posInt, cell, map, collisionData);

	++posInt.y;
	posInt.x -= 2;
	updateCollisionData(posInt, cell, map, collisionData);
	posInt.x += 2;
	updateCollisionData(posInt, cell, map, collisionData);

	++posInt.y;
	posInt.x -= 2;
	updateCollisionData(posInt, cell, map, collisionData);
	++posInt.x;
	updateCollisionData(posInt, cell, map, collisionData);
	++posInt.x;
	updateCollisionData(posInt, cell, map, collisionData);
}

__inline__ __device__ void Physics::calcRotationMatrix(float angle, float(&rotMatrix)[2][2])
{
	float sinAngle = __sinf(angle*DEG_TO_RAD);
	float cosAngle = __cosf(angle*DEG_TO_RAD);
	rotMatrix[0][0] = cosAngle;
	rotMatrix[0][1] = -sinAngle;
	rotMatrix[1][0] = sinAngle;
	rotMatrix[1][1] = cosAngle;
}

__inline__ __device__ void Physics::angleCorrection(int &angle)
{
	angle = ((angle % 360) + 360) % 360;
}

__inline__ __device__ void Physics::angleCorrection(float &angle)
{
	int intPart = (int)angle;
	float fracPart = angle - intPart;
	angleCorrection(intPart);
	angle = (float)intPart + fracPart;
}
