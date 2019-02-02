#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include "CudaInterface.cuh"
#include "Base.cuh"
#include "CudaConstants.cuh"
#include "CudaSimulationParameters.cuh"
#include "Map.cuh"
#include "SimulationDataInternal.cuh"
#include "CollisionData.cuh"


class CudaPhysics
{
public:
	__device__ __inline__ static void calcCollision(ClusterData *clusterA, CollisionEntry *collisionEntry, BasicMap const& map);
	__device__ __inline__ static void getCollisionDataForCell(Map<CellData> const &map, CellData *cell
		, CollisionData &collisionData);

	__inline__ __device__ static void rotationMatrix(float angle, float(&rotMatrix)[2][2]);
	__inline__ __device__ static void angleCorrection(float &angle);
	__inline__ __device__ static float2 tangentialVelocity(float2 const& positionFromCenter, float2 const& velocityOfCenter, float angularVel);
	__inline__ __device__ static float angularMomentum(float2 const& positionFromCenter, float2 const& velocityOfCenter);
	__inline__ __device__ static float angularVelocity(float angularMomentum, float angularMass);
	__device__ __inline__ static void rotateQuarterCounterClockwise(float2 &v);
	__device__ __inline__ static float2 calcNormalToCell(CellData *cell, float2 outward);
	__device__ __inline__ static void calcCollision(float2 const& vA1, float2 const& vB1, float2 const& rAPp, float2 const& rBPp,
		float angularVelA1, float angularVelB1, float2 const& n, float angularMassA, float angularMassB, float massA, float massB,
		float2& vA2, float2& vB2, float& angularVelA2, float& angularVelB2);

private:
	__device__ __inline__ static float2 calcOutwardVector(CellData* cellA, CellData* cellB, BasicMap const& map);
	__device__ __inline__ static void updateCollisionData(float2 pos, CellData *cell, Map<CellData> const& cellMap
		, CollisionData &collisionData);
	__inline__ __device__ static void angleCorrection(int &angle);
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

//math constants
#define PI 3.1415926535897932384626433832795
#define DEG_TO_RAD PI/180.0
#define RAD_TO_DEG 180.0/PI

__device__ __inline__ void CudaPhysics::rotateQuarterCounterClockwise(float2 &v)
{
	float temp = v.x;
	v.x = v.y;
	v.y = -temp;
}

__device__ __inline__ float2 CudaPhysics::calcNormalToCell(CellData *cell, float2 outward)
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

__device__ __inline__ void CudaPhysics::calcCollision(float2 const & vA1, float2 const & vB1, float2 const & rAPp, 
	float2 const & rBPp, float angularVelA1, float angularVelB1, float2 const & n, float angularMassA, 
	float angularMassB, float massA, float massB, float2 & vA2, float2 & vB2, float & angularVelA2, float & angularVelB2)
{
	float2 vAB = sub(sub(vA1, mul(rAPp, angularVelA1 * DEG_TO_RAD)), sub(vB1, mul(rBPp, angularVelB1 * DEG_TO_RAD)));

	float vAB_dot_n = dot(vAB, n);
	if (vAB_dot_n > 0.0) {
		vA2 = vA1;
		angularVelA2 = angularVelA1;
		vB2 = vB1;
		angularVelB2 = angularVelB1;
		return;
	}

	float rAPp_dot_n = dot(rAPp, n);
	float rBPp_dot_n = dot(rBPp, n);

	if (angularMassA > FP_PRECISION && angularMassB > FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			+ rAPp_dot_n * rAPp_dot_n / angularMassA + rBPp_dot_n * rBPp_dot_n / angularMassB);

		vA2 = add(vA1, mul(n, j / massA));
		angularVelA2 = angularVelA1 - (rAPp_dot_n * j / angularMassA) * RAD_TO_DEG;
		vB2 = sub(vB1, mul(n, j / massB));
		angularVelB2 = angularVelB1 + (rBPp_dot_n * j / angularMassB) * RAD_TO_DEG;
	}

	if (angularMassA <= FP_PRECISION && angularMassB > FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			+ rBPp_dot_n * rBPp_dot_n / angularMassB);

		vA2 = add(vA1, mul(n, j / massA));
		angularVelA2 = angularVelA1;
		vB2 = sub(vB1, mul(n, j / massB));
		angularVelB2 = angularVelB1 + (rBPp_dot_n * j / angularMassB) * RAD_TO_DEG;
	}

	if (angularMassA > FP_PRECISION && angularMassB <= FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			+ rAPp_dot_n * rAPp_dot_n / angularMassA);

		vA2 = add(vA1, mul(n, j / massA));
		angularVelA2 = angularVelA1 - (rAPp_dot_n * j / angularMassA) * RAD_TO_DEG;
		vB2 = sub(vB1, mul(n, j / massB));
		angularVelB2 = angularVelB1;
	}

	if (angularMassA <= FP_PRECISION && angularMassB <= FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB));

		vA2 = add(vA1, mul(n, j / massA));
		angularVelA2 = angularVelA1;
		vB2 = sub(vB1, mul(n, j / massB));
		angularVelB2 = angularVelB1;
	}

}

__device__ __inline__ void CudaPhysics::calcCollision(ClusterData *clusterA, CollisionEntry *collisionEntry, BasicMap const& map)
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

__device__ __inline__ float2 CudaPhysics::calcOutwardVector(CellData* cellA, CellData* cellB, BasicMap const& map)
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

__device__ __inline__ void CudaPhysics::updateCollisionData(float2 pos, CellData *cell
	, Map<CellData> const& cellMap, CollisionData &collisionData)
{
	CellData* mapCell = cellMap.getFromOrigMap(pos);
	if (!mapCell) {
		return;
	}
	ClusterData* mapCluster = mapCell->cluster;
	if (mapCluster == cell->cluster) {
		return;
	}
	if (mapCell->protectionCounter > 0) {
		return;
	}
	auto distanceSquared = cellMap.mapDistanceSquared(cell->absPos, mapCell->absPos);
	if (distanceSquared > cudaSimulationParameters.cellMaxDistance * cudaSimulationParameters.cellMaxDistance) {
		return;
	}

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

__device__ __inline__ void CudaPhysics::getCollisionDataForCell(Map<CellData> const &map, CellData *cell, CollisionData &collisionData)
{
	if (cell->protectionCounter > 0) {
		return;
	}

	auto absPos = cell->absPos;
	if (!map.isEntityPresentAtOrigMap(absPos, cell)) {
		return;
	}

	--absPos.x;
	--absPos.y;
	updateCollisionData(absPos, cell, map, collisionData);
	++absPos.x;
	updateCollisionData(absPos, cell, map, collisionData);
	++absPos.x;
	updateCollisionData(absPos, cell, map, collisionData);

	++absPos.y;
	absPos.x -= 2;
	updateCollisionData(absPos, cell, map, collisionData);
	absPos.x += 2;
	updateCollisionData(absPos, cell, map, collisionData);

	++absPos.y;
	absPos.x -= 2;
	updateCollisionData(absPos, cell, map, collisionData);
	++absPos.x;
	updateCollisionData(absPos, cell, map, collisionData);
	++absPos.x;
	updateCollisionData(absPos, cell, map, collisionData);
}

__inline__ __device__ void CudaPhysics::rotationMatrix(float angle, float(&rotMatrix)[2][2])
{
	float sinAngle = __sinf(angle*DEG_TO_RAD);
	float cosAngle = __cosf(angle*DEG_TO_RAD);
	rotMatrix[0][0] = cosAngle;
	rotMatrix[0][1] = -sinAngle;
	rotMatrix[1][0] = sinAngle;
	rotMatrix[1][1] = cosAngle;
}

__inline__ __device__ void CudaPhysics::angleCorrection(int &angle)
{
	angle = ((angle % 360) + 360) % 360;
}

__inline__ __device__ void CudaPhysics::angleCorrection(float &angle)
{
	int intPart = (int)angle;
	float fracPart = angle - intPart;
	angleCorrection(intPart);
	angle = (float)intPart + fracPart;
}

__inline__ __device__ float2 CudaPhysics::tangentialVelocity(float2 const& r, float2 const& vel, float angularVel)
{
	return { vel.x - angularVel*r.y * static_cast<float>(DEG_TO_RAD), vel.y + angularVel*r.x * static_cast<float>(DEG_TO_RAD) };
}

__inline__ __device__ float CudaPhysics::angularMomentum(float2 const & positionFromCenter, float2 const & velocityOfCenter)
{
	return positionFromCenter.x*velocityOfCenter.y - positionFromCenter.y*velocityOfCenter.x;
}

__inline__ __device__ float CudaPhysics::angularVelocity(float angularMomentum, float angularMass)
{
	if (std::abs(angularMass) < FP_PRECISION)
		return 0;
	else
		return angularMomentum / angularMass * RAD_TO_DEG;
}
