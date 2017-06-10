#pragma once

#include <cuda_runtime.h>

#include "CudaShared.cuh"
#include "CudaBase.cuh"
#include "CudaConstants.cuh"
#include "CudaMap.cuh"
#include "CudaSimulationData.cuh"

struct CollisionData
{
	int numCollisions;
	double2 velDelta;
	double angularVelDelta;

	__device__ inline void init_Kernel()
	{
		numCollisions = 0;
		velDelta = { 0.0, 0.0 };
		angularVelDelta = 0.0;
	}
};

__device__ inline void rotateQuarterCounterClockwise_Kernel(double2 &v)
{
	double temp = v.x;
	v.x = v.y;
	v.y = -temp;
}

__device__ inline double2 calcNormal_Kernel(CellCuda *cell, double2 outward)
{
	return outward;
}
__device__ inline void calcCollision_Kernel(ClusterCuda *cluster, CellCuda *collidingCell, CollisionData &collisionData)
{
	double2 posA = cluster->pos;
	double2 velA = cluster->vel;
	double angVelA = cluster->angularVel * DEG_TO_RAD;
	double angMassA = cluster->angularMass;

	ClusterCuda *collidingCluster = collidingCell->cluster;
	double2 posB = collidingCluster->pos;
	double2 velB = collidingCluster->vel;
	double angVelB = collidingCluster->angularVel * DEG_TO_RAD;
	double angMassB = collidingCluster->angularMass;

	double2 rAPp = { collidingCell->absPos.x - posA.x, collidingCell->absPos.y - posA.y };
	rotateQuarterCounterClockwise_Kernel(rAPp);
	double2 rBPp = { collidingCell->absPos.x - posB.x, collidingCell->absPos.y - posB.y };
	rotateQuarterCounterClockwise_Kernel(rBPp);
	double2 vAB = {
		-(velB.x - rBPp.x*angVelB) + (velA.x - rAPp.x*angVelA)
		, -(velB.y - rBPp.y*angVelB) + (velA.y - rAPp.y*angVelA)
	};

	double2 outward = { -vAB.x, -vAB.y };
	normalizeVector(outward);
	double2 n = calcNormal_Kernel(collidingCell, outward);

	if (angMassA < FP_PRECISION) {
		angVelA = 0.0;
	}
	if (angMassB < FP_PRECISION) {
		angVelB = 0.0;
	}

	double vAB_dot_n = vAB.x * n.x + vAB.y * n.y;
	if (vAB_dot_n > 0.0) {
		return;
	}

	double massA = static_cast<double>(cluster->numCells);
	double massB = static_cast<double>(collidingCluster->numCells);
	double rAPp_dot_n = rAPp.x * n.x + rAPp.y * n.y;
	double rBPp_dot_n = rBPp.x * n.x + rBPp.y * n.y;

	if (angMassA > FP_PRECISION && angMassB > FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((n.x*n.x + n.y*n.y) * (1.0/massA + 1.0/massB)
			+ rAPp_dot_n * rAPp_dot_n / angMassA + rBPp_dot_n * rBPp_dot_n / angMassB);

		atomicAdd(&collisionData.velDelta.x, j / massA * n.x);
		atomicAdd(&collisionData.velDelta.y, j / massA * n.y);
		atomicAdd(&collisionData.angularVelDelta, -rAPp_dot_n * j / angMassA);
	}

	if (angMassA <= FP_PRECISION && angMassB > FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((n.x*n.x + n.y*n.y) * (1.0 / massA + 1.0 / massB)
			 + rBPp_dot_n * rBPp_dot_n / angMassB);

		atomicAdd(&collisionData.velDelta.x, j / massA * n.x);
		atomicAdd(&collisionData.velDelta.y, j / massA * n.y);
	}

	if (angMassA > FP_PRECISION && angMassB <= FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((n.x*n.x + n.y*n.y) * (1.0 / massA + 1.0 / massB)
			+ rAPp_dot_n * rAPp_dot_n / angMassA);

		atomicAdd(&collisionData.velDelta.x, j / massA * n.x);
		atomicAdd(&collisionData.velDelta.y, j / massA * n.y);
		atomicAdd(&collisionData.angularVelDelta, -rAPp_dot_n * j / angMassA);
	}

	if (angMassA <= FP_PRECISION && angMassB <= FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((n.x*n.x + n.y*n.y) * (1.0 / massA + 1.0 / massB));

		atomicAdd(&collisionData.velDelta.x, j / massA * n.x);
		atomicAdd(&collisionData.velDelta.y, j / massA * n.y);
	}

}

__device__ inline void updateCollisionData_Kernel(int2 posInt, CellCuda *cell, CellCuda ** __restrict__ map
	, int2 const &size, CollisionData &collisionData)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	//	auto slice = size.x*size.y;

	//	for (int i = 0; i < LAYERS; ++i) {
	auto mapCell = map[mapEntry/* + i * slice*/];
	if (mapCell != nullptr) {
		if (mapCell->cluster != cell->cluster) {
			if (mapDistanceSquared_Kernel(cell->absPos, mapCell->absPos, size) < CELL_MAX_DISTANCE) {
				atomicAdd(&collisionData.numCollisions, 1);
				calcCollision_Kernel(cell->cluster, mapCell, collisionData);
			}
		}
	}
	//	}
}

__device__ inline void collectCollisionData_Kernel(CudaData const &data, CellCuda *cell, CollisionData &collisionData)
{
	auto absPos = cell->absPos;
	auto map = data.map1;
	auto size = data.size;
	int2 posInt = { (int)absPos.x , (int)absPos.y };
	if (!isCellPresentAtMap_Kernel(posInt, cell, map, size)) {
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
	posInt.y += 2;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);

	++posInt.y;
	posInt.x -= 2;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);
	++posInt.y;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);
	++posInt.y;
	updateCollisionData_Kernel(posInt, cell, map, size, collisionData);
}
