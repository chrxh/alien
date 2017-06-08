#pragma once

#include <cuda_runtime.h>

#include "CudaShared.cuh"
#include "CudaBase.cuh"
#include "CudaConstants.cuh"

struct CollisionEntry
{
	ClusterCuda* collidingCluster;
	double2 velDelta;
	double angularVelDelta;
};

class CollisionData
{
private:
	unsigned int numCollisions;
	CollisionEntry collisionEntries[MAX_COLLIDING_CLUSTERS];

public:
	__device__ void init()
	{
		numCollisions = 0;
	}

	__device__ int getNumCollisions()
	{
		return numCollisions;
	}

	__device__ CollisionEntry* getOrCreateEntry(ClusterCuda* collidingCluster)
	{
		for (int i = 0; i < numCollisions; ++i) {
			if (collisionEntries[i].collidingCluster == collidingCluster) {
				return &collisionEntries[i];
			}
		}
		auto entry = atomicInc(&numCollisions, MAX_COLLIDING_CLUSTERS);
		auto &result = collisionEntries[entry];
		result.collidingCluster = collidingCluster;
		result.velDelta = { 0.0, 0.0 };
		result.angularVelDelta = 0;
		return &result;
	}

};

__device__ void rotateQuarterCounterClockwise(double2 &v)
{
	double temp = v.x;
	v.x = v.y;
	v.y = -temp;
}

__device__ void calcCollision(ClusterCuda *cluster, CellCuda *collidingCell, double2 &velDelta, double &angVelDelta)
{
	double2 posA = cluster->pos;
	double2 velA = cluster->vel;
	double angVelA = cluster->angularVel * DEG_TO_RAD;
	double angMassA = 1.0;

	ClusterCuda *collidingCluster = collidingCell->cluster;
	double2 posB = collidingCluster->pos;
	double2 velB = collidingCluster->vel;
	double angVelB = collidingCluster->angularVel * DEG_TO_RAD;
	double angMassB = 1.0;

	double2 rAPp = { collidingCell->absPos.x - posA.x, collidingCell->absPos.y - posA.y };
	rotateQuarterCounterClockwise(rAPp);
	double2 rBPp = { collidingCell->absPos.x - posB.x, collidingCell->absPos.y - posB.y };
	rotateQuarterCounterClockwise(rBPp);
	double2 vAB = {
		-(velB.x - rBPp.x*angVelB) + (velA.x - rAPp.x*angVelA)
		, -(velB.y - rBPp.y*angVelB) + (velA.y - rAPp.y*angVelA)
	};

	double vAB_sqrt = sqrt(vAB.x*vAB.x + vAB.y*vAB.y);
	double2 n;
	if (vAB_sqrt < FP_PRECISION) {
		n.x = 1.0;
	}
	else {
		n = { -vAB.x / vAB_sqrt, -vAB.y / vAB_sqrt };
	}

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
		velDelta = { velDelta.x + j / massA * n.x, velDelta.y + j / massA * n.y };
		angVelDelta -= rAPp_dot_n * j / angMassA;
	}

	if (angMassA <= FP_PRECISION && angMassB > FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((n.x*n.x + n.y*n.y) * (1.0 / massA + 1.0 / massB)
			 + rBPp_dot_n * rBPp_dot_n / angMassB);

		velDelta = { velDelta.x + j / massA * n.x, velDelta.y + j / massA * n.y };
	}

	if (angMassA > FP_PRECISION && angMassB <= FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((n.x*n.x + n.y*n.y) * (1.0 / massA + 1.0 / massB)
			+ rAPp_dot_n * rAPp_dot_n / angMassA);

		velDelta = { velDelta.x + j / massA * n.x, velDelta.y + j / massA * n.y };
		angVelDelta -= rAPp_dot_n * j / angMassA;
	}

	if (angMassA <= FP_PRECISION && angMassB <= FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((n.x*n.x + n.y*n.y) * (1.0 / massA + 1.0 / massB));

		velDelta = { velDelta.x + j / massA * n.x, velDelta.y + j / massA * n.y };
	}

}
