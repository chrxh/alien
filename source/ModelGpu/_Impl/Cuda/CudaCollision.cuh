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

__device__ __inline__ void rotateQuarterCounterClockwise_Kernel(double2 &v)
{
	double temp = v.x;
	v.x = v.y;
	v.y = -temp;
}

__device__ __inline__ double2 calcNormalToCell_Kernel(CellCuda *cell, double2 outward)
{
	normalize(outward);
	if (cell->numConnections < 2) {
		return outward;
	}

	CellCuda* minCell = nullptr;
	CellCuda* maxCell = nullptr;
	double2 minVector;
	double2 maxVector;
	double min_h = 0.0;	//h = angular distance from outward vector
	double max_h = 0.0;

	for (int i = 0; i < cell->numConnections; ++i) {

		//calculate h (angular distance from outward vector)
		double2 u = sub(cell->connections[i]->absPos, cell->absPos);
		normalize(u);
		double h = dot(outward, u);
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
	double temp = maxVector.x;
	maxVector.x = maxVector.y;
	maxVector.y = -temp;
	temp = minVector.x;
	minVector.x = -minVector.y;
	minVector.y = temp;
	auto result = add(minVector, maxVector);
	normalize(result);
	return result;
}
__device__ __inline__ void calcCollision_Kernel(ClusterCuda *cluster, CellCuda *collidingCell, CollisionData &collisionData, int2 const& size, bool reverse)
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

	double2 rAPp = sub(collidingCell->absPos, posA);
	mapDisplacementCorrection_Kernel(rAPp, size);
	rotateQuarterCounterClockwise_Kernel(rAPp);
	double2 rBPp = sub(collidingCell->absPos, posB);
	mapDisplacementCorrection_Kernel(rBPp, size);
	rotateQuarterCounterClockwise_Kernel(rBPp);
	double2 vAB = sub(sub(velA, mul(rAPp, angVelA)), sub(velB, mul(rBPp, angVelB)));

	double2 outward = minus(vAB);
	double2 n = calcNormalToCell_Kernel(collidingCell, outward);

	if (angMassA < FP_PRECISION) {
		angVelA = 0.0;
	}
	if (angMassB < FP_PRECISION) {
		angVelB = 0.0;
	}

	double vAB_dot_n = dot(vAB, n);
	if (vAB_dot_n > 0.0) {
		return;
	}

	double massA = static_cast<double>(cluster->numCells);
	double massB = static_cast<double>(collidingCluster->numCells);
	double rAPp_dot_n = dot(rAPp, n);
	double rBPp_dot_n = dot(rBPp, n);

	if (angMassA > FP_PRECISION && angMassB > FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((1.0/massA + 1.0/massB)
			+ rAPp_dot_n * rAPp_dot_n / angMassA + rBPp_dot_n * rBPp_dot_n / angMassB);

		if (!reverse) {
			double temp = -(rAPp_dot_n * j / angMassA) * RAD_TO_DEG;
			atomicAdd(&collisionData.angularVelDelta, temp);
			temp = j / massA * n.x;
			atomicAdd(&collisionData.velDelta.x, temp);
			temp = j / massA * n.y;
			atomicAdd(&collisionData.velDelta.y, temp);
		}
		else {
			atomicAdd(&collisionData.angularVelDelta, (rBPp_dot_n * j / angMassB) * RAD_TO_DEG);
			atomicAdd(&collisionData.velDelta.x, -j / massB * n.x);
			atomicAdd(&collisionData.velDelta.y, -j / massB * n.y);
		}
	}

	if (angMassA <= FP_PRECISION && angMassB > FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			 + rBPp_dot_n * rBPp_dot_n / angMassB);

		if (!reverse) {
			atomicAdd(&collisionData.velDelta.x, j / massA * n.x);
			atomicAdd(&collisionData.velDelta.y, j / massA * n.y);
		}
		else {
			atomicAdd(&collisionData.angularVelDelta, (rBPp_dot_n * j / angMassB) * RAD_TO_DEG);
			atomicAdd(&collisionData.velDelta.x, -j / massB * n.x);
			atomicAdd(&collisionData.velDelta.y, -j / massB * n.y);
		}
	}

	if (angMassA > FP_PRECISION && angMassB <= FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			+ rAPp_dot_n * rAPp_dot_n / angMassA);

		if (!reverse) {
			atomicAdd(&collisionData.angularVelDelta, -(rAPp_dot_n * j / angMassA) * RAD_TO_DEG);
			atomicAdd(&collisionData.velDelta.x, j / massA * n.x);
			atomicAdd(&collisionData.velDelta.y, j / massA * n.y);
		}
		else {
			atomicAdd(&collisionData.velDelta.x, -j / massB * n.x);
			atomicAdd(&collisionData.velDelta.y, -j / massB * n.y);
		}
	}

	if (angMassA <= FP_PRECISION && angMassB <= FP_PRECISION) {
		double j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB));

		if (!reverse) {
			atomicAdd(&collisionData.velDelta.x, j / massA * n.x);
			atomicAdd(&collisionData.velDelta.y, j / massA * n.y);
		}
		else {
			atomicAdd(&collisionData.velDelta.x, -j / massB * n.x);
			atomicAdd(&collisionData.velDelta.y, -j / massB * n.y);
		}
	}

}

__device__ __inline__ void updateCollisionData_Kernel(int2 posInt, CellCuda *cell, CellCuda ** map
	, int2 const &size, CollisionData &collisionData)
{
	auto mapCell = getCellFromMap(posInt, map, size);
	if (mapCell != nullptr) {
		if (mapCell->cluster != cell->cluster) {
			if (mapDistanceSquared_Kernel(cell->absPos, mapCell->absPos, size) < CELL_MAX_DISTANCE*CELL_MAX_DISTANCE) {

				//*****
/*
				auto absPos = mapCell->absPos;
				int2 posInt = { static_cast<int>(absPos.x) , static_cast<int>(absPos.y) };
				mapPosCorrection(posInt, size);
				auto mapEntry = posInt.x + posInt.y * size.x;
				bool test = map[mapEntry] == mapCell;

				if(test)
					printf("+\n");
				else
					printf("-\n");
*/
				//*****

				atomicAdd(&collisionData.numCollisions, 1);
				calcCollision_Kernel(cell->cluster, mapCell, collisionData, size, false);
//				calcCollision_Kernel(mapCell->cluster, cell, collisionData, size, true);
			}
		}
	}
}

__device__ __inline__ void collectCollisionData_Kernel(CudaData const &data, CellCuda *cell, CollisionData &collisionData)
{
	auto absPos = cell->absPos;
	auto map = data.map1;
	auto size = data.size;
	int2 posInt = { static_cast<int>(absPos.x) , static_cast<int>(absPos.y) };
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
