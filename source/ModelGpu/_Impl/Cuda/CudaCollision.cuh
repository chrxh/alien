#pragma once

#include "CudaShared.cuh"
#include "CudaBase.cuh"
#include "CudaConstants.cuh"

struct CollisionData
{
	ClusterCuda* collidingCluster;
	double2 velDelta;
	double angVelDelta;
};

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
