#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include "CudaInterface.cuh"
#include "Base.cuh"
#include "CudaConstants.cuh"
#include "CudaSimulationParameters.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"


class Physics
{
public:
	__inline__ __device__ static void rotationMatrix(float angle, float(&rotMatrix)[2][2]);
	__inline__ __device__ static void inverseRotationMatrix(float angle, float(&rotMatrix)[2][2]);
	__inline__ __device__ static void angleCorrection(float &angle);
	__inline__ __device__ static float2 tangentialVelocity(float2 const& positionFromCenter, float2 const& velocityOfCenter, float angularVel);
	__inline__ __device__ static float angularMomentum(float2 const& positionFromCenter, float2 const& velocityOfCenter);
	__inline__ __device__ static float angularVelocity(float angularMomentum, float angularMass);
	__inline__ __device__ static void rotateQuarterCounterClockwise(float2 &v);
	__inline__ __device__ static float2 calcNormalToCell(Cell *cell, float2 outward);
	__inline__ __device__ static void calcCollision(float2 const& vA1, float2 const& vB1, float2 const& rAPp, float2 const& rBPp,
		float angularVelA1, float angularVelB1, float2 const& n, float angularMassA, float angularMassB, float massA, float massB,
		float2& vA2, float2& vB2, float& angularVelA2, float& angularVelB2);
	__inline__ __device__ static float linearKineticEnergy(float mass, float2 const& vel);

private:
	__device__ __inline__ static float2 calcOutwardVector(Cell* cellA, Cell* cellB, BasicMap const& map);
	__inline__ __device__ static void angleCorrection(int &angle);
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

//math constants
#define PI 3.1415926535897932384626433832795
#define DEG_TO_RAD PI/180.0
#define RAD_TO_DEG 180.0/PI

__device__ __inline__ void Physics::rotateQuarterCounterClockwise(float2 &v)
{
	float temp = v.x;
	v.x = v.y;
	v.y = -temp;
}

__device__ __inline__ float2 Physics::calcNormalToCell(Cell *cell, float2 outward)
{
	normalize(outward);
	if (cell->numConnections < 2) {
		return outward;
	}

	Cell* minCell = nullptr;
	Cell* maxCell = nullptr;
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

__device__ __inline__ void Physics::calcCollision(float2 const & vA1, float2 const & vB1, float2 const & rAPp, 
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

__device__ __inline__ float Physics::linearKineticEnergy(float mass, float2 const & vel)
{
	return 0.5f * mass * lengthSquared(vel);
}

__device__ __inline__ float2 Physics::calcOutwardVector(Cell* cellA, Cell* cellB, BasicMap const& map)
{
	Cluster* clusterA = cellA->cluster;
	float2 posA = clusterA->pos;
	float2 velA = clusterA->vel;
	float angVelA = clusterA->angularVel * DEG_TO_RAD;

	Cluster* clusterB = cellB->cluster;
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

__inline__ __device__ void Physics::rotationMatrix(float angle, float(&rotMatrix)[2][2])
{
	float sinAngle = __sinf(angle*DEG_TO_RAD);
	float cosAngle = __cosf(angle*DEG_TO_RAD);
	rotMatrix[0][0] = cosAngle;
	rotMatrix[0][1] = -sinAngle;
	rotMatrix[1][0] = sinAngle;
	rotMatrix[1][1] = cosAngle;
}

__inline__ __device__ void Physics::inverseRotationMatrix(float angle, float(&rotMatrix)[2][2])
{
	float sinAngle = __sinf(angle*DEG_TO_RAD);
	float cosAngle = __cosf(angle*DEG_TO_RAD);
	rotMatrix[0][0] = cosAngle;
	rotMatrix[0][1] = sinAngle;
	rotMatrix[1][0] = -sinAngle;
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

__inline__ __device__ float2 Physics::tangentialVelocity(float2 const& r, float2 const& vel, float angularVel)
{
	return { vel.x - angularVel*r.y * static_cast<float>(DEG_TO_RAD), vel.y + angularVel*r.x * static_cast<float>(DEG_TO_RAD) };
}

__inline__ __device__ float Physics::angularMomentum(float2 const & positionFromCenter, float2 const & velocityOfCenter)
{
	return positionFromCenter.x*velocityOfCenter.y - positionFromCenter.y*velocityOfCenter.x;
}

__inline__ __device__ float Physics::angularVelocity(float angularMomentum, float angularMass)
{
	if (std::abs(angularMass) < FP_PRECISION)
		return 0;
	else
		return angularMomentum / angularMass * RAD_TO_DEG;
}
