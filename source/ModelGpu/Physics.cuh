#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include "CudaAccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"
#include "Math.cuh"
#include "Cell.cuh"
#include "Cluster.cuh"


class Physics
{
public:
	__inline__ __device__ static float2 tangentialVelocity(float2 const& positionFromCenter, float2 const& velocityOfCenter, float angularVel);
	__inline__ __device__ static float angularMomentum(float2 const& positionFromCenter, float2 const& velocityOfCenter);
	__inline__ __device__ static float angularVelocity(float angularMomentum, float angularMass);
    __inline__ __device__ static float angularVelocity(float angularMassOld, float angularMassNew, float angularVelOld);
    __inline__ __device__ static float2 calcNormalToCell(Cell *cell, float2 outward);
	__inline__ __device__ static void calcCollision(float2 const& vA1, float2 const& vB1, float2 const& rAPp, float2 const& rBPp,
		float angularVelA1, float angularVelB1, float2 const& n, float angularMassA, float angularMassB, float massA, float massB,
		float2& vA2, float2& vB2, float& angularVelA2, float& angularVelB2);
    __inline__ __device__ static float kineticEnergy(float mass, float2 const& vel, float angularMass, float angularVel);
	__inline__ __device__ static float linearKineticEnergy(float mass, float2 const& vel);
    __inline__ __device__ static float rotationalKineticEnergy(float angularMass, float angularVel);
    __inline__ __device__ static void calcImpulseIncrement(float2 const& impulse, float2 relPos, float mass,
        float angularMass, float2& velInc, float& angularVelInc);

private:
	__device__ __inline__ static float2 calcOutwardVector(Cell* cellA, Cell* cellB, MapInfo const& map);
};


/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ float2 Physics::calcNormalToCell(Cell *cell, float2 outward)
{
    Math::normalize(outward);
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
		float2 u = cell->connections[i]->absPos - cell->absPos;
        Math::normalize(u);
		float h = Math::dot(outward, u);
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
		return cell->absPos - minCell->absPos;
	}

	//calc normal vectors
	float temp = maxVector.x;
	maxVector.x = maxVector.y;
	maxVector.y = -temp;
	temp = minVector.x;
	minVector.x = -minVector.y;
	minVector.y = temp;
	auto result = minVector + maxVector;
    Math::normalize(result);
	return result;
}

__device__ __inline__ void Physics::calcCollision(float2 const & vA1, float2 const & vB1, float2 const & rAPp, 
	float2 const & rBPp, float angularVelA1, float angularVelB1, float2 const & n, float angularMassA, 
	float angularMassB, float massA, float massB, float2 & vA2, float2 & vB2, float & angularVelA2, float & angularVelB2)
{
	float2 vAB = (vA1 - rAPp * angularVelA1 * DEG_TO_RAD) - (vB1 - rBPp* angularVelB1 * DEG_TO_RAD);

	float vAB_dot_n = Math::dot(vAB, n);
	if (vAB_dot_n > 0.0) {
		vA2 = vA1;
		angularVelA2 = angularVelA1;
		vB2 = vB1;
		angularVelB2 = angularVelB1;
		return;
	}

	float rAPp_dot_n = Math::dot(rAPp, n);
	float rBPp_dot_n = Math::dot(rBPp, n);

	if (angularMassA > FP_PRECISION && angularMassB > FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			+ rAPp_dot_n * rAPp_dot_n / angularMassA + rBPp_dot_n * rBPp_dot_n / angularMassB);

		vA2 = vA1 + n * j / massA;
		angularVelA2 = angularVelA1 - (rAPp_dot_n * j / angularMassA) * RAD_TO_DEG;
		vB2 = vB1 - n * j / massB;
		angularVelB2 = angularVelB1 + (rBPp_dot_n * j / angularMassB) * RAD_TO_DEG;
	}

	if (angularMassA <= FP_PRECISION && angularMassB > FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			+ rBPp_dot_n * rBPp_dot_n / angularMassB);

		vA2 = vA1 + n * j / massA;
		angularVelA2 = angularVelA1;
		vB2 = vB1 - n * j / massB;
		angularVelB2 = angularVelB1 + (rBPp_dot_n * j / angularMassB) * RAD_TO_DEG;
	}

	if (angularMassA > FP_PRECISION && angularMassB <= FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB)
			+ rAPp_dot_n * rAPp_dot_n / angularMassA);

		vA2 = vA1 + n * j / massA;
		angularVelA2 = angularVelA1 - (rAPp_dot_n * j / angularMassA) * RAD_TO_DEG;
		vB2 = vB1 - n * j / massB;
		angularVelB2 = angularVelB1;
	}

	if (angularMassA <= FP_PRECISION && angularMassB <= FP_PRECISION) {
		float j = -2.0*vAB_dot_n / ((1.0 / massA + 1.0 / massB));

		vA2 = vA1 + n * j / massA;
		angularVelA2 = angularVelA1;
		vB2 = vB1 - n * j / massB;
		angularVelB2 = angularVelB1;
	}

}

__device__ __inline__ float Physics::linearKineticEnergy(float mass, float2 const & vel)
{
	return 0.5f * mass * Math::lengthSquared(vel);
}

__inline__ __device__ float Physics::rotationalKineticEnergy(float angularMass, float angularVel)
{
    angularVel *= DEG_TO_RAD;
    return 0.5f * angularMass * angularVel * angularVel;
}

__inline__ __device__ void Physics::calcImpulseIncrement(float2 const & impulse, float2 rAPp, float mass,
    float angularMass, float2 & velInc, float & angularVelInc)
{
    Math::rotateQuarterCounterClockwise(rAPp);
    velInc = impulse / mass;
    angularVelInc = -Math::dot(rAPp, impulse) / angularMass * RAD_TO_DEG;
}

__inline__ __device__ float Physics::kineticEnergy(float mass, float2 const& vel, float angularMass, float angularVel)
{
    return linearKineticEnergy(mass, vel) + rotationalKineticEnergy(angularMass, angularVel);
}

__device__ __inline__ float2 Physics::calcOutwardVector(Cell* cellA, Cell* cellB, MapInfo const& map)
{
	Cluster* clusterA = cellA->cluster;
	float2 posA = clusterA->pos;
	float2 velA = clusterA->vel;
	float angVelA = clusterA->angularVel * DEG_TO_RAD;

	Cluster* clusterB = cellB->cluster;
	float2 posB = clusterB->pos;
	float2 velB = clusterB->vel;
	float angVelB = clusterB->angularVel * DEG_TO_RAD;

    float2 rAPp = cellB->absPos - posA;
	map.mapDisplacementCorrection(rAPp);
    Math::rotateQuarterCounterClockwise(rAPp);
	float2 rBPp = cellB->absPos - posB;
	map.mapDisplacementCorrection(rBPp);
    Math::rotateQuarterCounterClockwise(rBPp);
	return (velB - rBPp * angVelB) - (velA - rAPp * angVelA);
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

__inline__ __device__ float Physics::angularVelocity(float angularMassOld, float angularMassNew, float angularVelOld)
{
    angularVelOld = angularVelOld*DEG_TO_RAD;
    float rotEnergy = angularMassOld*angularVelOld*angularVelOld;
    double angularVelNew = sqrtf(rotEnergy / angularMassNew);
    return angularVelNew*RAD_TO_DEG;
}
