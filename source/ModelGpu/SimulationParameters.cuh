#pragma once

struct SimulationParameters
{
	float cellMaxDistance;
	float cellMinDistance;
	float cellMinEnergy;
	float cellFusionVelocity;
	float cellMaxForce;
	float cellMaxForceDecayProb;
	float cellMass;
	float cellTransformationProb;
	float radiationProbability;
	float radiationExponent;
	float radiationFactor;
	float radiationVelocityPerturbation;
	float radiationVelocityMultiplier;
};

__constant__ __device__ SimulationParameters cudaSimulationParameters;