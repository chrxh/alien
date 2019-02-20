#pragma once

struct CudaSimulationParameters
{
	float cellMaxDistance;
	float cellMinDistance;
	float cellMinEnergy;
	float cellFusionVelocity;
	float radiationProbability;
	float radiationExponent;
	float radiationFactor;
	float radiationVelocityPerturbation;
};

__constant__ __device__ CudaSimulationParameters cudaSimulationParameters;