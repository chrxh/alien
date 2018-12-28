#pragma once

struct CudaSimulationParameters
{
	float cellMaxDistance;
	float cellMinEnergy;
	float radiationProbability;
	float radiationExponent;
	float radiationFactor;
	float radiationVelocityPerturbation;
};

__constant__ __device__ CudaSimulationParameters cudaSimulationParameters;