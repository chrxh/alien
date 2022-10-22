#pragma once

#include <cooperative_groups.h>
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "TOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "CellProcessor.cuh"
#include "ParticleProcessor.cuh"
#include "GarbageCollectorKernels.cuh"
#include "Operations.cuh"
#include "DebugKernels.cuh"
#include "SimulationResult.cuh" 

__global__ void cudaPrepareNextTimestep(SimulationData data, SimulationResult result);
__global__ void cudaNextTimestep_substep1(SimulationData data);
__global__ void cudaNextTimestep_substep2(SimulationData data);
__global__ void cudaNextTimestep_substep3(SimulationData data);
__global__ void cudaNextTimestep_verletPositionUpdate(SimulationData data);
__global__ void cudaNextTimestep_connectionForces(SimulationData data, bool considerAngles);
__global__ void cudaNextTimestep_verletVelocityUpdate(SimulationData data);
__global__ void cudaNextTimestep_collectCellFunctionOperation(SimulationData data);
__global__ void cudaNextTimestep_nerveFunction(SimulationData data, SimulationResult result);
__global__ void cudaNextTimestep_neuronFunction(SimulationData data, SimulationResult result);
__global__ void cudaNextTimestep_innerFriction(SimulationData data);
__global__ void cudaNextTimestep_frictionAndDecay(SimulationData data);
__global__ void cudaNextTimestep_structuralOperations_step1(SimulationData data);
__global__ void cudaNextTimestep_structuralOperations_step2(SimulationData data);
__global__ void cudaNextTimestep_substep14(SimulationData data);

__global__ void cudaInitClusterData(SimulationData data);
__global__ void cudaFindClusterIteration(SimulationData data);
__global__ void cudaFindClusterBoundaries(SimulationData data);
__global__ void cudaAccumulateClusterPosAndVel(SimulationData data);
__global__ void cudaAccumulateClusterAngularProp(SimulationData data);
__global__ void cudaApplyClusterData(SimulationData data);
