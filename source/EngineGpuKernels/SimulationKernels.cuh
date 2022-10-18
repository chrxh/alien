#pragma once

#include <cooperative_groups.h>
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
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
__global__ void cudaNextTimestep_substep4(SimulationData data);
__global__ void cudaNextTimestep_substep5(SimulationData data);
__global__ void cudaNextTimestep_substep6(SimulationData data, SimulationResult result);
__global__ void cudaNextTimestep_substep7(SimulationData data);
__global__ void cudaNextTimestep_substep8(SimulationData data, SimulationResult result);
__global__ void cudaNextTimestep_substep9(SimulationData data);
__global__ void cudaNextTimestep_substep10(SimulationData data);
__global__ void cudaNextTimestep_substep11(SimulationData data);
__global__ void cudaNextTimestep_substep12(SimulationData data);
__global__ void cudaNextTimestep_substep13(SimulationData data);
__global__ void cudaNextTimestep_substep14(SimulationData data);

__global__ void cudaInitClusterData(SimulationData data);
__global__ void cudaFindClusterIteration(SimulationData data);
__global__ void cudaFindClusterBoundaries(SimulationData data);
__global__ void cudaAccumulateClusterPosAndVel(SimulationData data);
__global__ void cudaAccumulateClusterAngularProp(SimulationData data);
__global__ void cudaApplyClusterData(SimulationData data);
