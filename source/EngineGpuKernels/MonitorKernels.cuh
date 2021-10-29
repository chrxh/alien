#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "SimulationData.cuh"
#include "CudaMonitorData.cuh"

/************************************************************************/
/* Helpers    															*/
/************************************************************************/
__global__ void getEnergyForMonitorData(SimulationData data, CudaMonitorData monitorData)
{
    {
        auto& cells = data.entities.cellPointers;
        auto const partition =
            calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& cell = cells.at(index);
            monitorData.incInternalEnergy(cell->energy);
        }
    }
    {
        auto& particles = data.entities.particlePointers;
        auto const partition =
            calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& particle = particles.at(index);
            monitorData.incInternalEnergy(particle->energy);
        }
    }
    {
        auto& tokens = data.entities.tokenPointers;
        auto const partition =
            calcPartition(tokens.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            auto& token = tokens.at(index);
            monitorData.incInternalEnergy(token->energy);
        }
    }
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void cudaGetCudaMonitorData(SimulationData data, CudaMonitorData monitorData)
{
    monitorData.reset();

    monitorData.setNumCells(data.entities.cellPointers.getNumEntries());
    monitorData.setNumParticles(data.entities.particlePointers.getNumEntries());
    monitorData.setNumTokens(data.entities.tokenPointers.getNumEntries());

//    KERNEL_CALL(getEnergyForMonitorData, data, monitorData);
}

