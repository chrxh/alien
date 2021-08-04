#pragma once

#include <cooperative_groups.h>
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "CellProcessor.cuh"
#include "ParticleProcessor.cuh"
#include "TokenProcessor.cuh"
#include "CleanupKernels.cuh"
#include "Operation.cuh"
#include "DebugKernels.cuh"

/************************************************************************/
/* Helpers for tokens													*/
/************************************************************************/
/*
__global__ void resetCellFunctionData(SimulationData data)
{
    data.cellFunctionData.mapSectionCollector.reset_system();
}
__global__ void tokenProcessingStep1(SimulationData data, int numClusters)
{
    auto const clusterPartition = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        TokenProcessor tokenProcessor;
        tokenProcessor.init_block(data, clusterIndex);
        tokenProcessor.repair_block();
        tokenProcessor.processingEnergyAveraging_block();
        tokenProcessor.processingSpreading_block();
        tokenProcessor.processingLightWeigthedFeatures_block();
    }
}

__global__ void tokenProcessingStep2(SimulationData data, int numClusters)
{
    auto const clusterPartition = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterPartition.startIndex; clusterIndex <= clusterPartition.endIndex; ++clusterIndex) {
        TokenProcessor tokenProcessor;
        tokenProcessor.init_block(data, clusterIndex);
        tokenProcessor.createCellFunctionData_block();
    }
}

__global__ void tokenProcessingStep3(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        TokenProcessor tokenProcessor;
        tokenProcessor.init_block(data, clusterIndex);
        tokenProcessor.processingConstructors_block();
    }
}

__global__ void tokenProcessingStep4(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        TokenProcessor tokenProcessor;
        tokenProcessor.init_block(data, clusterIndex);
        tokenProcessor.processingCommunicatorsAnsSensors_block();
    }
}
*/

__global__ void processingStep1(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.init(data);
    cellProcessor.clearTag(data);
    cellProcessor.updateMap(data);
    cellProcessor.radiation(data);  //do not use ParticleProcessor in this kernel
}

__global__ void processingStep2(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.collisions(data);

    ParticleProcessor particleProcessor;
    particleProcessor.updateMap(data);
}

__global__ void processingStep3(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.applyAndInitForces(data);
    cellProcessor.clearTag(data);
}

__global__ void processingStep4(SimulationData data, int numTokenPointers)
{
    CellProcessor cellProcessor;
    cellProcessor.calcForces(data, data.entities.cellPointers.getNumEntries());

    ParticleProcessor particleProcessor;
    particleProcessor.movement(data);
    particleProcessor.collision(data);

    TokenProcessor tokenProcessor;
    tokenProcessor.movement(data, numTokenPointers);
}

__global__ void processingStep5(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.calcPositions(data);
}

__global__ void processingStep6(SimulationData data, int numTokenPointers, int numCellPointers)
{
    CellProcessor cellProcessor;
    cellProcessor.calcForces(data, numCellPointers);

    TokenProcessor tokenProcessor;
    tokenProcessor.executeCellFunctions(data, numTokenPointers);
}

__global__ void processingStep7(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.calcVelocities(data);
}

__global__ void processingStep8(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.calcAveragedVelocities(data);
}

__global__ void processingStep9(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.applyAveragedVelocities(data);
}

__global__ void processingStep10(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.decay(data);
} 

__global__ void processingStep11(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.processAddConnectionOperations(data);
}

__global__ void processingStep12(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.processDelOperations(data);
}

__global__ void processingStep13(SimulationData data)
{
    ParticleProcessor particleProcessor;
    particleProcessor.transformation(data);
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void cudaCalcSimulationTimestep(SimulationData data)
{
    data.cellMap.reset();
    data.particleMap.reset();
    data.dynamicMemory.reset();

    *data.numAddConnectionOperations = 0; 
    *data.numDelOperations = 0;
    data.addConnectionOperations =
        data.dynamicMemory.getArray<AddConnectionOperation>(data.entities.cellPointers.getNumEntries());
    data.delOperations =
        data.dynamicMemory.getArray<DelOperation>(data.entities.cellPointers.getNumEntries());

    KERNEL_CALL(processingStep1, data);
    KERNEL_CALL(processingStep2, data);
    KERNEL_CALL(processingStep3, data);
    KERNEL_CALL(processingStep4, data, data.entities.tokenPointers.getNumEntries());
    KERNEL_CALL(processingStep5, data);
    KERNEL_CALL(
        processingStep6,
        data,
        data.entities.tokenPointers.getNumEntries(),
        data.entities.cellPointers.getNumEntries());
    KERNEL_CALL(processingStep7, data);
    KERNEL_CALL(processingStep8, data);
    KERNEL_CALL(processingStep9, data);
    KERNEL_CALL(processingStep10, data);
    KERNEL_CALL(processingStep11, data);
    KERNEL_CALL(processingStep12, data);
    KERNEL_CALL(processingStep13, data);

    KERNEL_CALL_1_1(cleanupAfterSimulation, data);
}

