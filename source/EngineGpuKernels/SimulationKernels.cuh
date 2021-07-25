#pragma once

#include <cooperative_groups.h>
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "CellProcessor.cuh"
#include "ParticleProcessor.cuh"
#include "CleanupKernels.cuh"
#include "Operation.cuh"
#include "DebugKernels.cuh"

/************************************************************************/
/* Helpers for clusters													*/
/************************************************************************/
/*
__global__ void clusterProcessingStep1(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_block(data, clusterIndex);
        clusterProcessor.repair_block();
        clusterProcessor.processingMovement_block();
        clusterProcessor.updateMap_block();
    }
}

__global__ void clusterProcessingStep2(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_block(data, clusterIndex);
        clusterProcessor.destroyCloseCell_block();
        clusterProcessor.processingCollisionPrepare_block();
    }
}

__global__ void clusterProcessingStep3(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_block(data, clusterIndex);
        clusterProcessor
            .processingCollision_block();  //attention: can result a temporarily inconsistent state, will be resolved in step 4
        clusterProcessor.processingRadiation_block();
    }
}

__global__ void clusterProcessingStep4(SimulationData data, int numClusters)
{
    PartitionData clusterBlock = calcPartition(numClusters, blockIdx.x, gridDim.x);
    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {
        ClusterProcessor clusterProcessor;
        clusterProcessor.init_block(data, clusterIndex);
        clusterProcessor.processingFinalizeCollision_block();
        clusterProcessor.processingCellDeath_block();
        clusterProcessor.processingDecomposition_block();
        clusterProcessor.processingClusterCopy_block();
    }
}
*/

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
    cellProcessor.initForces(data);
}

__global__ void processingStep4(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.calcForces(data);

    ParticleProcessor particleProcessor;
    particleProcessor.movement(data);
    particleProcessor.collision(data);
}

__global__ void processingStep5(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.calcPositions(data);
}

__global__ void processingStep6(SimulationData data)
{
    CellProcessor cellProcessor;
    cellProcessor.calcForces(data);
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
    KERNEL_CALL(processingStep4, data);
    KERNEL_CALL(processingStep5, data);
    KERNEL_CALL(processingStep6, data);
    KERNEL_CALL(processingStep7, data);
    KERNEL_CALL(processingStep8, data);
    KERNEL_CALL(processingStep9, data);
    KERNEL_CALL(processingStep10, data);
    KERNEL_CALL(processingStep11, data);
    KERNEL_CALL(processingStep12, data);
    KERNEL_CALL(processingStep13, data);

    KERNEL_CALL_1_1(cleanupAfterSimulation, data);
}

