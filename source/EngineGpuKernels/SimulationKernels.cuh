#pragma once

#include <cooperative_groups.h>
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "CellProcessor.cuh"
#include "CleanupKernels.cuh"

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

/************************************************************************/
/* Helpers for particles												*/
/************************************************************************/
/*
__global__ void particleProcessingStep1(SimulationData data)
{
	ParticleProcessor particleProcessor;
    particleProcessor.init_system(data);
    particleProcessor.repair_system();
    particleProcessor.processingMovement_system();
    particleProcessor.updateMap_system();
    particleProcessor.processingTransformation_system();
}

__global__ void particleProcessingStep2(SimulationData data)
{
    ParticleProcessor particleProcessor;
    particleProcessor.init_system(data);
    particleProcessor.processingCollision_system();
}

__global__ void particleProcessingStep3(SimulationData data)
{
	ParticleProcessor particleProcessor;
    particleProcessor.init_system(data);
    particleProcessor.processingDataCopy_system();
}
*/

__global__ void cellProcessingStep1(SimulationData data)
{
    CellProcessor::init(data);
}

__global__ void cellProcessingStep2(SimulationData data)
{
    CellProcessor::calcForces(data);
}

__global__ void cellProcessingStep3(SimulationData data)
{
    CellProcessor::calcPositions(data);
}

__global__ void cellProcessingStep4(SimulationData data)
{
    CellProcessor::calcForces(data);
}

__global__ void cellProcessingStep5(SimulationData data)
{
    CellProcessor::calcVelocities(data);
}

__global__ void cellProcessingStep6(SimulationData data)
{
    CellProcessor::calcAveragedVelocities(data);
}

__global__ void cellProcessingStep7(SimulationData data)
{
    CellProcessor::applyAveragedVelocities(data);
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void cudaCalcSimulationTimestep(SimulationData data)
{
    data.cellMap.reset();
    data.particleMap.reset();
    data.dynamicMemory.reset();

    KERNEL_CALL(cellProcessingStep1, data);
    KERNEL_CALL(cellProcessingStep2, data);
    KERNEL_CALL(cellProcessingStep3, data);
    KERNEL_CALL(cellProcessingStep4, data);
    KERNEL_CALL(cellProcessingStep5, data);
    KERNEL_CALL(cellProcessingStep6, data);
    KERNEL_CALL(cellProcessingStep7, data);
    /*
*/

    /*
    KERNEL_CALL(resetCellFunctionData, data);
    KERNEL_CALL(clusterProcessingStep1, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(tokenProcessingStep1, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(tokenProcessingStep2, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(tokenProcessingStep3, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(tokenProcessingStep4, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(clusterProcessingStep2, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(clusterProcessingStep3, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(clusterProcessingStep4, data, data.entities.clusterPointers.getNumEntries());
    KERNEL_CALL(particleProcessingStep1, data);
    KERNEL_CALL(particleProcessingStep2, data);
    KERNEL_CALL(particleProcessingStep3, data);

*/
    KERNEL_CALL_1_1(cleanupAfterSimulation, data);
}

