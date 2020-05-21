#pragma once

#include "ModelBasic/Colors.h"

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "CleanupKernels.cuh"

#include "SimulationData.cuh"

struct CudaApplyForceData
{
    float2 startPos;
    float2 endPos;
    float2 force;
};

__global__ void applyForceToClusters(CudaApplyForceData applyData, int2 universeSize, Array<Cluster*> clusters)
{
    auto const clusterBlock =
        calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);

    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {

        auto const& cluster = clusters.at(clusterIndex);

        __shared__ MapInfo map;
        __shared__ float2 accumulatedVelInc;
        __shared__ float accumulatedAngularVelInc;
        if (0 == threadIdx.x) {
            map.init(universeSize);
            accumulatedVelInc.x = 0;
            accumulatedVelInc.y = 0;
            accumulatedAngularVelInc = 0;
        }
        __syncthreads();

        auto const cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto const& cell = cluster->cellPointers[cellIndex];
            auto pos = cell->absPos;
            map.mapPosCorrection(pos);
            if (Math::isContainedInLineSegment(applyData.startPos, applyData.endPos, pos, 10)) {

                float2 velInc;
                float angularVelInc;
                Physics::calcImpulseIncrement(
                    applyData.force, cell->relPos, cluster->getMass(), cluster->angularMass, velInc, angularVelInc);
                atomicAdd_block(&accumulatedVelInc.x, velInc.x);
                atomicAdd_block(&accumulatedVelInc.y, velInc.y);
                atomicAdd_block(&accumulatedAngularVelInc, angularVelInc);

            }
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            cluster->addVelocity(accumulatedVelInc);
            cluster->addAngularVelocity(accumulatedAngularVelInc);
        }
    }
}

__global__ void applyForceToParticles(CudaApplyForceData applyData, int2 universeSize, Array<Particle*> particles)
{
    auto const particleBlock =
        calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = particles.at(index);
        auto const& pos = particle->absPos;
        if (Math::isContainedInLineSegment(applyData.startPos, applyData.endPos, pos, 10)) {
            particle->vel = particle->vel + applyData.force;
        }
    }
}


__global__ void cudaApplyForce(CudaApplyForceData applyData, SimulationData data)
{
    KERNEL_CALL(applyForceToClusters, applyData, data.size, data.entities.clusterPointers);
    if (data.entities.clusterFreezedPointers.getNumEntries() > 0) {
        KERNEL_CALL(applyForceToClusters, applyData, data.size, data.entities.clusterFreezedPointers);
    }
    KERNEL_CALL(applyForceToParticles, applyData, data.size, data.entities.particlePointers);
}
