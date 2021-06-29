#pragma once

#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "Tagger.cuh"
#include "DEBUG_cluster.cuh"

class ClusterProcessor
{
public:
    __inline__ __device__ void init_block(SimulationData& data, int clusterIndex);

    __inline__ __device__ void processingMovement_block();
    __inline__ __device__ void updateMap_block();

    __inline__ __device__ void destroyCloseCell_block();

    __inline__ __device__ void processingCollision_block();
    __inline__ __device__ void processingRadiation_block();

    __inline__ __device__ void processingCellDeath_block();
    __inline__ __device__ void processingDecomposition_block();
    __inline__ __device__ void processingClusterCopy_block();

    __inline__ __device__ void repair_block();

private:
    __inline__ __device__ float2 calcForce(Cell* cell, float2 const& pos, float2 force);

    __inline__ __device__ void processingDecomposition_optimizedForSmallCluster_block();
    __inline__ __device__ void processingDecomposition_optimizedForLargeCluster_block();

    __inline__ __device__ void destroyDyingCell(Cell* cell);
    __inline__ __device__ void destroyCloseCell(Cell* cell);
    __inline__ __device__ void destroyCloseCell(float2 const& pos, Cell *cell);
    __inline__ __device__ bool areConnectable(Cell *cell1, Cell *cell2);

    __inline__ __device__ void copyClusterWithDecomposition_block();
    __inline__ __device__ void copyClusterWithFusion_block();
    __inline__ __device__ void copyTokenPointers_block(Cluster* sourceCluster, Cluster* targetCluster);
    __inline__ __device__ void copyTokenPointers_block(
        Cluster* sourceCluster1,
        Cluster* sourceCluster2,
        int additionalTokenPointers,
        Cluster* targetCluster);
    __inline__ __device__ void getNumberOfTokensToCopy_block(Cluster* sourceCluster, Cluster* targetCluster,
        int& counter, PartitionData const& tokenBlock);


    SimulationData* _data;
    EntityFactory _factory;

    Cluster* _cluster;
    Cluster** _clusterPointer;

    PartitionData _cellBlock;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

 __inline__ __device__ void ClusterProcessor::processingCollision_block()
{
    for (auto index = _cellBlock.startIndex; index <= _cellBlock.endIndex; ++index) {
         auto cell = _cluster->cellPointers[index];
         for (float dx = -1.5f; dx < 1.51f; dx += 1.0f) {
             for (float dy = -1.5f; dy < 1.51f; dy += 1.0f) {
                 Cell* otherCell = _data->cellMap.get(cell->absPos + float2{dx, dy});

                 if (!otherCell || otherCell == cell) {
                     continue;
                 }

                 auto const otherCluster = otherCell->cluster;
                 if (_cluster == otherCluster) {
                     continue;
                 }
                 if (_cluster->isActive()) {
                     otherCluster->unfreeze(30);
                 }

/*
                 if (cell->getProtectionCounter_safe() > 0 || otherCell->getProtectionCounter_safe() > 0) {
                     continue;
                 }
*/
                 if (0 == cell->alive || 0 == otherCell->alive) {
                     continue;
                 }
                 if (_data->cellMap.mapDistance(cell->absPos, otherCell->absPos)
                     >= cudaSimulationParameters.cellMaxDistance) {
                     continue;
                 }

                 auto delta = cell->absPos - otherCell->absPos;
                 _data->cellMap.mapDisplacementCorrection(delta);
                 auto force =
                     Math::normalized(delta) * (cudaSimulationParameters.cellMaxDistance - Math::length(delta));

                 atomicAdd_block(&cell->force.x, force.x);
                 atomicAdd_block(&cell->force.y, force.y);
                 atomicAdd_block(&otherCell->force.x, -force.x);
                 atomicAdd_block(&otherCell->force.y, -force.y);
             }
         }
     }
     __syncthreads();

/*
    __shared__ Cluster* cluster;
    __shared__ unsigned long long int largestOtherClusterData;
    __shared__ Cluster* clustersArray;

    if (0 == threadIdx.x) {
        cluster = _cluster;
        largestOtherClusterData = 0;
        clustersArray = _data->entities.clusters.getArrayForDevice();
    }
    __syncthreads();

    //find colliding cluster
    for (auto index = _cellBlock.startIndex; index <= _cellBlock.endIndex; ++index) {
        Cell* cell = cluster->cellPointers[index];
        for (float dx = -0.5f; dx < 0.51f; dx += 1.0f) {
            for (float dy = -0.5f; dy < 0.51f; dy += 1.0f) {
                Cell* otherCell = _data->cellMap.get(cell->absPos + float2{ dx, dy });

                if (!otherCell || otherCell == cell) {
                    continue;
                }

                auto const otherCluster = otherCell->cluster;
                if (cluster == otherCluster) {
                    continue;
                }
                if (cluster->isActive()) {
                    otherCluster->unfreeze(30);
                }

                if (cell->getProtectionCounter_safe() > 0 || otherCell->getProtectionCounter_safe() > 0) {
                    continue;
                }
                if (0 == cell->alive || 0 == otherCell->alive) {
                    continue;
                }
                if (_data->cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
                    continue;
                }
                unsigned long long int otherClusterData = otherCell->cluster - clustersArray;
                otherClusterData |= (static_cast<unsigned long long int>(otherCluster->numCellPointers) << 32);
                atomicMax_block(&largestOtherClusterData, otherClusterData);
            }
        }
    }
    __syncthreads();

    if (0 == largestOtherClusterData) {
        __syncthreads();
        return;
    }

    __shared__ SystemDoubleLock lock;
    __shared__ Cluster* largestOtherCluster;
    __shared__ float2 collisionCenterPos;
    __shared__ int numberOfCollidingCells;
    __shared__ bool avoidCollision;
    enum CollisionState { ElasticCollision, Fusion };
    __shared__ CollisionState state;
    if (0 == threadIdx.x) {
        collisionCenterPos.x = 0;
        collisionCenterPos.y = 0;
        numberOfCollidingCells = 0;
        avoidCollision = false;
        state = CollisionState::ElasticCollision;
        largestOtherClusterData = largestOtherClusterData & 0xffffffff;
        largestOtherCluster = &clustersArray[largestOtherClusterData];
        lock.init(&cluster->locked, &largestOtherCluster->locked);
        lock.getLock();
    }
    __syncthreads();

    if (!lock.isLocked()) {
        __syncthreads();
        return;
    }

    for (auto index = _cellBlock.startIndex; index <= _cellBlock.endIndex; ++index) {
        Cell* cell = cluster->cellPointers[index];
        Cell* closestOtherCell = nullptr;
        float distanceOfClosestOtherCell = 0;
        for (float dx = -0.5f; dx < 0.51f; dx += 1.0f) {
            for (float dy = -0.5f; dy < 0.51f; dy += 1.0f) {
                Cell* otherCell = _data->cellMap.get(cell->absPos + float2{ dx, dy });
                if (!otherCell || otherCell == cell) {
                    continue;
                }
                if (largestOtherCluster != otherCell->cluster) {
                    continue;
                }
                if (0 == cell->alive || 0 == otherCell->alive) {
                    continue;
                }
                if (_data->cellMap.mapDistance(cell->absPos, otherCell->absPos)
                    >= cudaSimulationParameters.cellMaxDistance) {
                    continue;
                }
                if (cell->getProtectionCounter_safe() > 0 || otherCell->getProtectionCounter_safe() > 0) {
                    continue;
                }
                if (Math::length(cell->vel - otherCell->vel) >= cudaSimulationParameters.cellFusionVelocity
                    && areConnectable(cell, otherCell)) {
                    state = CollisionState::Fusion;
                }
                
                auto const distance = _data->cellMap.mapDistance(cell->absPos, otherCell->absPos);
                if (!closestOtherCell || distance < distanceOfClosestOtherCell) {
                    closestOtherCell = otherCell;
                    distanceOfClosestOtherCell = distance;
                }
            }
            if (avoidCollision) {
                break;
            }
        }
        if (avoidCollision) {
            break;
        }
        if (closestOtherCell) {
            atomicAdd_block(&collisionCenterPos.x, closestOtherCell->absPos.x);
            atomicAdd_block(&collisionCenterPos.y, closestOtherCell->absPos.y);
            atomicAdd_block(&numberOfCollidingCells, 1);
        }
    }
    __syncthreads();

    //also checking numberOfCollidingCells because it might happen that the colliding cells are not alive anymore... 
    if (avoidCollision || 0 == numberOfCollidingCells) {
        if (0 == threadIdx.x) {
            lock.releaseLock();
        }
        __syncthreads();
        return;
    }

    if (CollisionState::Fusion == state) {
        if (nullptr == cluster->clusterToFuse && nullptr == largestOtherCluster->clusterToFuse) {
            for (auto index = _cellBlock.startIndex; index <= _cellBlock.endIndex; ++index) {
                Cell* cell = cluster->cellPointers[index];
                for (float dx = -0.5f; dx < 0.51f; dx += 1.0f) {
                    for (float dy = -0.5f; dy < 0.51f; dy += 1.0f) {
                        Cell* otherCell = _data->cellMap.get(cell->absPos + float2{ dx, dy });
                        if (!otherCell || otherCell == cell) {
                            continue;
                        }
                        if (largestOtherCluster != otherCell->cluster) {
                            continue;
                        }
                        if (0 == cell->alive || 0 == otherCell->alive) {
                            continue;
                        }
                        if (_data->cellMap.mapDistance(cell->absPos, otherCell->absPos)
                            >= cudaSimulationParameters.cellMaxDistance) {
                            continue;
                        }
                        if (Math::length(cell->vel - otherCell->vel) < cudaSimulationParameters.cellFusionVelocity
                            || !areConnectable(cell, otherCell)) {
                            continue;
                        }
                        auto otherIndex = atomicAdd(&otherCell->numConnections, 1);
                        if (otherIndex < otherCell->maxConnections) {
                            auto index = cell->numConnections++;
                            cell->connections[index].cell = otherCell;
                            otherCell->connections[otherIndex].cell = cell;
                            cell->setFused(true);
                            otherCell->setFused(true);
                        }
                        else {
                            atomicAdd(&otherCell->numConnections, -1);
                        }
                    }
                }
            }
            __syncthreads();
            if (0 == threadIdx.x) {
                cluster->clusterToFuse = largestOtherCluster;
                largestOtherCluster->clusterToFuse = cluster;
                cluster->unfreeze(30);
                largestOtherCluster->unfreeze(30);
            }
        }
    }

    if (CollisionState::ElasticCollision == state) {
        __shared__ float2 rAPp;
        __shared__ float2 rBPp;
        __shared__ float2 outwardVector;
        __shared__ float2 n;
        __shared__ float2 clusterVel;
        __shared__ float clusterAngularVel;
        __shared__ float2 largestOtherClusterVel;
        __shared__ float largestOtherClusterAngularVel;

        if (0 == threadIdx.x) {
            collisionCenterPos = collisionCenterPos / numberOfCollidingCells;
            rAPp = { collisionCenterPos.x - cluster->pos.x, collisionCenterPos.y - cluster->pos.y };
            _data->cellMap.mapDisplacementCorrection(rAPp);
            rBPp = { collisionCenterPos.x - largestOtherCluster->pos.x, collisionCenterPos.y - largestOtherCluster->pos.y };
            _data->cellMap.mapDisplacementCorrection(rBPp);

            clusterVel = cluster->getVelocity();
            clusterAngularVel = cluster->getAngularVelocity();
            largestOtherClusterVel = largestOtherCluster->getVelocity();
            largestOtherClusterAngularVel = largestOtherCluster->getAngularVelocity();

            outwardVector =
                Physics::tangentialVelocity(rBPp, largestOtherClusterVel, largestOtherClusterAngularVel)
                - Physics::tangentialVelocity(rAPp, clusterVel, clusterAngularVel);
            Math::rotateQuarterCounterClockwise(rAPp);
            Math::rotateQuarterCounterClockwise(rBPp);
            n.x = 0.0f;
            n.y = 0.0f;
        }
        __syncthreads();

        for (auto index = _cellBlock.startIndex; index <= _cellBlock.endIndex; ++index) {
            Cell* cell = cluster->cellPointers[index];
            Cell* closestOtherCell = nullptr;
            float distanceOfClosestOtherCell = 0;
            for (float dx = -0.5f; dx < 0.51f; dx += 1.0f) {
                for (float dy = -0.5f; dy < 0.51f; dy += 1.0f) {
                    Cell* otherCell = _data->cellMap.get(cell->absPos + float2{ dx, dy });
                    if (!otherCell || otherCell == cell) {
                        continue;
                    }
                    if (largestOtherCluster != otherCell->cluster) {
                        continue;
                    }
                    if (0 == cell->alive || 0 == otherCell->alive) {
                        continue;
                    }
                    if (_data->cellMap.mapDistance(cell->absPos, otherCell->absPos)
                        >= cudaSimulationParameters.cellMaxDistance) {
                        continue;
                    }
                    if (cell->getProtectionCounter_safe() > 0 || otherCell->getProtectionCounter_safe() > 0) {
                        continue;
                    }
                    auto const distance = _data->cellMap.mapDistance(cell->absPos, otherCell->absPos);
                    if (!closestOtherCell || distance < distanceOfClosestOtherCell) {
                        closestOtherCell = otherCell;
                        distanceOfClosestOtherCell = distance;
                    }
                }
            }
            if (closestOtherCell) {
                float2 normal = Physics::calcNormalToCell(closestOtherCell, outwardVector);
                atomicAdd_block(&n.x, normal.x);
                atomicAdd_block(&n.y, normal.y);
                cell->activateProtectionCounter_safe();
                closestOtherCell->activateProtectionCounter_safe();
            }
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            float mA = cluster->numCellPointers;
            float mB = largestOtherCluster->numCellPointers;
            float2 vA2{ 0.0f, 0.0f };
            float2 vB2{ 0.0f, 0.0f };
            float angularVelA2{ 0.0f };
            float angularVelB2{ 0.0f };
            Math::normalize(n);
            
            Physics::calcCollision(clusterVel, largestOtherClusterVel, rAPp, rBPp, clusterAngularVel,
                largestOtherClusterAngularVel, n, cluster->angularMass, largestOtherCluster->angularMass,
                mA, mB, vA2, vB2, angularVelA2, angularVelB2);

            cluster->setVelocity(vA2);
            cluster->setAngularVelocity(angularVelA2);
            largestOtherCluster->setVelocity(vB2);
            largestOtherCluster->setAngularVelocity(angularVelB2);
        }
        updateCellVelocity_block(cluster);
        updateCellVelocity_block(largestOtherCluster);
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        lock.releaseLock();
    }
    __syncthreads();
*/
}

__inline__ __device__ void ClusterProcessor::destroyCloseCell_block()
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell *cell = _cluster->cellPointers[cellIndex];
        destroyCloseCell(cell);
        destroyDyingCell(cell);
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::processingCellDeath_block()
{
    if (1 == _cluster->decompositionRequired && !_cluster->clusterToFuse) {
        PartitionData tokenBlock = calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto token = _cluster->tokenPointers[tokenIndex];
            if (0 == token->cell->alive) {
                token->cell->changeEnergy_safe(token->getEnergy());
                token->setEnergy(0);
            }
        }
        __syncthreads();

        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            auto cell = _cluster->cellPointers[cellIndex];
            if (0 == cell->alive) {
                auto pos = cell->absPos;
                _data->cellMap.mapPosCorrection(pos);
                auto const kineticEnergy = Physics::linearKineticEnergy(1.0f, cell->vel);
                _factory.createParticle(cell->getEnergy_safe() + kineticEnergy, pos, cell->vel, { cell->metadata.color });
                cell->setEnergy_safe(0);
            }
        }
    }

    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::processingDecomposition_block()
{
    if (0 == _cluster->decompositionRequired || _cluster->clusterToFuse) {
        return;
    }

    if (_cluster->numCellPointers < 100) {
        processingDecomposition_optimizedForSmallCluster_block();
    }
    else {
        processingDecomposition_optimizedForLargeCluster_block();
    }
}

__inline__ __device__ float2 ClusterProcessor::calcForce(Cell* cell, float2 const& pos, float2 force)
{
    float2 displacement;
    float2 prevDisplacement;
    for (int index = 0; index < cell->numConnections; ++index) {
        auto connectingCell = cell->connections[index].cell;
        auto otherPos = connectingCell->absPos /*+ otherVel / 2*/;

        auto actualDistance = _data->cellMap.mapDistance(otherPos, pos);
        auto deviation = /*min(*/actualDistance - cell->connections[index].distance/*, cell->connections[index].distance)*/;
        auto delta = otherPos - pos;
        _data->cellMap.mapDisplacementCorrection(delta);
        //            if (abs(deviation) > 0.05) {
        force = force + Math::normalized(delta) * deviation / 2;
        //            }

        displacement = delta;
        _data->cellMap.mapDisplacementCorrection(displacement);
/*
        if (index > 0) {
            auto angle = Math::angleOfVector(displacement);
            auto prevAngle = Math::angleOfVector(prevDisplacement);
            auto actualAngleToPrevious = Math::subtractAngle(angle, prevAngle);
            auto origActualAngleToPrevious = actualAngleToPrevious;
            if (actualAngleToPrevious > 180) {
                actualAngleToPrevious = abs(actualAngleToPrevious - 360.0f);
            }
            auto deviation = actualAngleToPrevious - cell->connections[index].angleToPrevious;
            auto correctionMovementForLowAngle = Math::normalized((displacement + prevDisplacement) / 2);
            if (abs(deviation) > 5) {
                auto forceInc = correctionMovementForLowAngle * deviation / -10000;
                force = force + forceInc;
            }
        }
*/
        prevDisplacement = displacement;
    }
    return force;
}

__inline__ __device__ void ClusterProcessor::processingMovement_block()
{
/*
    for (int i = 0; i < 10; ++i) {
        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            auto cell = _cluster->cellPointers[cellIndex];
            cell->tempForce = {0, 0};
        }
        __syncthreads();

        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            auto cell = _cluster->cellPointers[cellIndex];
            auto dissipatedForce = cell->force / (cell->numConnections + 1);
            for (int index = 0; index < cell->numConnections; ++index) {
                auto connectingCell = cell->connections[index].cell;
                atomicAdd_block(&connectingCell->tempForce.x, dissipatedForce.x);
                atomicAdd_block(&connectingCell->tempForce.y, dissipatedForce.y);
            }
            atomicAdd_block(&cell->tempForce.x, dissipatedForce.x);
            atomicAdd_block(&cell->tempForce.y, dissipatedForce.y);
        }
        __syncthreads();

        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            auto cell = _cluster->cellPointers[cellIndex];
            cell->force = cell->tempForce;
        }
        __syncthreads();
    }
*/

    //Verlet integration
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto cell = _cluster->cellPointers[cellIndex];

        cell->tempForce = calcForce(cell, cell->absPos, cell->force);
        cell->tempPos = cell->absPos + cell->vel + cell->tempForce / 2;
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto cell = _cluster->cellPointers[cellIndex];

        cell->absPos = cell->tempPos;
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto cell = _cluster->cellPointers[cellIndex];

        auto force = calcForce(cell, cell->absPos, {0, 0});
        cell->tempVel = cell->vel + (cell->tempForce + force) / 2;
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto cell = _cluster->cellPointers[cellIndex];

        cell->vel = cell->tempVel;
        cell->force = {0, 0};
        cell->decrementProtectionCounter();
    }
    __syncthreads();

    //stab
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto cell = _cluster->cellPointers[cellIndex];

        cell->tempForce = {0, 0};
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto cell = _cluster->cellPointers[cellIndex];

        auto dissipatedVel = cell->vel;
        for (int index = 0; index < cell->numConnections; ++index) {
            auto connectingCell = cell->connections[index].cell;
            dissipatedVel = dissipatedVel + connectingCell->vel;
        }
        cell->tempVel = dissipatedVel / (cell->numConnections + 1);
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto cell = _cluster->cellPointers[cellIndex];

        cell->vel = cell->tempVel;
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::updateMap_block()
{
    _data->cellMap.set_block(_cluster->numCellPointers, _cluster->cellPointers);
}

__inline__ __device__ void ClusterProcessor::processingRadiation_block()
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell *cell = _cluster->cellPointers[cellIndex];

        if (_data->numberGen.random() < cudaSimulationParameters.radiationProb) {
            auto const cellEnergy = cell->getEnergy_safe();
            auto &pos = cell->absPos;
            float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
                + float2{ (_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                         (_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation };
            float2 particlePos = pos + Math::normalized(particleVel) * 1.5f;
            _data->cellMap.mapPosCorrection(particlePos);

            particlePos = particlePos - particleVel;	//because particle will still be moved in current time step
            float radiationEnergy = powf(cellEnergy, cudaSimulationParameters.radiationExponent) * cudaSimulationParameters.radiationFactor;
            radiationEnergy = radiationEnergy / cudaSimulationParameters.radiationProb;
            radiationEnergy = 2 * radiationEnergy * _data->numberGen.random();
            if (cellEnergy > 1) {
                if (radiationEnergy > cellEnergy - 1) {
                    radiationEnergy = cellEnergy - 1;
                }
                cell->changeEnergy_safe(-radiationEnergy);
                auto particle = _factory.createParticle(radiationEnergy, particlePos, particleVel, { cell->metadata.color });
            }
        }
        if (cell->getEnergy_safe() < cudaSimulationParameters.cellMinEnergy) {
            atomicExch(&cell->alive, 0);
            atomicExch(&cell->cluster->decompositionRequired, 1);
        }
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::destroyDyingCell(Cell * cell)
{
    if (cell->tokenUsages > cudaSimulationParameters.cellMinTokenUsages) {
        if (_data->numberGen.random() < cudaSimulationParameters.cellTokenUsageDecayProb) {
            atomicExch(&cell->alive, 0);
            atomicExch(&cell->cluster->decompositionRequired, 1);
        }
    }
}

__inline__ __device__ void ClusterProcessor::destroyCloseCell(Cell * cell)
{
    if (cell->getProtectionCounter() > 0) {
        return;
    }
    destroyCloseCell(cell->absPos, cell);
}

__inline__ __device__ void ClusterProcessor::destroyCloseCell(float2 const & pos, Cell * cell)
{
    Cell* cellFromMap = _data->cellMap.get(pos);
    if (!cellFromMap || cellFromMap == cell) {
        return;
    }

    Cluster* mapCluster = cellFromMap->cluster;
    auto distance = _data->cellMap.mapDistance(cell->absPos, cellFromMap->absPos);
    if (distance < cudaSimulationParameters.cellMinDistance) {
        Cluster* cluster = cell->cluster;
        if (mapCluster->numCellPointers >= cluster->numCellPointers) {
            atomicExch(&cell->alive, 0);
            atomicExch(&cluster->decompositionRequired, 1);
        }
        else {
            atomicExch(&cellFromMap->alive, 0);
            atomicExch(&mapCluster->decompositionRequired, 1);
        }
    }
}

__inline__ __device__ bool ClusterProcessor::areConnectable(Cell * cell1, Cell * cell2)
{
    return cell1->numConnections < cell1->maxConnections && cell2->numConnections < cell2->maxConnections;
}

__inline__ __device__ void ClusterProcessor::init_block(SimulationData& data, int clusterIndex)
{
    _data = &data;

    _factory.init(_data);

    _clusterPointer = &data.entities.clusterPointers.at(clusterIndex);
    _cluster = *_clusterPointer;

    _cellBlock = calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x);
}

#define MAX_DECOMPOSITIONS 5

__inline__ __device__ void ClusterProcessor::copyClusterWithDecomposition_block()
{
    __shared__ int numDecompositions;
    struct Entry {
        int tag;
        float invRotMatrix[2][2];
        Cluster cluster;
    };
    __shared__ Entry entries[MAX_DECOMPOSITIONS];
    if (0 == threadIdx.x) {
        *_clusterPointer = nullptr;
        numDecompositions = 0;
        for (int i = 0; i < MAX_DECOMPOSITIONS; ++i) {
            entries[i].tag = -1;
            entries[i].cluster.numCellPointers = 0;
            entries[i].cluster.numTokenPointers = 0;
            entries[i].cluster.decompositionRequired = 1;
            entries[i].cluster.clusterToFuse = nullptr;
            entries[i].cluster.locked = 0;
            entries[i].cluster.metadata.nameLen = 0;
            entries[i].cluster.init();
        }
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell* cell = _cluster->cellPointers[cellIndex];
        if (0 == cell->alive) {
            continue;
        }
        bool foundMatch = false;
        for (int index = 0; index < MAX_DECOMPOSITIONS; ++index) {
            int origTag = atomicCAS(&entries[index].tag, -1, cell->tag);
            if (-1 == origTag) {	//use free 
                atomicAdd(&numDecompositions, 1);
                atomicAdd(&entries[index].cluster.numCellPointers, 1);
                entries[index].cluster.id = _data->numberGen.createNewId_kernel();
                foundMatch = true;
                break;
            }
            if (cell->tag == origTag) {	//matching entry
                atomicAdd(&entries[index].cluster.numCellPointers, 1);
                foundMatch = true;
                break;
            }
        }
        if (!foundMatch) {	//no match? use last entry
            cell->tag = entries[MAX_DECOMPOSITIONS - 1].tag;
            atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.numCellPointers, 1);
        }
    }
    __syncthreads();

    __shared__ Cluster* newClusters[MAX_DECOMPOSITIONS];
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(_data);
        if (1 == numDecompositions) {
            entries[0].cluster.decompositionRequired = 0;
        }
    }
    __syncthreads();

    PartitionData decompositionBlock =
        calcPartition(numDecompositions, threadIdx.x, blockDim.x);
    for (int index = decompositionBlock.startIndex; index <= decompositionBlock.endIndex; ++index) {
        auto& clusterEntry = entries[index].cluster;
        auto numCells = clusterEntry.numCellPointers;
        clusterEntry.cellPointers = _data->entities.cellPointers.getNewSubarray(numCells);
        clusterEntry.numCellPointers = 0;

        auto const newCluster = factory.createCluster();

        newClusters[index] = newCluster;
        *newClusters[index] = entries[index].cluster;
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell* cell = _cluster->cellPointers[cellIndex];
        for (int index = 0; index < numDecompositions; ++index) {
            if (cell->tag == entries[index].tag) {
                Cluster* newCluster = newClusters[index];

                int newCellIndex = atomicAdd(&newCluster->numCellPointers, 1);
                Cell*& newCellPointer = newCluster->cellPointers[newCellIndex];
                newCellPointer = cell;
                newCellPointer->cluster = newCluster;
            }
        }
    }
    __syncthreads();

    for (int index = 0; index < numDecompositions; ++index) {
        Cluster* newCluster = newClusters[index];
        copyTokenPointers_block(_cluster, newCluster);
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::copyClusterWithFusion_block()
{
/*
    if (_cluster < _cluster->clusterToFuse) {
        __shared__ Cluster* newCluster;
        __shared__ Cluster* otherCluster;
        __shared__ float2 correction;
        if (0 == threadIdx.x) {
            EntityFactory factory;
            factory.init(_data);
            otherCluster = _cluster->clusterToFuse;
            newCluster = factory.createCluster(_clusterPointer);

            newCluster->id = _cluster->id;
            newCluster->angle = 0.0f;
            newCluster->numTokenPointers = 0;
            newCluster->numCellPointers = _cluster->numCellPointers + otherCluster->numCellPointers;
            newCluster->cellPointers = _data->entities.cellPointers.getNewSubarray(newCluster->numCellPointers);
            newCluster->decompositionRequired = 1;
            newCluster->locked = 0;
            newCluster->clusterToFuse = nullptr;

            correction = _data->cellMap.correctionIncrement(_cluster->pos, otherCluster->pos);	//to be added to otherCluster

            newCluster->pos = (_cluster->pos * _cluster->numCellPointers
                + ((otherCluster->pos + correction) * otherCluster->numCellPointers))
                / newCluster->numCellPointers;
            newCluster->setVelocity(
                (_cluster->getVelocity() * _cluster->numCellPointers
                 + otherCluster->getVelocity() * otherCluster->numCellPointers)
                / newCluster->numCellPointers);
            newCluster->setAngularVelocity(0);
            newCluster->angularMass = 0.0f;
        }
        __syncthreads();

        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            Cell* cell = _cluster->cellPointers[cellIndex];
            newCluster->cellPointers[cellIndex] = cell;
            auto relPos = cell->absPos - newCluster->pos;
            cell->relPos = relPos;
            cell->cluster = newCluster;
            atomicAdd(&newCluster->angularMass, Math::lengthSquared(relPos));

            auto r = cell->absPos - _cluster->pos;
            _data->cellMap.mapDisplacementCorrection(r);
            float2 relVel = cell->vel - _cluster->getVelocity();
            float angularMomentum = Physics::angularMomentum(r, relVel);
            newCluster->addAngularVelocity_safe(angularMomentum);
        }
        __syncthreads();

        PartitionData otherCellBlock = calcPartition(otherCluster->numCellPointers, threadIdx.x, blockDim.x);

        for (int otherCellIndex = otherCellBlock.startIndex; otherCellIndex <= otherCellBlock.endIndex; ++otherCellIndex) {
            Cell* cell = otherCluster->cellPointers[otherCellIndex];
            newCluster->cellPointers[_cluster->numCellPointers + otherCellIndex] = cell;
            auto r = cell->absPos - otherCluster->pos;
            _data->cellMap.mapDisplacementCorrection(r);

            cell->absPos = cell->absPos + correction;
            auto relPos = cell->absPos - newCluster->pos;
            cell->relPos = relPos;
            cell->cluster = newCluster;
            atomicAdd(&newCluster->angularMass, Math::lengthSquared(relPos));

            float2 relVel = cell->vel - otherCluster->getVelocity();
            float angularMomentum = Physics::angularMomentum(r, relVel);
            newCluster->addAngularVelocity_safe(angularMomentum);
        }
        __syncthreads();

        __shared__ float regainedEnergy;
        if (0 == threadIdx.x) {
            newCluster->setAngularVelocity(Physics::angularVelocity(newCluster->getAngularVelocity(), newCluster->angularMass));

            auto const origKineticEnergies =
                Physics::kineticEnergy(
                    _cluster->numCellPointers, _cluster->getVelocity(), _cluster->angularMass, _cluster->getAngularVelocity())
                + Physics::kineticEnergy(
                    otherCluster->numCellPointers,
                    otherCluster->getVelocity(),
                    otherCluster->angularMass,
                    otherCluster->getAngularVelocity());

            auto const newKineticEnergy =
                Physics::kineticEnergy(
                    newCluster->numCellPointers, newCluster->getVelocity(), newCluster->angularMass, newCluster->getAngularVelocity());

            regainedEnergy = origKineticEnergies - newKineticEnergy;    //should be negative for fusion
        }
        __syncthreads();

        auto const newCellPartition = calcPartition(newCluster->numCellPointers, threadIdx.x, blockDim.x);
        updateCellVelocity_block(newCluster);

        __shared__ int numFusedCell;
        __shared__ EntityFactory factory;
        if (0 == threadIdx.x) {
            numFusedCell = 0;
            factory.init(_data);
        }
        __syncthreads();

        for (int index = newCellPartition.startIndex; index <= newCellPartition.endIndex; ++index) {
            auto& cell = newCluster->cellPointers[index];
            cell->tag = 0;
            if (cell->isFused()) {
                atomicAdd_block(&numFusedCell, 1);
            }
        }
        __syncthreads();

        copyTokenPointers_block(_cluster, _cluster->clusterToFuse, numFusedCell, newCluster);
        __syncthreads();

        auto const newTokenPartition = calcPartition(newCluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (int index = newTokenPartition.startIndex; index <= newTokenPartition.endIndex; ++index) {
            auto& token = newCluster->tokenPointers[index];
            atomicAdd_block(&token->cell->tag, 1);
        }

        __syncthreads();

        for (int index = newCellPartition.startIndex; index <= newCellPartition.endIndex; ++index) {
            auto& cell = newCluster->cellPointers[index];
            if (!cell->isFused()) {
                continue;
            }
            if (cell->tag < cudaSimulationParameters.cellMaxToken
                && cell->getEnergy_safe() + regainedEnergy / numFusedCell > cudaSimulationParameters.tokenMinEnergy) {
                auto energyForToken = regainedEnergy / numFusedCell;
                if (energyForToken < cudaSimulationParameters.tokenMinEnergy) {
                    auto const energyFromCell = cudaSimulationParameters.tokenMinEnergy - energyForToken;
                    cell->setEnergy_safe(cell->getEnergy_safe() - energyFromCell);
                    energyForToken = cudaSimulationParameters.tokenMinEnergy;
                }
                auto newToken = factory.createToken(cell, cell);
                auto const tokenIndex = atomicAdd_block(&newCluster->numTokenPointers, 1);
                newCluster->tokenPointers[tokenIndex] = newToken;
                newToken->setEnergy(energyForToken);
            }
            else {
                if (regainedEnergy > 0) {
                    cell->changeEnergy_safe(regainedEnergy / numFusedCell);
                }
            }
        }
        __syncthreads();
        for (int index = newCellPartition.startIndex; index <= newCellPartition.endIndex; ++index) {
            auto& cell = newCluster->cellPointers[index];
            cell->setFused(false);
        }
    }
    else {
        if (0 == threadIdx.x) {
            *_clusterPointer = nullptr;
        }
    }
    __syncthreads();
*/
}

__inline__ __device__ void ClusterProcessor::copyTokenPointers_block(Cluster* sourceCluster, Cluster* targetCluster)
{
    __shared__ int numberOfTokensToCopy;
    __shared__ int tokenCopyIndex;
    if (0 == threadIdx.x) {
        numberOfTokensToCopy = 0;
        tokenCopyIndex = 0;
    }
    __syncthreads();

    PartitionData tokenBlock = calcPartition(sourceCluster->numTokenPointers, threadIdx.x, blockDim.x);
    getNumberOfTokensToCopy_block(sourceCluster, targetCluster, numberOfTokensToCopy, tokenBlock);

    if (0 == threadIdx.x) {
        targetCluster->numTokenPointers = numberOfTokensToCopy;
        targetCluster->tokenPointers = _data->entities.tokenPointers.getNewSubarray(numberOfTokensToCopy);
    }
    __syncthreads();

    for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
        auto& token = sourceCluster->tokenPointers[tokenIndex];
        auto& cell = token->cell;
        if (cell->cluster == targetCluster) {
            int prevTokenCopyIndex = atomicAdd(&tokenCopyIndex, 1);
            targetCluster->tokenPointers[prevTokenCopyIndex] = token;
        }
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::copyTokenPointers_block(
    Cluster* sourceCluster1,
    Cluster* sourceCluster2,
    int additionalTokenPointers,
    Cluster* targetCluster)
{
    __shared__ int numberOfTokensToCopy;
    __shared__ int tokenCopyIndex;
    if (0 == threadIdx.x) {
        numberOfTokensToCopy = 0;
        tokenCopyIndex = 0;
    }
    __syncthreads();

    PartitionData tokenBlock1 = calcPartition(sourceCluster1->numTokenPointers, threadIdx.x, blockDim.x);
    PartitionData tokenBlock2 = calcPartition(sourceCluster2->numTokenPointers, threadIdx.x, blockDim.x);

    getNumberOfTokensToCopy_block(sourceCluster1, targetCluster, numberOfTokensToCopy, tokenBlock1);
    getNumberOfTokensToCopy_block(sourceCluster2, targetCluster, numberOfTokensToCopy, tokenBlock2);

    if (0 == threadIdx.x) {
        targetCluster->numTokenPointers = numberOfTokensToCopy;
        targetCluster->tokenPointers =
            _data->entities.tokenPointers.getNewSubarray(numberOfTokensToCopy + additionalTokenPointers);
    }
    __syncthreads();

    for (int tokenIndex = tokenBlock1.startIndex; tokenIndex <= tokenBlock1.endIndex; ++tokenIndex) {
        auto& token = sourceCluster1->tokenPointers[tokenIndex];
        auto& cell = token->cell;
        if (cell->cluster == targetCluster) {
            int prevTokenCopyIndex = atomicAdd(&tokenCopyIndex, 1);
            targetCluster->tokenPointers[prevTokenCopyIndex] = token;
        }
    }
    __syncthreads();

    for (int tokenIndex = tokenBlock2.startIndex; tokenIndex <= tokenBlock2.endIndex; ++tokenIndex) {
        auto& token = sourceCluster2->tokenPointers[tokenIndex];
        auto& cell = token->cell;
        if (cell->cluster == targetCluster) {
            int prevTokenCopyIndex = atomicAdd(&tokenCopyIndex, 1);
            targetCluster->tokenPointers[prevTokenCopyIndex] = token;
        }
    }
    __syncthreads();

}

__inline__ __device__ void
ClusterProcessor::getNumberOfTokensToCopy_block(Cluster* sourceCluster, Cluster* targetCluster,
    int& counter, PartitionData const& tokenBlock)
{
    for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
        auto const& token = sourceCluster->tokenPointers[tokenIndex];
        auto const& cell = token->cell;
        if (cell->cluster == targetCluster) {
            atomicAdd(&counter, 1);
        }
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::processingClusterCopy_block()
{
    if (_cluster->numCellPointers == 1 && 0 == _cluster->cellPointers[0]->alive && !_cluster->clusterToFuse) {
        if (0 == threadIdx.x) {
            *_clusterPointer = nullptr;
        }
        __syncthreads();
        return;
    }

    if (1 == _cluster->decompositionRequired && !_cluster->clusterToFuse) {
        copyClusterWithDecomposition_block();
    }
    else if (_cluster->clusterToFuse) {
        copyClusterWithFusion_block();
    }
    _cluster->timestepSimulated();
}

__inline__ __device__ void ClusterProcessor::repair_block()
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto cell = _cluster->cellPointers[cellIndex];
        cell->repair();
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::processingDecomposition_optimizedForSmallCluster_block()
{
    __shared__ bool changes;

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        _cluster->cellPointers[cellIndex]->tag = cellIndex;
    }
    do {
        changes = false;
        __syncthreads();
        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            Cell* cell = _cluster->cellPointers[cellIndex];
            if (1 == cell->alive) {
                for (int i = 0; i < cell->numConnections; ++i) {
                    Cell& otherCell = *cell->connections[i].cell;
                    if (1 == otherCell.alive) {
                        if (otherCell.tag < cell->tag) {
                            cell->tag = otherCell.tag;
                            changes = true;
                        }
                    }
                    else {
                        for (int j = i + 1; j < cell->numConnections; ++j) {
                            cell->connections[j - 1] = cell->connections[j];
                        }
                        --cell->numConnections;
                        --i;
                    }
                }
            }
            else {
                cell->numConnections = 0;
            }
        }
        __syncthreads();
    } while (changes);
}

__inline__ __device__ void ClusterProcessor::processingDecomposition_optimizedForLargeCluster_block()
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto& cell = _cluster->cellPointers[cellIndex];
        if (1 == cell->alive) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto& otherCell = cell->connections[i].cell;
                if (1 != otherCell->alive) {
                    for (int j = i + 1; j < cell->numConnections; ++j) {
                        cell->connections[j - 1].cell = cell->connections[j].cell;
                    }
                    --cell->numConnections;
                    --i;
                }
            }
        }
        else {
            cell->numConnections = 0;
        }
    }
    __syncthreads();


    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto& cell = _cluster->cellPointers[cellIndex];
        cell->tag = 0;
    }
    __syncthreads();

    __shared__ int currentTag;
    if (0 == threadIdx.x) {
        currentTag = 1;
    }
    __syncthreads();

    __shared__ Tagger::DynamicMemory dynamicMemory;
    if (0 == threadIdx.x) {
        dynamicMemory.cellsToEvaluate = _data->dynamicMemory.getArray<Cell*>(_cluster->numCellPointers);
        dynamicMemory.cellsToEvaluateNextRound = _data->dynamicMemory.getArray<Cell*>(_cluster->numCellPointers);
    }
    __syncthreads();

    __shared__ int startCellFound; // 0 = no, 1 = yes
    __shared__ Cell* startCell;

    do {
        __syncthreads();
        if (0 == threadIdx.x) {
            startCellFound = 0;
        }
        __syncthreads();

        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            auto& cell = _cluster->cellPointers[cellIndex];
            if (0 == cell->tag && 0 == startCellFound) {
                int orig = atomicExch_block(&startCellFound, 1);
                if (0 == orig) {
                    startCell = cell;
                    startCell->tag = currentTag;
                    break;
                }
            }
        }
        __syncthreads();

        if (1 == startCellFound) {
            if (startCell->alive) {
                Tagger::tagComponent_block(_cluster, startCell, nullptr, currentTag, 0, dynamicMemory);
            }
            __syncthreads();

            if (0 == threadIdx.x) {
                ++currentTag;
            }
        }
        __syncthreads();

    } while (1 == startCellFound);

    __syncthreads();
}
