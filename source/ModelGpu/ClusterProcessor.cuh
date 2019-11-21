#pragma once

#include "device_functions.h"
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
    __inline__ __device__ void init_blockCall(SimulationData& data, int clusterArrayIndex, int clusterIndex);

    __inline__ __device__ void processingMovement_blockCall();
    __inline__ __device__ void updateMap_blockCall();

    __inline__ __device__ void destroyCell_blockCall();

    __inline__ __device__ void processingCollision_blockCall();
    __inline__ __device__ void processingRadiation_blockCall();

    __inline__ __device__ void processingCellDeath_blockCall();
    __inline__ __device__ void processingMutation_blockCall();
    __inline__ __device__ void processingDecomposition_blockCall();
    __inline__ __device__ void processingClusterCopy_blockCall();

private:
    __inline__ __device__ void processingDecomposition_optimizedForSmallCluster_blockCall();
    __inline__ __device__ void processingDecomposition_optimizedForLargeCluster_blockCall();

    __inline__ __device__ void updateCellVelocity_blockCall(Cluster* cluster);
    __inline__ __device__ void cellAging(Cell* cell);
    __inline__ __device__ void destroyCloseCell(Cell* cell);
    __inline__ __device__ void destroyCloseCell(float2 const& pos, Cell *cell);
    __inline__ __device__ bool areConnectable(Cell *cell1, Cell *cell2);

    __inline__ __device__ void copyClusterWithDecomposition_blockCall();
    __inline__ __device__ void copyClusterWithFusion_blockCall();
    __inline__ __device__ void copyTokenPointers_blockCall(Cluster* sourceCluster, Cluster* targetCluster);
    __inline__ __device__ void copyTokenPointers_blockCall(Cluster* sourceCluster1, Cluster* sourceCluster2, Cluster* targetCluster);
    __inline__ __device__ void getNumberOfTokensToCopy_blockCall(Cluster* sourceCluster, Cluster* targetCluster, 
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

#define PROTECTION_TIMESTEPS 14

__inline__ __device__ void ClusterProcessor::processingCollision_blockCall()
{
    __shared__ Cluster* cluster;
    __shared__ Cluster* firstOtherCluster;
    __shared__ int firstOtherClusterId;
    __shared__ int numberOfCollidingCells;
    __shared__ float2 collisionCenterPos;
    __shared__ bool avoidCollision;
    enum CollisionState { ElasticCollision, Fusion };
    __shared__ CollisionState state;
    if (0 == threadIdx.x) {
        cluster = _cluster;
        firstOtherCluster = nullptr;
        firstOtherClusterId = 0;
        collisionCenterPos.x = 0;
        collisionCenterPos.y = 0;
        numberOfCollidingCells = 0;
        avoidCollision = false;
        state = CollisionState::ElasticCollision;
    }
    __syncthreads();

    //find colliding cluster
    for (auto index = _cellBlock.startIndex; index <= _cellBlock.endIndex; ++index) {
        Cell* cell = cluster->cellPointers[index];
        for (float dx = -1.0f; dx < 1.9f; dx += 1.0f) {
            for (float dy = -1.0f; dy < 1.9f; dy += 1.0f) {
                Cell* otherCell = _data->cellMap.get(cell->absPos + float2{ dx, dy });

                if (!otherCell || otherCell == cell) {
                    continue;
                }
                if (cluster == otherCell->cluster) {
                    continue;
                }
                if (cell->protectionCounter > 0 || otherCell->protectionCounter > 0) {
                    continue;
                }
                if (0 == cell->alive || 0 == otherCell->alive) {
                    continue;
                }
                if (_data->cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
                    continue;
                }
                int origFirstOtherClusterId = atomicCAS(&firstOtherClusterId, 0, otherCell->cluster->id);
                if (0 != origFirstOtherClusterId) {
                    continue;
                }

                firstOtherCluster = otherCell->cluster;
            }
        }
    }
    __syncthreads();

    if (!firstOtherCluster) {
        __syncthreads();
        return;
    }

    __shared__ DoubleLock lock;
    if (0 == threadIdx.x) {
        lock.init(&cluster->locked, &firstOtherCluster->locked);
        lock.tryLock();
    }
    __syncthreads();
    if (!lock.isLocked()) {
        __syncthreads();
        return;
    }

    for (auto index = _cellBlock.startIndex; index <= _cellBlock.endIndex; ++index) {
        Cell* cell = cluster->cellPointers[index];
        for (float dx = -1.0f; dx < 1.9f; dx += 1.0f) {
            for (float dy = -1.0f; dy < 1.9f; dy += 1.0f) {
                Cell* otherCell = _data->cellMap.get(cell->absPos + float2{ dx, dy });
                if (!otherCell || otherCell == cell) {
                    continue;
                }
                if (firstOtherCluster != otherCell->cluster) {
                    continue;
                }
                if (0 == cell->alive || 0 == otherCell->alive) {
                    continue;
                }
                if (_data->cellMap.mapDistance(cell->absPos, otherCell->absPos)
                    >= cudaSimulationParameters.cellMaxDistance) {
                    continue;
                }
                if (cell->protectionCounter > 0 || otherCell->protectionCounter > 0) {
                    avoidCollision = true;
                    break;
                }
                if (Math::length(cell->vel - otherCell->vel) >= cudaSimulationParameters.cellFusionVelocity
                    && areConnectable(cell, otherCell)) {
                    state = CollisionState::Fusion;
                }
                atomicAdd(&collisionCenterPos.x, otherCell->absPos.x);
                atomicAdd(&collisionCenterPos.y, otherCell->absPos.y);
                atomicAdd(&numberOfCollidingCells, 1);
            }
            if (avoidCollision) {
                break;
            }
        }
        if (avoidCollision) {
            break;
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
        if (nullptr == cluster->clusterToFuse && nullptr == firstOtherCluster->clusterToFuse
            /*&& !cluster->decompositionRequired && !firstOtherCluster->decompositionRequired*/) {
            for (auto index = _cellBlock.startIndex; index <= _cellBlock.endIndex; ++index) {
                Cell* cell = cluster->cellPointers[index];
                for (float dx = -1.0f; dx < 1.9f; dx += 1.0f) {
                    for (float dy = -1.0f; dy < 1.9f; dy += 1.0f) {
                        Cell* otherCell = _data->cellMap.get(cell->absPos + float2{ dx, dy });
                        if (!otherCell || otherCell == cell) {
                            continue;
                        }
                        if (firstOtherCluster != otherCell->cluster) {
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
                            cell->connections[index] = otherCell;
                            otherCell->connections[otherIndex] = cell;
                        }
                        else {
                            atomicAdd(&otherCell->numConnections, -1);
                        }
                    }
                }
            }
            __syncthreads();
            if (0 == threadIdx.x) {
                cluster->clusterToFuse = firstOtherCluster;
                firstOtherCluster->clusterToFuse = cluster;
            }
        }
    }

    if (CollisionState::ElasticCollision == state) {
        __shared__ float2 rAPp;
        __shared__ float2 rBPp;
        __shared__ float2 outwardVector;
        __shared__ float2 n;
        if (0 == threadIdx.x) {
            collisionCenterPos = collisionCenterPos / numberOfCollidingCells;
            rAPp = { collisionCenterPos.x - cluster->pos.x, collisionCenterPos.y - cluster->pos.y };
            _data->cellMap.mapDisplacementCorrection(rAPp);
            rBPp = { collisionCenterPos.x - firstOtherCluster->pos.x, collisionCenterPos.y - firstOtherCluster->pos.y };
            _data->cellMap.mapDisplacementCorrection(rBPp);
            outwardVector =
                Physics::tangentialVelocity(rBPp, firstOtherCluster->vel, firstOtherCluster->angularVel)
                - Physics::tangentialVelocity(rAPp, cluster->vel, cluster->angularVel);
            Math::rotateQuarterCounterClockwise(rAPp);
            Math::rotateQuarterCounterClockwise(rBPp);
            n.x = 0.0f;
            n.y = 0.0f;
        }
        __syncthreads();

        for (auto index = _cellBlock.startIndex; index <= _cellBlock.endIndex; ++index) {
            Cell* cell = cluster->cellPointers[index];
            for (float dx = -1.0f; dx < 1.9f; dx += 1.0f) {
                for (float dy = -1.0f; dy < 1.9f; dy += 1.0f) {
                    Cell* otherCell = _data->cellMap.get(cell->absPos + float2{ dx, dy });
                    if (!otherCell || otherCell == cell) {
                        continue;
                    }
                    if (firstOtherCluster != otherCell->cluster) {
                        continue;
                    }
                    if (0 == cell->alive || 0 == otherCell->alive) {
                        continue;
                    }
                    if (_data->cellMap.mapDistance(cell->absPos, otherCell->absPos)
                        >= cudaSimulationParameters.cellMaxDistance) {
                        continue;
                    }
                    float2 normal = Physics::calcNormalToCell(otherCell, outwardVector);
                    atomicAdd(&n.x, normal.x);
                    atomicAdd(&n.y, normal.y);
                    cell->protectionCounter = PROTECTION_TIMESTEPS;
                    otherCell->protectionCounter = PROTECTION_TIMESTEPS;
                }
            }
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            float mA = cluster->numCellPointers;
            float mB = firstOtherCluster->numCellPointers;
            float2 vA2{ 0.0f, 0.0f };
            float2 vB2{ 0.0f, 0.0f };
            float angularVelA2{ 0.0f };
            float angularVelB2{ 0.0f };
            Math::normalize(n);
            Physics::calcCollision(cluster->vel, firstOtherCluster->vel, rAPp, rBPp, cluster->angularVel,
                firstOtherCluster->angularVel, n, cluster->angularMass, firstOtherCluster->angularMass,
                mA, mB, vA2, vB2, angularVelA2, angularVelB2);

            cluster->vel = vA2;
            cluster->angularVel = angularVelA2;
            firstOtherCluster->vel = vB2;
            firstOtherCluster->angularVel = angularVelB2;
        }
        updateCellVelocity_blockCall(cluster);
        updateCellVelocity_blockCall(firstOtherCluster);
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        lock.releaseLock();
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::destroyCell_blockCall()
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell *cell = _cluster->cellPointers[cellIndex];
        destroyCloseCell(cell);
        cellAging(cell);
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::processingCellDeath_blockCall()
{
    if (1 == _cluster->decompositionRequired && !_cluster->clusterToFuse) {
        PartitionData tokenBlock = calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto token = _cluster->tokenPointers[tokenIndex];
            if (0 == token->cell->alive) {
                token->cell->changeEnergy(token->getEnergy());
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
                _factory.createParticle(cell->getEnergy() + kineticEnergy, pos, cell->vel, { cell->metadata.color });
                cell->setEnergy(0);
            }
        }
    }

    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::processingMutation_blockCall()
{
    for (auto cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        if (_data->numberGen.random() < cudaSimulationParameters.cellMutationProb) {
            auto const kindOfMutation = _data->numberGen.random(2);
            auto cell = _cluster->cellPointers[cellIndex];
            if (0 == kindOfMutation) {
                auto const index = static_cast<int>(_data->numberGen.random(MAX_CELL_STATIC_BYTES - 1));
                cell->staticData[index] = _data->numberGen.random(255);
            }
            if (1 == kindOfMutation) {
                if (Enums::CellFunction::COMPUTER == cell->cellFunctionType) {
                    cell->numStaticBytes =
                        _data->numberGen.random(cudaSimulationParameters.cellFunctionComputerMaxInstructions * 3);
                }
            }
            if (2 == kindOfMutation) {
                cell->cellFunctionType = _data->numberGen.random(Enums::CellFunction::_COUNTER - 1);
            }
        }
    }

    auto const tokenBlock = calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x);
    for (auto tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
        if (_data->numberGen.random() < (cudaSimulationParameters.cellMutationProb)*10) {
            auto token = _cluster->tokenPointers[tokenIndex];
            auto const index = static_cast<int>(_data->numberGen.random(MAX_TOKEN_MEM_SIZE - 1));
            token->memory[index] = _data->numberGen.random(255);
        }
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::processingDecomposition_blockCall()
{
    if (0 == _cluster->decompositionRequired || _cluster->clusterToFuse) {
        return;
    }

    if (_cluster->numCellPointers < 100) {
        processingDecomposition_optimizedForSmallCluster_blockCall();
    }
    else {
        processingDecomposition_optimizedForLargeCluster_blockCall();
    }
}

__inline__ __device__ void ClusterProcessor::processingMovement_blockCall()
{
    __shared__ float rotMatrix[2][2];
    if (0 == threadIdx.x) {
        _cluster->angle += _cluster->angularVel;
        Math::angleCorrection(_cluster->angle);
        _cluster->pos = _cluster->pos + _cluster->vel;
        _data->cellMap.mapPosCorrection(_cluster->pos);
        Math::rotationMatrix(_cluster->angle, rotMatrix);
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell *cell = _cluster->cellPointers[cellIndex];

        float2 absPos = Math::applyMatrix(cell->relPos, rotMatrix) + _cluster->pos;
        cell->absPos = absPos;

        if (cell->protectionCounter > 0) {
            --cell->protectionCounter;
        }
    }
    __syncthreads();

    updateCellVelocity_blockCall(_cluster);
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::updateMap_blockCall()
{
    _data->cellMap.set_blockCall(_cluster->numCellPointers, _cluster->cellPointers);
}

__inline__ __device__ void ClusterProcessor::processingRadiation_blockCall()
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell *cell = _cluster->cellPointers[cellIndex];

        if (_data->numberGen.random() < cudaSimulationParameters.radiationProb) {
            auto const cellEnergy = cell->getEnergy();
            auto &pos = cell->absPos;
            float2 particlePos = { static_cast<int>(pos.x) + _data->numberGen.random(3) - 1.5f,
                static_cast<int>(pos.y) + _data->numberGen.random(3) - 1.5f };
            _data->cellMap.mapPosCorrection(particlePos);
            float2 particleVel = (cell->vel * cudaSimulationParameters.radiationVelocityMultiplier)
                + float2{(_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                         (_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation};

            particlePos = particlePos - particleVel;	//because particle will still be moved in current time step
            float radiationEnergy = powf(cellEnergy, cudaSimulationParameters.radiationExponent) * cudaSimulationParameters.radiationFactor;
            radiationEnergy = radiationEnergy / cudaSimulationParameters.radiationProb;
            radiationEnergy = 2 * radiationEnergy * _data->numberGen.random();
            if (cellEnergy > 1) {
                if (radiationEnergy > cellEnergy - 1) {
                    radiationEnergy = cellEnergy - 1;
                }
                cell->changeEnergy(-radiationEnergy);
                auto particle = _factory.createParticle(radiationEnergy, particlePos, particleVel, { cell->metadata.color });
            }
        }
        if (cell->getEnergy() < cudaSimulationParameters.cellMinEnergy) {
            atomicExch(&cell->alive, 0);
            atomicExch(&cell->cluster->decompositionRequired, 1);
        }
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::updateCellVelocity_blockCall(Cluster* cluster)
{
    auto const cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
    for (int cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
        auto const& cell = cluster->cellPointers[cellIndex];

        auto r = cell->absPos - cluster->pos;
        _data->cellMap.mapDisplacementCorrection(r);
        auto newVel = Physics::tangentialVelocity(r, cluster->vel, cluster->angularVel);

        auto a = newVel - cell->vel;
        if (Math::length(a) > cudaSimulationParameters.cellMaxForce) {
            if (_data->numberGen.random() < cudaSimulationParameters.cellMaxForceDecayProb) {
                atomicExch(&cell->alive, 0);
                atomicExch(&cluster->decompositionRequired, 1);
            }
        }
        cell->vel = newVel;
    }
    __threadfence();
}

__inline__ __device__ void ClusterProcessor::cellAging(Cell * cell)
{
    if (++cell->age > cudaSimulationParameters.cellMinAge) {
        if (_data->numberGen.random() < 0.000001) {
            if (_data->numberGen.random() < 0.1) {
                atomicExch(&cell->alive, 0);
                atomicExch(&cell->cluster->decompositionRequired, 1);
            }
        }
    }
}

__inline__ __device__ void ClusterProcessor::destroyCloseCell(Cell * cell)
{
    if (cell->protectionCounter > 0) {
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

__inline__ __device__ void ClusterProcessor::init_blockCall(SimulationData& data,
    int clusterArrayIndex, int clusterIndex)
{
    _data = &data;

    _factory.init(_data);

    _clusterPointer = &data.entities.clusterPointerArrays.getArray(clusterArrayIndex).at(clusterIndex);
    _cluster = *_clusterPointer;

    _cellBlock = calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x);
}

#define MAX_DECOMPOSITIONS 5

__inline__ __device__ void ClusterProcessor::copyClusterWithDecomposition_blockCall()
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
            entries[i].cluster.pos = { 0.0f, 0.0f };
            entries[i].cluster.vel = { 0.0f, 0.0f };
            entries[i].cluster.angle = _cluster->angle;
            entries[i].cluster.angularVel = 0.0f;
            entries[i].cluster.angularMass = 0.0f;
            entries[i].cluster.numCellPointers = 0;
            entries[i].cluster.numTokenPointers = 0;
            entries[i].cluster.decompositionRequired = 0;
            entries[i].cluster.clusterToFuse = nullptr;
            entries[i].cluster.locked = 0;
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
                atomicAdd(&entries[index].cluster.pos.x, cell->absPos.x);
                atomicAdd(&entries[index].cluster.pos.y, cell->absPos.y);
                atomicAdd(&entries[index].cluster.vel.x, cell->vel.x);
                atomicAdd(&entries[index].cluster.vel.y, cell->vel.y);

                entries[index].cluster.id = _data->numberGen.createNewId_kernel();
                Math::inverseRotationMatrix(entries[index].cluster.angle, entries[index].invRotMatrix);
                foundMatch = true;
                break;
            }
            if (cell->tag == origTag) {	//matching entry
                atomicAdd(&entries[index].cluster.numCellPointers, 1);
                atomicAdd(&entries[index].cluster.pos.x, cell->absPos.x);
                atomicAdd(&entries[index].cluster.pos.y, cell->absPos.y);
                atomicAdd(&entries[index].cluster.vel.x, cell->vel.x);
                atomicAdd(&entries[index].cluster.vel.y, cell->vel.y);

                foundMatch = true;
                break;
            }
        }
        if (!foundMatch) {	//no match? use last entry
            cell->tag = entries[MAX_DECOMPOSITIONS - 1].tag;
            atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.numCellPointers, 1);
            atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.pos.x, cell->absPos.x);
            atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.pos.y, cell->absPos.y);
            atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.vel.x, cell->vel.x);
            atomicAdd(&entries[MAX_DECOMPOSITIONS - 1].cluster.vel.y, cell->vel.y);

            entries[MAX_DECOMPOSITIONS - 1].cluster.decompositionRequired = 1;
        }
    }
    __syncthreads();

    __shared__ Cluster* newClusters[MAX_DECOMPOSITIONS];
    PartitionData decompositionBlock = 
        calcPartition(numDecompositions, threadIdx.x, blockDim.x);
    for (int index = decompositionBlock.startIndex; index <= decompositionBlock.endIndex; ++index) {
        auto numCells = entries[index].cluster.numCellPointers;
        entries[index].cluster.pos.x /= numCells;
        entries[index].cluster.pos.y /= numCells;
        entries[index].cluster.vel.x /= numCells;
        entries[index].cluster.vel.y /= numCells;
        entries[index].cluster.cellPointers = _data->entities.cellPointers.getNewSubarray(numCells);
        entries[index].cluster.numCellPointers = 0;
        
        auto newClusterPointer = _data->entities.clusterPointerArrays.getNewClusterPointer(numCells);
        auto newCluster = _data->entities.clusters.getNewElement();
        *newClusterPointer = newCluster;

        newClusters[index] = newCluster;
        *newClusters[index] = entries[index].cluster;
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell* cell = _cluster->cellPointers[cellIndex];
        for (int index = 0; index < numDecompositions; ++index) {
            if (cell->tag == entries[index].tag) {
                Cluster* newCluster = newClusters[index];
                float2 deltaPos = cell->absPos - newCluster->pos;
                _data->cellMap.mapDisplacementCorrection(deltaPos);
                auto angularMass = deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y;
                atomicAdd(&newCluster->angularMass, angularMass);

                auto r = cell->absPos - _cluster->pos;
                _data->cellMap.mapDisplacementCorrection(r);
                float2 relVel = cell->vel - newCluster->vel;
                float angularMomentum = Physics::angularMomentum(r, relVel);
                atomicAdd(&newCluster->angularVel, angularMomentum);

                float(&invRotMatrix)[2][2] = entries[index].invRotMatrix;
                cell->relPos = Math::applyMatrix(deltaPos, invRotMatrix);
                int newCellIndex = atomicAdd(&newCluster->numCellPointers, 1);
                Cell*& newCellPointer = newCluster->cellPointers[newCellIndex];
                newCellPointer = cell;
                newCellPointer->cluster = newCluster;
            }
        }
    }
    __syncthreads();

    for (int index = decompositionBlock.startIndex; index <= decompositionBlock.endIndex; ++index) {
        Cluster* newCluster = newClusters[index];

        //newCluster->angularVel contains angular momentum until here
        newCluster->angularVel = Physics::angularVelocity(newCluster->angularVel, newCluster->angularMass);
    }


    for (int index = 0; index < numDecompositions; ++index) {
        Cluster* newCluster = newClusters[index];
        copyTokenPointers_blockCall(_cluster, newCluster);
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::copyClusterWithFusion_blockCall()
{
    if (_cluster < _cluster->clusterToFuse) {
        __shared__ Cluster* newCluster;
        __shared__ Cluster* otherCluster;
        __shared__ float2 correction;
        if (0 == threadIdx.x) {
            otherCluster = _cluster->clusterToFuse;
            newCluster = _data->entities.clusters.getNewElement();
            *_clusterPointer = newCluster;

            newCluster->id = _cluster->id;
            newCluster->angle = 0.0f;
            newCluster->numTokenPointers = 0;
            newCluster->numCellPointers = _cluster->numCellPointers + otherCluster->numCellPointers;
            newCluster->cellPointers = _data->entities.cellPointers.getNewSubarray(newCluster->numCellPointers);
            newCluster->decompositionRequired = _cluster->decompositionRequired | otherCluster->decompositionRequired;
            newCluster->locked = 0;
            newCluster->clusterToFuse = nullptr;

            correction = _data->cellMap.correctionIncrement(_cluster->pos, otherCluster->pos);	//to be added to otherCluster

            newCluster->pos = (_cluster->pos * _cluster->numCellPointers
                               + ((otherCluster->pos + correction) * otherCluster->numCellPointers))
                / newCluster->numCellPointers;
            newCluster->vel =
                (_cluster->vel * _cluster->numCellPointers + otherCluster->vel * otherCluster->numCellPointers)
                / newCluster->numCellPointers;
            newCluster->angularVel = 0.0f;
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
            float2 relVel = cell->vel - _cluster->vel;
            float angularMomentum = Physics::angularMomentum(r, relVel);
            atomicAdd(&newCluster->angularVel, angularMomentum);
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

            float2 relVel = cell->vel - otherCluster->vel;
            float angularMomentum = Physics::angularMomentum(r, relVel);
            atomicAdd(&newCluster->angularVel, angularMomentum);
        }
        __syncthreads();

        __shared__ float regainedEnergyPerCell;
        if (0 == threadIdx.x) {
            newCluster->angularVel = Physics::angularVelocity(newCluster->angularVel, newCluster->angularMass);

            auto const origKineticEnergies =
                Physics::kineticEnergy(
                    _cluster->numCellPointers, _cluster->vel, _cluster->angularMass, _cluster->angularVel)
                + Physics::kineticEnergy(
                    otherCluster->numCellPointers,
                    otherCluster->vel,
                    otherCluster->angularMass,
                    otherCluster->angularVel);

            auto const newKineticEnergy =
                Physics::kineticEnergy(
                    newCluster->numCellPointers, newCluster->vel, newCluster->angularMass, newCluster->angularVel);

            auto const energyDiff = newKineticEnergy - origKineticEnergies;    //is negative for fusion
            regainedEnergyPerCell = -energyDiff / newCluster->numCellPointers;
        }
        __syncthreads();

        auto const newCellPartition = calcPartition(newCluster->numCellPointers, threadIdx.x, blockDim.x);

        for (int index = newCellPartition.startIndex; index <= newCellPartition.endIndex; ++index) {
            auto& cell = newCluster->cellPointers[index];
            if (cell->getEnergy() + regainedEnergyPerCell > 0) {
                cell->changeEnergy(regainedEnergyPerCell);
            }
        }
        updateCellVelocity_blockCall(newCluster);
        copyTokenPointers_blockCall(_cluster, _cluster->clusterToFuse, newCluster);
    }
    else {
        if (0 == threadIdx.x) {
            *_clusterPointer = nullptr;
        }
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessor::copyTokenPointers_blockCall(Cluster* sourceCluster, Cluster* targetCluster)
{
    __shared__ int numberOfTokensToCopy;
    __shared__ int tokenCopyIndex;
    if (0 == threadIdx.x) {
        numberOfTokensToCopy = 0;
        tokenCopyIndex = 0;
    }
    __syncthreads();

    PartitionData tokenBlock = calcPartition(sourceCluster->numTokenPointers, threadIdx.x, blockDim.x);
    getNumberOfTokensToCopy_blockCall(sourceCluster, targetCluster, numberOfTokensToCopy, tokenBlock);

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

__inline__ __device__ void
ClusterProcessor::copyTokenPointers_blockCall(Cluster* sourceCluster1, Cluster* sourceCluster2, Cluster* targetCluster)
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

    getNumberOfTokensToCopy_blockCall(sourceCluster1, targetCluster, numberOfTokensToCopy, tokenBlock1);
    getNumberOfTokensToCopy_blockCall(sourceCluster2, targetCluster, numberOfTokensToCopy, tokenBlock2);

    if (0 == threadIdx.x) {
        targetCluster->numTokenPointers = numberOfTokensToCopy;
        targetCluster->tokenPointers = _data->entities.tokenPointers.getNewSubarray(numberOfTokensToCopy);
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
ClusterProcessor::getNumberOfTokensToCopy_blockCall(Cluster* sourceCluster, Cluster* targetCluster, 
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

__inline__ __device__ void ClusterProcessor::processingClusterCopy_blockCall()
{
    if (_cluster->numCellPointers == 1 && 0 == _cluster->cellPointers[0]->alive && !_cluster->clusterToFuse) {
        if (0 == threadIdx.x) {
            *_clusterPointer = nullptr;
        }
        __syncthreads();
        return;
    }

    if (1 == _cluster->decompositionRequired && !_cluster->clusterToFuse) {
        copyClusterWithDecomposition_blockCall();
    }
    else if (_cluster->clusterToFuse) {
        copyClusterWithFusion_blockCall();
    }
}

__inline__ __device__ void ClusterProcessor::processingDecomposition_optimizedForSmallCluster_blockCall()
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
                    Cell& otherCell = *cell->connections[i];
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

__inline__ __device__ void ClusterProcessor::processingDecomposition_optimizedForLargeCluster_blockCall()
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        auto& cell = _cluster->cellPointers[cellIndex];
        if (1 == cell->alive) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto& otherCell = cell->connections[i];
                if (1 != otherCell->alive) {
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
        dynamicMemory.cellsToEvaluate = _data->arrays.getArray<Cell*>(_cluster->numCellPointers);
        dynamicMemory.cellsToEvaluateNextRound = _data->arrays.getArray<Cell*>(_cluster->numCellPointers);
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
                Tagger::tagComponent_blockCall(_cluster, startCell, nullptr, currentTag, 0, dynamicMemory);
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
