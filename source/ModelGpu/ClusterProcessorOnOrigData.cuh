#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"

class ClusterProcessorOnOrigData
{
public:
    __inline__ __device__ void init_blockCall(SimulationData& data, int clusterIndex);

    __inline__ __device__ void processingMovement_blockCall();
    __inline__ __device__ void processingCollision_blockCall();
    __inline__ __device__ void destroyCloseCell_blockCall();
    __inline__ __device__ void processingRadiation_blockCall();

private:
    __inline__ __device__ void destroyCloseCell(Cell *cell);
    __inline__ __device__ void destroyCloseCell(float2 const& pos, Cell *cell);
    __inline__ __device__ bool areConnectable(Cell *cell1, Cell *cell2);

    SimulationData* _data;
    Map<Cell> _cellMap;

    Cluster *_cluster;
    BlockData _cellBlock;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ClusterProcessorOnOrigData::init_blockCall(SimulationData & data, int clusterIndex)
{
    _data = &data;
    _cluster = data.entities.clusterPointers.at(clusterIndex);
    _cellMap.init(data.size, data.cellMap);

    _cellBlock = calcPartition(_cluster->numCellPointers, threadIdx.x, blockDim.x);
}

__inline__ __device__ void ClusterProcessorOnOrigData::processingCollision_blockCall()
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
                Cell* otherCell = _cellMap.get(Math::add(cell->absPos, { dx, dy }));

                if (!otherCell || otherCell == cell) {
                    continue;
                }
                if (cluster == otherCell->cluster) {
                    continue;
                }
                if (cell->protectionCounter > 0 || otherCell->protectionCounter > 0) {
                    continue;
                }
                if (!cell->alive || !otherCell->alive) {
                    continue;
                }
                if (_cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
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
                Cell* otherCell = _cellMap.get(Math::add(cell->absPos, { dx, dy }));
                if (!otherCell || otherCell == cell) {
                    continue;
                }
                if (firstOtherCluster != otherCell->cluster) {
                    continue;
                }
                if (!cell->alive || !otherCell->alive) {
                    continue;
                }
                if (_cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
                    continue;
                }
                if (cell->protectionCounter > 0 || otherCell->protectionCounter > 0) {
                    avoidCollision = true;
                    break;
                }
                if (Math::length(Math::sub(cell->vel, otherCell->vel)) >= cudaSimulationParameters.cellFusionVelocity
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
                        Cell* otherCell = _cellMap.get(Math::add(cell->absPos, { dx, dy }));
                        if (!otherCell || otherCell == cell) {
                            continue;
                        }
                        if (firstOtherCluster != otherCell->cluster) {
                            continue;
                        }
                        if (!cell->alive || !otherCell->alive) {
                            continue;
                        }
                        if (_cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
                            continue;
                        }
                        if (Math::length(Math::sub(cell->vel, otherCell->vel)) < cudaSimulationParameters.cellFusionVelocity
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
            collisionCenterPos = Math::div(collisionCenterPos, numberOfCollidingCells);
            rAPp = { collisionCenterPos.x - cluster->pos.x, collisionCenterPos.y - cluster->pos.y };
            _cellMap.mapDisplacementCorrection(rAPp);
            rBPp = { collisionCenterPos.x - firstOtherCluster->pos.x, collisionCenterPos.y - firstOtherCluster->pos.y };
            _cellMap.mapDisplacementCorrection(rBPp);
            outwardVector = Math::sub(
                Physics::tangentialVelocity(rBPp, firstOtherCluster->vel, firstOtherCluster->angularVel),
                Physics::tangentialVelocity(rAPp, cluster->vel, cluster->angularVel));
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
                    Cell* otherCell = _cellMap.get(Math::add(cell->absPos, { dx, dy }));
                    if (!otherCell || otherCell == cell) {
                        continue;
                    }
                    if (firstOtherCluster != otherCell->cluster) {
                        continue;
                    }
                    if (!cell->alive || !otherCell->alive) {
                        continue;
                    }
                    if (_cellMap.mapDistance(cell->absPos, otherCell->absPos) >= cudaSimulationParameters.cellMaxDistance) {
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
    }
    __syncthreads();

    if (0 == threadIdx.x) {
        lock.releaseLock();
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessorOnOrigData::destroyCloseCell_blockCall()
{
    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell *origCell = _cluster->cellPointers[cellIndex];
        destroyCloseCell(origCell);
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessorOnOrigData::processingMovement_blockCall()
{
    __shared__ float rotMatrix[2][2];
    if (0 == threadIdx.x) {
        _cluster->angle += _cluster->angularVel;
        Math::angleCorrection(_cluster->angle);
        _cluster->pos = Math::add(_cluster->pos, _cluster->vel);
        _cellMap.mapPosCorrection(_cluster->pos);
        Math::rotationMatrix(_cluster->angle, rotMatrix);
    }
    __syncthreads();


    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell *cell = _cluster->cellPointers[cellIndex];

        float2 absPos;
        absPos.x = cell->relPos.x*rotMatrix[0][0] + cell->relPos.y*rotMatrix[0][1] + _cluster->pos.x;
        absPos.y = cell->relPos.x*rotMatrix[1][0] + cell->relPos.y*rotMatrix[1][1] + _cluster->pos.y;
        cell->absPos = absPos;

        auto r = Math::sub(cell->absPos, _cluster->pos);
        _cellMap.mapDisplacementCorrection(r);
        auto newVel = Physics::tangentialVelocity(r, _cluster->vel, _cluster->angularVel);

        auto a = Math::sub(newVel, cell->vel);
        if (Math::length(a) > cudaSimulationParameters.cellMaxForce) {
            if (_data->numberGen.random() < cudaSimulationParameters.cellMaxForceDecayProb) {
                cell->alive = false;
                _cluster->decompositionRequired = true;
            }
        }

        cell->vel = newVel;

        if (cell->protectionCounter > 0) {
            --cell->protectionCounter;
        }
        _cellMap.set(absPos, cell);
    }
    __syncthreads();

}

__inline__ __device__ void ClusterProcessorOnOrigData::processingRadiation_blockCall()
{
    __shared__ EntityFactory factory;
    if (0 == threadIdx.x) {
        factory.init(_data);
    }
    __syncthreads();

    for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
        Cell *cell = _cluster->cellPointers[cellIndex];

        if (_data->numberGen.random() < cudaSimulationParameters.radiationProbability) {
            auto &pos = cell->absPos;
            float2 particlePos = { static_cast<int>(pos.x) + _data->numberGen.random(3) - 1.5f,
                static_cast<int>(pos.y) + _data->numberGen.random(3) - 1.5f };
            _cellMap.mapPosCorrection(particlePos);
            float2 particleVel =
                Math::add(Math::mul(cell->vel, cudaSimulationParameters.radiationVelocityMultiplier),
                { (_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation,
                    (_data->numberGen.random() - 0.5f) * cudaSimulationParameters.radiationVelocityPerturbation });

            particlePos = Math::sub(particlePos, particleVel);	//because particle will still be moved in current time step
            float radiationEnergy = powf(cell->energy, cudaSimulationParameters.radiationExponent) * cudaSimulationParameters.radiationFactor;
            radiationEnergy = radiationEnergy / cudaSimulationParameters.radiationProbability;
            radiationEnergy = 2 * radiationEnergy * _data->numberGen.random();
            if (radiationEnergy > cell->energy - 1) {
                radiationEnergy = cell->energy - 1;
            }
            cell->energy -= radiationEnergy;
            factory.createParticle(radiationEnergy, particlePos, particleVel);
        }
        if (cell->energy < cudaSimulationParameters.cellMinEnergy) {
            cell->alive = false;
            cell->cluster->decompositionRequired = true;
        }
    }
    __syncthreads();

    if (_cluster->decompositionRequired) {
        BlockData tokenBlock = calcPartition(_cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (int tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto token = _cluster->tokenPointers[tokenIndex];
            if (!token->cell->alive) {
                atomicAdd(&token->cell->energy, token->energy);
            }
        }

        for (int cellIndex = _cellBlock.startIndex; cellIndex <= _cellBlock.endIndex; ++cellIndex) {
            auto cell = _cluster->cellPointers[cellIndex];
            if (!cell->alive) {
                auto pos = cell->absPos;
                _cellMap.mapPosCorrection(pos);
                factory.createParticle(cell->energy, pos, cell->vel);
            }
        }
    }

    __syncthreads();
}

__inline__ __device__ void ClusterProcessorOnOrigData::destroyCloseCell(Cell * cell)
{
    if (cell->protectionCounter > 0) {
        return;
    }
    destroyCloseCell(cell->absPos, cell);
}

__inline__ __device__ void ClusterProcessorOnOrigData::destroyCloseCell(float2 const & pos, Cell * cell)
{
    Cell* mapCell = _cellMap.get(pos);
    if (!mapCell || mapCell == cell) {
        return;
    }

    Cluster* mapCluster = mapCell->cluster;
    auto distance = _cellMap.mapDistance(cell->absPos, mapCell->absPos);
    if (distance < cudaSimulationParameters.cellMinDistance) {
        Cluster* cluster = cell->cluster;
        if (mapCluster->numCellPointers >= cluster->numCellPointers) {
            cell->alive = false;
            cluster->decompositionRequired = true;
        }
        else {
            mapCell->alive = false;
            mapCluster->decompositionRequired = true;
        }
    }
}

__inline__ __device__ bool ClusterProcessorOnOrigData::areConnectable(Cell * cell1, Cell * cell2)
{
    return cell1->numConnections < cell1->maxConnections && cell2->numConnections < cell2->maxConnections;
}
