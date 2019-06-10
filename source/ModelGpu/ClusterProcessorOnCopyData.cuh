#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaAccessTOs.cuh"
#include "CudaConstants.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class ClusterProcessorOnCopyData
{
public:
    __inline__ __device__ void init_blockCall(SimulationData& data, int clusterIndex);

    __inline__ __device__ void processingDecomposition_blockCall();
    __inline__ __device__ void processingClusterCopy_blockCall();

private:
    __inline__ __device__ void copyClusterWithDecomposition_blockCall();
    __inline__ __device__ void copyCluster_blockCall();
    __inline__ __device__ void copyClusterWithFusion_blockCall();
    __inline__ __device__ void copyTokens_blockCall(Cluster* sourceCluster, Cluster* targetCluster);
    __inline__ __device__ void copyTokens_blockCall(Cluster* sourceCluster1, Cluster* sourceCluster2, Cluster* targetCluster);
    __inline__ __device__ void getNumberOfTokensToCopy_blockCall(Cluster* sourceCluster, Cluster* targetCluster, int& counter);


    SimulationData* _data;
    Map<Cell> _cellMap;

    Cluster *_origCluster;

    int _startCellIndex;
    int _endCellIndex;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ClusterProcessorOnCopyData::init_blockCall(SimulationData& data, int clusterIndex)
{
    _data = &data;
    _origCluster = &data.clusters.getEntireArray()[clusterIndex];
    _cellMap.init(data.size, data.cellMap);

    calcPartition(_origCluster->numCellPointers, threadIdx.x, blockDim.x, _startCellIndex, _endCellIndex);

    __syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::copyClusterWithDecomposition_blockCall()
{
    __shared__ int numDecompositions;
    struct Entry {
        int tag;
        float invRotMatrix[2][2];
        Cluster cluster;
    };
    __shared__ Entry entries[MAX_DECOMPOSITIONS];
    if (0 == threadIdx.x) {
        numDecompositions = 0;
        for (int i = 0; i < MAX_DECOMPOSITIONS; ++i) {
            entries[i].tag = -1;
            entries[i].cluster.pos = { 0.0f, 0.0f };
            entries[i].cluster.vel = { 0.0f, 0.0f };
            entries[i].cluster.angle = _origCluster->angle;
            entries[i].cluster.angularVel = 0.0f;
            entries[i].cluster.angularMass = 0.0f;
            entries[i].cluster.numCellPointers = 0;
            entries[i].cluster.numTokens = 0;
            entries[i].cluster.decompositionRequired = false;
            entries[i].cluster.clusterToFuse = nullptr;
            entries[i].cluster.locked = 0;
        }
    }
    __syncthreads();
    for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
        Cell* cell = _origCluster->cellPointers[cellIndex];
        if (!cell->alive) {
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

            entries[MAX_DECOMPOSITIONS - 1].cluster.decompositionRequired = true;
        }
    }
    __syncthreads();

    int startDecompositionIndex;
    int endDecompositionIndex;
    __shared__ Cluster* newClusters[MAX_DECOMPOSITIONS];
    calcPartition(numDecompositions, threadIdx.x, blockDim.x, startDecompositionIndex, endDecompositionIndex);
    for (int index = startDecompositionIndex; index <= endDecompositionIndex; ++index) {
        auto numCells = entries[index].cluster.numCellPointers;
        entries[index].cluster.pos.x /= numCells;
        entries[index].cluster.pos.y /= numCells;
        entries[index].cluster.vel.x /= numCells;
        entries[index].cluster.vel.y /= numCells;
        entries[index].cluster.cellPointers = _data->cellPointers.getNewSubarray(numCells);
        entries[index].cluster.numCellPointers = 0;
        entries[index].cluster.maxCellPointers = numCells * CELL_POINTER_CAPACITY_MULTIPLIER;

        newClusters[index] = _data->clustersNew.getNewElement();
        *newClusters[index] = entries[index].cluster;
    }
    __syncthreads();

    for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
        Cell* cell = _origCluster->cellPointers[cellIndex];
        for (int index = 0; index < numDecompositions; ++index) {
            if (cell->tag == entries[index].tag) {
                Cluster* newCluster = newClusters[index];
                float2 deltaPos = Math::sub(cell->absPos, newCluster->pos);
                _cellMap.mapDisplacementCorrection(deltaPos);
                auto angularMass = deltaPos.x * deltaPos.x + deltaPos.y * deltaPos.y;
                atomicAdd(&newCluster->angularMass, angularMass);

                auto r = Math::sub(cell->absPos, _origCluster->pos);
                _cellMap.mapDisplacementCorrection(r);
                float2 relVel = Math::sub(cell->vel, newCluster->vel);
                float angularMomentum = Physics::angularMomentum(r, relVel);
                atomicAdd(&newCluster->angularVel, angularMomentum);

                float(&invRotMatrix)[2][2] = entries[index].invRotMatrix;
                cell->relPos.x = deltaPos.x*invRotMatrix[0][0] + deltaPos.y*invRotMatrix[0][1];
                cell->relPos.y = deltaPos.x*invRotMatrix[1][0] + deltaPos.y*invRotMatrix[1][1];

                int newCellIndex = atomicAdd(&newCluster->numCellPointers, 1);
                Cell*& newCellPointer = newCluster->cellPointers[newCellIndex];
                newCellPointer = cell;
                newCellPointer->cluster = newCluster;
            }
        }
    }
    __syncthreads();

    for (int index = startDecompositionIndex; index <= endDecompositionIndex; ++index) {
        Cluster* newCluster = newClusters[index];

        //newCluster->angularVel contains angular momentum until here
        newCluster->angularVel = Physics::angularVelocity(newCluster->angularVel, newCluster->angularMass);
    }
    for (int index = 0; index < numDecompositions; ++index) {
        Cluster* newCluster = newClusters[index];
        copyTokens_blockCall(_origCluster, newCluster);
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::copyCluster_blockCall()
{
    __shared__ Cluster* newCluster;

    if (threadIdx.x == 0) {
        newCluster = _data->clustersNew.getNewElement();
        *newCluster = *_origCluster;
    }
    __syncthreads();

    for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
        auto *cell = newCluster->cellPointers[cellIndex];
        cell->cluster = newCluster;
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::copyClusterWithFusion_blockCall()
{
    if (_origCluster < _origCluster->clusterToFuse) {
        __shared__ Cluster* newCluster;
        __shared__ Cluster* otherCluster;
        __shared__ float2 correction;
        __shared__ bool copyCellPointers;
        __shared__ bool copyOtherCellPointers;
        __shared__ int offset;
        if (0 == threadIdx.x) {
            otherCluster = _origCluster->clusterToFuse;
            newCluster = _data->clustersNew.getNewElement();
            newCluster->id = _origCluster->id;
            newCluster->angle = 0.0f;
            newCluster->numTokens = 0;
            newCluster->numCellPointers = _origCluster->numCellPointers + otherCluster->numCellPointers;
            newCluster->decompositionRequired = _origCluster->decompositionRequired || otherCluster->decompositionRequired;
            newCluster->locked = 0;
            newCluster->clusterToFuse = nullptr;

            copyCellPointers = (otherCluster->maxCellPointers - otherCluster->numCellPointers) >=
                _origCluster->numCellPointers;
            copyOtherCellPointers = (_origCluster->maxCellPointers - _origCluster->numCellPointers) >=
                otherCluster->numCellPointers;
            if (copyOtherCellPointers) {
                newCluster->cellPointers = _origCluster->cellPointers;
                offset = _origCluster->numCellPointers;
            }
            else if (copyCellPointers) {
                newCluster->cellPointers = otherCluster->cellPointers;
                offset = otherCluster->numCellPointers;
            }
            else {
                newCluster->maxCellPointers = newCluster->numCellPointers
                    * CELL_POINTER_CAPACITY_MULTIPLIER;
                newCluster->cellPointers = _data->cellPointers.getNewSubarray(newCluster->maxCellPointers);
                offset = 0;
            }

            correction = _cellMap.correctionIncrement(_origCluster->pos, otherCluster->pos);	//to be added to otherCluster

            newCluster->pos =
                Math::div(Math::add(Math::mul(_origCluster->pos, _origCluster->numCellPointers), Math::mul(Math::add(otherCluster->pos, correction), otherCluster->numCellPointers)),
                    newCluster->numCellPointers);
            newCluster->vel = Math::div(
                Math::add(Math::mul(_origCluster->vel, _origCluster->numCellPointers), Math::mul(otherCluster->vel, otherCluster->numCellPointers)), newCluster->numCellPointers);
            newCluster->angularVel = 0.0f;
            newCluster->angularMass = 0.0f;
        }
        __syncthreads();

        for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
            Cell* cell = _origCluster->cellPointers[cellIndex];
            if (copyCellPointers) {
                Cell*& newCellPointer = newCluster->cellPointers[offset + cellIndex];
                newCellPointer = cell;
            }
            cell->cluster = newCluster;
            auto relPos = Math::sub(cell->absPos, newCluster->pos);
            cell->relPos = relPos;
            cell->cluster = newCluster;
            atomicAdd(&newCluster->angularMass, Math::lengthSquared(relPos));

            auto r = Math::sub(cell->absPos, _origCluster->pos);
            _cellMap.mapDisplacementCorrection(r);
            float2 relVel = Math::sub(cell->vel, _origCluster->vel);
            float angularMomentum = Physics::angularMomentum(r, relVel);
            atomicAdd(&newCluster->angularVel, angularMomentum);
        }
        __syncthreads();
       
        if (0 == threadIdx.x && copyCellPointers) {
            offset += _origCluster->numCellPointers;
        }
        __syncthreads();

        int startOtherCellIndex;
        int endOtherCellIndex;
        calcPartition(otherCluster->numCellPointers, threadIdx.x, blockDim.x, startOtherCellIndex, endOtherCellIndex);

        for (int otherCellIndex = startOtherCellIndex; otherCellIndex <= endOtherCellIndex; ++otherCellIndex) {
            Cell* cell = otherCluster->cellPointers[otherCellIndex];
            if (copyOtherCellPointers) {
                Cell*& newCellPointer = newCluster->cellPointers[offset + otherCellIndex];
                newCellPointer = cell;
            }
            cell->cluster = newCluster;
            auto r = Math::sub(cell->absPos, otherCluster->pos);
            _cellMap.mapDisplacementCorrection(r);

            cell->absPos = Math::add(cell->absPos, correction);
            auto relPos = Math::sub(cell->absPos, newCluster->pos);
            cell->relPos = relPos;
            cell->cluster = newCluster;
            atomicAdd(&newCluster->angularMass, Math::lengthSquared(relPos));

            float2 relVel = Math::sub(cell->vel, otherCluster->vel);
            float angularMomentum = Physics::angularMomentum(r, relVel);
            atomicAdd(&newCluster->angularVel, angularMomentum);
        }
        __syncthreads();

        if (0 == threadIdx.x) {
            newCluster->angularVel = Physics::angularVelocity(newCluster->angularVel, newCluster->angularMass);
        }

        copyTokens_blockCall(_origCluster, _origCluster->clusterToFuse, newCluster);
    }
    else {
        //do not copy anything
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::copyTokens_blockCall(Cluster* sourceCluster, Cluster* targetCluster)
{
    __shared__ int numberOfTokensToCopy;
    __shared__ int tokenCopyIndex;
    if (0 == threadIdx.x) {
        numberOfTokensToCopy = 0;
        tokenCopyIndex = 0;
    }
    __syncthreads();

    getNumberOfTokensToCopy_blockCall(sourceCluster, targetCluster, numberOfTokensToCopy);

    if (0 == threadIdx.x) {
        targetCluster->tokens = _data->tokensNew.getNewSubarray(numberOfTokensToCopy);
        targetCluster->numTokens = numberOfTokensToCopy;
    }
    __syncthreads();

    int startTokenIndex;
    int endTokenIndex;
    calcPartition(sourceCluster->numTokens, threadIdx.x, blockDim.x, startTokenIndex, endTokenIndex);
    for (int tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
        auto& token = sourceCluster->tokens[tokenIndex];
        auto& cell = token.cell;
        if (!cell->alive) {
            continue;
        }
        if (cell->cluster == targetCluster) {
            int prevTokenCopyIndex = atomicAdd(&tokenCopyIndex, 1);
            auto& targetToken = targetCluster->tokens[prevTokenCopyIndex];
            targetToken = token;
            targetToken.cell = cell;
        }
    }
    __syncthreads();
}

__inline__ __device__ void
ClusterProcessorOnCopyData::copyTokens_blockCall(Cluster* sourceCluster1, Cluster* sourceCluster2, Cluster* targetCluster)
{
    __shared__ int numberOfTokensToCopy;
    __shared__ int tokenCopyIndex;
    if (0 == threadIdx.x) {
        numberOfTokensToCopy = 0;
        tokenCopyIndex = 0;
    }
    __syncthreads();

    getNumberOfTokensToCopy_blockCall(sourceCluster1, targetCluster, numberOfTokensToCopy);
    getNumberOfTokensToCopy_blockCall(sourceCluster2, targetCluster, numberOfTokensToCopy);

    if (0 == threadIdx.x) {
        targetCluster->tokens = _data->tokensNew.getNewSubarray(numberOfTokensToCopy);
        targetCluster->numTokens = numberOfTokensToCopy;
    }
    __syncthreads();

    int startTokenIndex;
    int endTokenIndex;
    calcPartition(sourceCluster1->numTokens, threadIdx.x, blockDim.x, startTokenIndex, endTokenIndex);
    for (int tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
        auto& token = sourceCluster1->tokens[tokenIndex];
        auto& cell = token.cell;
        if (!cell->alive) {
            continue;
        }
        if (cell->cluster == targetCluster) {
            int prevTokenCopyIndex = atomicAdd(&tokenCopyIndex, 1);
            auto& targetToken = targetCluster->tokens[prevTokenCopyIndex];
            targetToken = token;
            targetToken.cell = cell;
        }
    }
    __syncthreads();

    calcPartition(sourceCluster2->numTokens, threadIdx.x, blockDim.x, startTokenIndex, endTokenIndex);
    for (int tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
        auto& token = sourceCluster2->tokens[tokenIndex];
        auto& cell = token.cell;
        if (!cell->alive) {
            continue;
        }
        if (cell->cluster == targetCluster) {
            int prevTokenCopyIndex = atomicAdd(&tokenCopyIndex, 1);
            auto& targetToken = targetCluster->tokens[prevTokenCopyIndex];
            targetToken = token;
            targetToken.cell = cell;
        }
    }
    __syncthreads();

}

__inline__ __device__ void
ClusterProcessorOnCopyData::getNumberOfTokensToCopy_blockCall(Cluster* sourceCluster, Cluster* targetCluster, int& counter)
{
    int startTokenIndex, endTokenIndex;
    calcPartition(sourceCluster->numTokens, threadIdx.x, blockDim.x, startTokenIndex, endTokenIndex);

    for (int tokenIndex = startTokenIndex; tokenIndex <= endTokenIndex; ++tokenIndex) {
        auto const& token = sourceCluster->tokens[tokenIndex];
        auto const& cell = token.cell;
        if (!cell->alive) {
            continue;
        }
        if (cell->cluster == targetCluster) {
            atomicAdd(&counter, 1);
        }
    }
    __syncthreads();
}

__inline__ __device__ void ClusterProcessorOnCopyData::processingClusterCopy_blockCall()
{
    if (_origCluster->numCellPointers == 1 && !_origCluster->cellPointers[0]->alive && !_origCluster->clusterToFuse) {
        __syncthreads();
        return;
    }

    if (_origCluster->decompositionRequired && !_origCluster->clusterToFuse) {
        copyClusterWithDecomposition_blockCall();
    }
    else if (_origCluster->clusterToFuse) {
        copyClusterWithFusion_blockCall();
    }
    else {
        copyCluster_blockCall();
    }
}

__inline__ __device__ void ClusterProcessorOnCopyData::processingDecomposition_blockCall()
{
    if (_origCluster->decompositionRequired && !_origCluster->clusterToFuse) {
        __shared__ bool changes;

        for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
            _origCluster->cellPointers[cellIndex]->tag = cellIndex;
        }
        do {
            changes = false;
            __syncthreads();
            for (int cellIndex = _startCellIndex; cellIndex <= _endCellIndex; ++cellIndex) {
                Cell* cell = _origCluster->cellPointers[cellIndex];
                if (cell->alive) {
                    for (int i = 0; i < cell->numConnections; ++i) {
                        Cell& otherCell = *cell->connections[i];
                        if (otherCell.alive) {
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
}

