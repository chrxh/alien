#pragma once

#include "Base.cuh"
#include "Cell.cuh"

struct ClusterMetadata
{
    int nameLen;
    char* name;
};

struct Cluster
{
    uint64_t id;
    float2 pos;
    float2 vel;
    float angle;
    float angularVel;
    float angularMass;
    int numCellPointers;
    Cell** cellPointers;
    int numTokenPointers;
    Token** tokenPointers;
    ClusterMetadata metadata;

    //auxiliary data
    int decompositionRequired;  //0 = false, 1 = true
    int locked;	//0 = unlocked, 1 = locked
    Cluster* clusterToFuse;

    __device__ __inline__ void init()
    {
        _timestepsUntilFreezing = 30;
        _freezed = false;
        _pointerArrayElement = nullptr;
    }

    __device__ __inline__ bool isFreezed()
    {
        return _freezed;
    }

    __device__ __inline__ void freeze(Cluster** pointerArrayElement)
    {
        _freezed = true;
        _pointerArrayElement = pointerArrayElement;
    }

    __device__ __inline__ Cluster** getPointerArrayElement()
    {
        return _pointerArrayElement;
    }

    __device__ __inline__ void unfreeze(int timesteps = 0)
    {
        _timestepsUntilFreezing = timesteps;
        _freezed = false;
    }

    __device__ __inline__ bool isCandidateToFreeze()
    {
        return _timestepsUntilFreezing == 0
            && numTokenPointers == 0
            && decompositionRequired == 0
            && clusterToFuse == nullptr;
    }

    __device__ __inline__ void timestepSimulated()
    {
        if (_timestepsUntilFreezing > 0) {
            --_timestepsUntilFreezing;
        }
    }

    __device__ __inline__ void tagCellByIndex_block(PartitionData const& blockData)
    {
        for (auto cellIndex = blockData.startIndex; cellIndex <= blockData.endIndex; ++cellIndex) {
            Cell& cell = *cellPointers[cellIndex];
            cell.tag = cellIndex;
        }
        __syncthreads();
    }
private:
    bool _freezed;
    int _timestepsUntilFreezing;
    Cluster** _pointerArrayElement;
};
