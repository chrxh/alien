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
        _timestepsUntilFreeze = 30;
        _freezed = false;
    }

    __device__ __inline__ bool isFreezed()
    {
        return _freezed
            && numTokenPointers == 0
            && decompositionRequired == 0
            && clusterToFuse == nullptr;
    }

    __device__ __inline__ void setUnfreezed(int forTimesteps = 0)
    {
        _timestepsUntilFreeze = forTimesteps;
        _freezed = false;
    }

    __device__ __inline__ void timestepSimulated()
    {
        if (_timestepsUntilFreeze > 0) {
            --_timestepsUntilFreeze;
        }
        else {
            _freezed = true;
        }
    }

    __device__ __inline__ void tagCellByIndex_blockCall(PartitionData const& blockData)
    {
        for (auto cellIndex = blockData.startIndex; cellIndex <= blockData.endIndex; ++cellIndex) {
            Cell& cell = *cellPointers[cellIndex];
            cell.tag = cellIndex;
        }
        __syncthreads();
    }
private:
    bool _freezed;
    int _timestepsUntilFreeze;
};
