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
        _freezed = 0;
        _pointerArrayElement = nullptr;
        _selected = false;
    }

    __device__ __inline__ bool isFreezed()
    {
        return _freezed;
    }

    __device__ __inline__ void freeze(Cluster** pointerArrayElement)
    {
        atomicExch(&_freezed, 1);
        _pointerArrayElement = pointerArrayElement;
    }

    __device__ __inline__ Cluster** getPointerArrayElement()
    {
        return _pointerArrayElement;
    }

    __device__ __inline__ void unfreeze(int timesteps = 0)
    {
        atomicExch(&_timestepsUntilFreezing, timesteps);
        atomicExch(&_freezed, 0);
    }

    __device__ __inline__ bool isActive()
    {
        return numTokenPointers > 0;
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

    __device__ __inline__ int getMass()
    {
        return static_cast<float>(numCellPointers);
    }

    __device__ __inline__ bool isSelected()
    {
        return _selected;
    }

    __device__ __inline__ void setSelected(bool value)
    {
        _selected = value;
    }

private:

    int _freezed;       // 0 = unfreezed, 1 = freezed
    int _timestepsUntilFreezing;
    Cluster** _pointerArrayElement;
    bool _selected;
};
