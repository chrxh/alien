#pragma once

#include "Definitions.cuh"

#include "Base.cuh"
#include "Array.cuh"
#include "List.cuh"

class MapSectionCollector
{
public:
    __host__ __inline__ void
    init(int2 const& universeSize, int sectionSize)
    {
        _numSections = { universeSize.x / sectionSize, universeSize.y / sectionSize };
        _sectionSize = sectionSize;
/*
        _clusterListBySectionIndex.init(_numSections.x *_numSections.y);
*/
    }

    __host__ __inline__ void free()
    {
/*
        _clusterListBySectionIndex.free();
*/
    }

    __device__ __inline__ void reset_system()
    {
/*
        auto const partition = calcPartition(
            _numSections.x * _numSections.y, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
        for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
            _clusterListBySectionIndex.at(index).init();
        }
*/
    }

    __device__ __inline__ void insert(Cluster* cluster, DynamicMemory* dynamicMemory)
    {
/*
        auto const section = getSection(cluster->pos);

        auto& clusterList = _clusterListBySectionIndex.at(section.x + section.y * _numSections.x);
        clusterList.pushBack(cluster, dynamicMemory);
*/
    }

    __device__ __inline__ void getClusters_block(float2 const& pos, float radius, MapInfo const& map, 
        DynamicMemory* dynamicMemory, List<Cluster*>& result)
    {
/*
        __shared__ int2 sectionCenter;
        __shared__ int sectionLength;
        if (0 == threadIdx.x) {
            sectionCenter = getSection(pos);
            sectionLength = floorInt(radius) / _sectionSize + 1;
            result.init();
        }
        __syncthreads();

        int2 section;
        for (section.x = sectionCenter.x - sectionLength; section.x <= sectionCenter.x + sectionLength; ++section.x) {
            for (section.y = sectionCenter.y - sectionLength; section.y <= sectionCenter.y + sectionLength; ++section.y) {
                __shared__ int numClusters;
                __shared__ Cluster** clusterArray;
                if (0 == threadIdx.x) {
                    auto const& clusterList = getClusters(section);
                    numClusters = clusterList.getSize();
                    clusterArray = clusterList.asArray(dynamicMemory);
                }
                __syncthreads();

                if (numClusters > 0) {
                    auto const partition = calcPartition(numClusters, threadIdx.x, blockDim.x);
                    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
                        auto const& cluster = clusterArray[index];
                        auto const distance = map.mapDistance(cluster->pos, pos);
                        if (distance < radius) {
                            result.pushBack(cluster, dynamicMemory);
                        }
                    }
                }
                __syncthreads();
            }
        }
*/
    }

private:
    __device__ __inline__ int2 getSection(float2 const& pos)
    {
        auto const intPos = toInt2(pos);
        auto section = int2{ intPos.x / _sectionSize, intPos.y / _sectionSize };
        correctSection(section);
        return section;

    }

/*
    __device__ __inline__ List<Cluster*> const& getClusters(int2 section)
    {
        correctSection(section);
        return _clusterListBySectionIndex.at(section.x + section.y * _numSections.x);
    }
*/

    __device__ __inline__ void correctSection(int2& section)
    {
        section.x = ((section.x % _numSections.x) + _numSections.x) % _numSections.x;
        section.y = ((section.y % _numSections.y) + _numSections.y) % _numSections.y;
    }

private:
    int2 _numSections;
    int _sectionSize;
//    Array<List<Cluster*>> _clusterListBySectionIndex;
};
