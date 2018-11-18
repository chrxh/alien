#pragma once

#include <cuda_runtime.h>
#include <vector_types.h>

#include "CudaInterface.cuh"
#include "Base.cuh"
#include "Constants.cuh"
#include "Map.cuh"

struct CollisionEntry
{
	ClusterData* cluster;
	int numCollisions;
	float2 collisionPos;
	float2 normalVec;
};

class CollisionData
{
public:

	int numEntries;
	CollisionEntry entries[MAX_COLLIDING_CLUSTERS];

	__device__ void init()
	{
		numEntries = 0;
	}

	__device__ CollisionEntry* getOrCreateEntry(ClusterData* cluster)
	{
		int old;
		int curr;
		do {
			for (int i = 0; i < numEntries; ++i) {
				if (entries[i].cluster = cluster) {
					return &entries[i];
				}
			}

			old = numEntries;
			curr = atomicCAS(&numEntries, old, (old + 1) % MAX_COLLIDING_CLUSTERS);
			if (old == curr) {
				auto result = &entries[old];
				result->cluster = cluster;
				result->numCollisions = 0;
				result->collisionPos = { 0, 0 };
				result->normalVec = { 0, 0 };
				return result;
			}

		} while (true);
	}
};
