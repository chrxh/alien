#pragma once

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "CudaInterface.cuh"
#include "Base.cuh"
#include "Physics.cuh"
#include "Map.cuh"

class TokenProcessorOnOrigData
{
public:
	__inline__ __device__ void init(SimulationData& data);

	__inline__ __device__ void processingEnergyGuidance();

private:

	SimulationData* _data;
	Token *_cluster;

	int _startTokenIndex;
	int _endTokenIndex;
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void TokenProcessorOnOrigData::init(SimulationData & data)
{
}

__inline__ __device__ void TokenProcessorOnOrigData::processingEnergyGuidance()
{
}
