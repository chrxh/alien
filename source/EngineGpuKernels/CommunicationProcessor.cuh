#pragma once

#include "SimulationData.cuh"
#include "QuantityConverter.cuh"

class CommunicationProcessor
{
public:
    __inline__ __device__ static void process(Token* token, SimulationData& data);

};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void CommunicationProcessor::process(Token* token, SimulationData& data)
{
    //#TODO
} 