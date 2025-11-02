#pragma once

#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/MutationType.h>

#include "Definitions.cuh"

class _TestKernelsService
{
public:
    _TestKernelsService();
    ~_TestKernelsService();

    void testOnly_mutate(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t cellId, MutationType mutationType);
    void testOnly_mutationCheck(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t cellId);
    void testOnly_createConnection(CudaSettings const& gpuSettings, SimulationData const& data, uint64_t cellId1, uint64_t cellId2);
    bool testOnly_areArraysValid(CudaSettings const& gpuSettings, SimulationData const& data);

private:
    bool* _cudaBoolResult;
};
