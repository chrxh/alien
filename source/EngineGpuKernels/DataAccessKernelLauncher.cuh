#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "DataAccessKernels.cuh"
#include "Macros.cuh"
#include "GarbageCollectorKernelLauncher.cuh"

class DataAccessKernelLauncher
{
public:
    void getData(
        GpuSettings const& gpuSettings,
        SimulationData const& simulationData,
        int2 const& rectUpperLeft,
        int2 const& rectLowerRight,
        DataAccessTO const& dataTO);

    void addData(GpuSettings const& gpuSettings, SimulationData data, DataAccessTO dataTO, bool selectData);
    void clearData(GpuSettings const& gpuSettings, SimulationData data);

private:
    GarbageCollectorKernelLauncher _garbageCollector;
};

