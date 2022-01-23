#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/InspectedEntityIds.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Macros.cuh"

class _DataAccessKernelsLauncher
{
public:
    _DataAccessKernelsLauncher();

    void getData(GpuSettings const& gpuSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
    void getSelectedData(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters, DataAccessTO const& dataTO);
    void getInspectedData(GpuSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, DataAccessTO const& dataTO);
    void getOverlayData(GpuSettings const& gpuSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, DataAccessTO const& dataTO);

    void addData(GpuSettings const& gpuSettings, SimulationData const& data, DataAccessTO const& dataTO, bool selectData, bool createIds);
    void clearData(GpuSettings const& gpuSettings, SimulationData const& data);

private:
    GarbageCollectorKernelsLauncher _garbageCollectorKernels;
    EditKernelsLauncher _editKernels;
};

