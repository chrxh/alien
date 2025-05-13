﻿#pragma once

#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/InspectedEntityIds.h"

#include "Base.cuh"
#include "Definitions.cuh"
#include "Macros.cuh"

class _DataAccessKernelsService
{
public:
    _DataAccessKernelsService();
    ~_DataAccessKernelsService();

    void getData(GpuSettings const& gpuSettings, SimulationData const& data, int2 const& rectUpperLeft, int2 const& rectLowerRight, DataTO const& dataTO);
    void getSelectedData(GpuSettings const& gpuSettings, SimulationData const& data, bool includeClusters, DataTO const& dataTO);
    void getInspectedData(GpuSettings const& gpuSettings, SimulationData const& data, InspectedEntityIds entityIds, DataTO const& dataTO);
    void getOverlayData(GpuSettings const& gpuSettings, SimulationData const& data, int2 rectUpperLeft, int2 rectLowerRight, DataTO const& dataTO);

    void addData(GpuSettings const& gpuSettings, SimulationData const& data, DataTO const& dataTO, bool selectData, bool createIds);
    void clearData(GpuSettings const& gpuSettings, SimulationData const& data);

private:
    GarbageCollectorKernelsService _garbageCollectorKernels;
    EditKernelsService _editKernels;

    // gpu memory
    Cell** _cudaCellArray;
};

