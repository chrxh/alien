#pragma once

#include <cstdint>
#include <atomic>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/gl.h>

#include "EngineInterface/OverallStatistics.h"
#include "EngineInterface/Settings.h"
#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"

#include "Definitions.cuh"
#include "DllExport.h"

class _CudaSimulationAdapter
{
public:
    ENGINEGPUKERNELS_EXPORT static void initCuda();

    ENGINEGPUKERNELS_EXPORT
    _CudaSimulationAdapter(uint64_t timestep, Settings const& settings, GpuSettings const& gpuSettings);
    ENGINEGPUKERNELS_EXPORT ~_CudaSimulationAdapter();

    ENGINEGPUKERNELS_EXPORT void* registerImageResource(GLuint image);

    ENGINEGPUKERNELS_EXPORT void calcTimestep();

    ENGINEGPUKERNELS_EXPORT void drawVectorGraphics(
        float2 const& rectUpperLeft,
        float2 const& rectLowerRight,
        void* cudaResource,
        int2 const& imageSize,
        double zoom);
    ENGINEGPUKERNELS_EXPORT void
    getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
    ENGINEGPUKERNELS_EXPORT void getSelectedSimulationData(bool includeClusters, DataAccessTO const& dataTO);
    ENGINEGPUKERNELS_EXPORT void getInspectedSimulationData(
        std::vector<uint64_t> entityIds,
        DataAccessTO const& dataTO);
    ENGINEGPUKERNELS_EXPORT void
    getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
    ENGINEGPUKERNELS_EXPORT void addAndSelectSimulationData(DataAccessTO const& dataTO);
    ENGINEGPUKERNELS_EXPORT void setSimulationData(DataAccessTO const& dataTO);
    ENGINEGPUKERNELS_EXPORT void removeSelectedEntities(bool includeClusters);
    ENGINEGPUKERNELS_EXPORT void changeInspectedSimulationData(DataAccessTO const& changeDataTO);

    ENGINEGPUKERNELS_EXPORT void applyForce(ApplyForceData const& applyData);
    ENGINEGPUKERNELS_EXPORT void switchSelection(PointSelectionData const& switchData);
    ENGINEGPUKERNELS_EXPORT void swapSelection(PointSelectionData const& selectionData);
    ENGINEGPUKERNELS_EXPORT void setSelection(AreaSelectionData const& selectionData);
    ENGINEGPUKERNELS_EXPORT SelectionShallowData getSelectionShallowData();
    ENGINEGPUKERNELS_EXPORT void shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& shallowUpdateData);
    ENGINEGPUKERNELS_EXPORT void removeSelection();
    ENGINEGPUKERNELS_EXPORT void updateSelection();
    ENGINEGPUKERNELS_EXPORT void colorSelectedEntities(unsigned char color, bool includeClusters);

    ENGINEGPUKERNELS_EXPORT void setGpuConstants(GpuSettings const& cudaConstants);
    ENGINEGPUKERNELS_EXPORT void setSimulationParameters(SimulationParameters const& parameters);
    ENGINEGPUKERNELS_EXPORT void setSimulationParametersSpots(SimulationParametersSpots const& spots);
    ENGINEGPUKERNELS_EXPORT void setFlowFieldSettings(FlowFieldSettings const& settings);

    ENGINEGPUKERNELS_EXPORT ArraySizes getArraySizes() const;

    ENGINEGPUKERNELS_EXPORT OverallStatistics getMonitorData();
    ENGINEGPUKERNELS_EXPORT uint64_t getCurrentTimestep() const;
    ENGINEGPUKERNELS_EXPORT void setCurrentTimestep(uint64_t timestep);

    ENGINEGPUKERNELS_EXPORT void clear();

    ENGINEGPUKERNELS_EXPORT void resizeArraysIfNecessary(ArraySizes const& additionals);

private:
    void copyDataTOtoDevice(DataAccessTO const& dataTO);
    void copyDataTOtoHost(DataAccessTO const& dataTO);
    void automaticResizeArrays();
    void resizeArrays(ArraySizes const& additionals);

    std::atomic<uint64_t> _currentTimestep;
    GpuSettings _gpuSettings;
    SimulationData* _cudaSimulationData;
    RenderingData* _cudaRenderingData;
    SimulationResult* _cudaSimulationResult;
    SelectionResult* _cudaSelectionResult;
    DataAccessTO* _cudaAccessTO;
    CudaMonitorData* _cudaMonitorData;

    SimulationKernelLauncher* _simulationKernels;
    DataAccessKernelLauncher* _dataAccessKernels;
};
