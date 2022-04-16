#pragma once

#include <cstdint>
#include <atomic>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/gl.h>

#include "EngineInterface/MonitorData.h"
#include "EngineInterface/Settings.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"

#include "Definitions.cuh"

class _CudaSimulationFacade
{
public:
    static void initCuda();

    _CudaSimulationFacade(uint64_t timestep, Settings const& settings);
    ~_CudaSimulationFacade();

    void* registerImageResource(GLuint image);

    void calcTimestep();

    void drawVectorGraphics(float2 const& rectUpperLeft, float2 const& rectLowerRight, void* cudaResource, int2 const& imageSize, double zoom);
    void getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
    void getSelectedSimulationData(bool includeClusters, DataAccessTO const& dataTO);
    void getInspectedSimulationData(std::vector<uint64_t> entityIds, DataAccessTO const& dataTO);
    void getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO);
    void addAndSelectSimulationData(DataAccessTO const& dataTO);
    void setSimulationData(DataAccessTO const& dataTO);
    void removeSelectedEntities(bool includeClusters);
    void relaxSelectedEntities(bool includeClusters);
    void uniformVelocitiesForSelectedEntities(bool includeClusters);
    void makeSticky(bool includeClusters);
    void removeStickiness(bool includeClusters);
    void setBarrier(bool value, bool includeClusters);
    void changeInspectedSimulationData(DataAccessTO const& changeDataTO);

    void applyForce(ApplyForceData const& applyData);
    void switchSelection(PointSelectionData const& switchData);
    void swapSelection(PointSelectionData const& selectionData);
    void setSelection(AreaSelectionData const& selectionData);
    SelectionShallowData getSelectionShallowData();
    void shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& shallowUpdateData);
    void removeSelection();
    void updateSelection();
    void colorSelectedEntities(unsigned char color, bool includeClusters);
    void reconnectSelectedEntities();

    void setGpuConstants(GpuSettings const& cudaConstants);
    void setSimulationParameters(SimulationParameters const& parameters);
    void setSimulationParametersSpots(SimulationParametersSpots const& spots);
    void setFlowFieldSettings(FlowFieldSettings const& settings);

    ArraySizes getArraySizes() const;

    MonitorData getMonitorData();
    uint64_t getCurrentTimestep() const;
    void setCurrentTimestep(uint64_t timestep);

    void clear();

    void resizeArraysIfNecessary(ArraySizes const& additionals);

private:
    void syncAndCheck();
    void copyDataTOtoDevice(DataAccessTO const& dataTO);
    void copyDataTOtoHost(DataAccessTO const& dataTO);
    void automaticResizeArrays();
    void resizeArrays(ArraySizes const& additionals);

    std::atomic<uint64_t> _currentTimestep;
    Settings _settings;

    std::shared_ptr<SimulationData> _cudaSimulationData;
    std::shared_ptr<RenderingData> _cudaRenderingData;
    std::shared_ptr<SimulationResult> _cudaSimulationResult;
    std::shared_ptr<SelectionResult> _cudaSelectionResult;
    std::shared_ptr<DataAccessTO> _cudaAccessTO;
    std::shared_ptr<CudaMonitorData> _cudaMonitorData;

    SimulationKernelsLauncher _simulationKernels;
    DataAccessKernelsLauncher _dataAccessKernels;
    GarbageCollectorKernelsLauncher _garbageCollectorKernels;
    RenderingKernelsLauncher _renderingKernels;
    EditKernelsLauncher _editKernels;
    MonitorKernelsLauncher _monitorKernels;
};
