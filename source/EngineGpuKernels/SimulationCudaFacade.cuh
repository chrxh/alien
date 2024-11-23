#pragma once

#include <cstdint>
#include <mutex>
#include <vector>
#include <optional>

#if defined(_WIN32)
#include <windows.h>
#endif

#include <vector_types.h>
#include <GL/gl.h>

#include "EngineInterface/RawStatisticsData.h"
#include "EngineInterface/Settings.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/MutationType.h"
#include "EngineInterface/StatisticsHistory.h"
#include "EngineInterface/SimulationParametersUpdateConfig.h"

#include "Definitions.cuh"

struct cudaGraphicsResource;

class _SimulationCudaFacade
{
public:
    struct GpuInfo
    {
        int deviceNumber = 0;
        std::string gpuModelName;
    };
    static GpuInfo checkAndReturnGpuInfo();

    _SimulationCudaFacade(uint64_t timestep, Settings const& settings);
    ~_SimulationCudaFacade();

    void* registerImageResource(GLuint image);

    void calcTimestep(uint64_t timesteps, bool forceUpdateStatistics);
    void applyCataclysm(int power);

    void drawVectorGraphics(float2 const& rectUpperLeft, float2 const& rectLowerRight, void* cudaResource, int2 const& imageSize, double zoom);
    void getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataTO const& dataTO);
    void getSelectedSimulationData(bool includeClusters, DataTO const& dataTO);
    void getInspectedSimulationData(std::vector<uint64_t> entityIds, DataTO const& dataTO);
    void getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataTO const& dataTO);
    void addAndSelectSimulationData(DataTO const& dataTO);
    void setSimulationData(DataTO const& dataTO);
    void removeSelectedObjects(bool includeClusters);
    void relaxSelectedObjects(bool includeClusters);
    void uniformVelocitiesForSelectedObjects(bool includeClusters);
    void makeSticky(bool includeClusters);
    void removeStickiness(bool includeClusters);
    void setBarrier(bool value, bool includeClusters);
    void changeInspectedSimulationData(DataTO const& changeDataTO);

    void applyForce(ApplyForceData const& applyData);
    void switchSelection(PointSelectionData const& switchData);
    void swapSelection(PointSelectionData const& selectionData);
    void setSelection(AreaSelectionData const& selectionData);
    SelectionShallowData getSelectionShallowData();
    void shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& shallowUpdateData);
    void removeSelection();
    void updateSelection();
    void colorSelectedObjects(unsigned char color, bool includeClusters);
    void reconnectSelectedObjects();
    void setDetached(bool value);

    void setGpuConstants(GpuSettings const& cudaConstants);
    SimulationParameters getSimulationParameters() const;
    void setSimulationParameters(
        SimulationParameters const& parameters,
        SimulationParametersUpdateConfig const& updateConfig = SimulationParametersUpdateConfig::All);

    ArraySizes getArraySizes() const;

    RawStatisticsData getRawStatistics();
    void updateStatistics();
    StatisticsHistory const& getStatisticsHistory() const;
    void setStatisticsHistory(StatisticsHistoryData const& data);

    void resetTimeIntervalStatistics();
    uint64_t getCurrentTimestep() const;
    void setCurrentTimestep(uint64_t timestep);

    void clear();

    void resizeArraysIfNecessary(ArraySizes const& additionals = ArraySizes());

    //for tests
    void testOnly_mutate(uint64_t cellId, MutationType mutationType);

private:
    void initCuda();

    void syncAndCheck();
    void copyDataTOtoDevice(DataTO const& dataTO);
    void copyDataTOtoHost(DataTO const& dataTO);
    void automaticResizeArrays();
    void resizeArrays(ArraySizes const& additionals = ArraySizes());
    void checkAndProcessSimulationParameterChanges();

    SimulationData getSimulationDataIntern() const;

    GpuInfo _gpuInfo;
    cudaGraphicsResource* _cudaResource = nullptr;

    mutable std::mutex _mutexForSimulationParameters;
    std::optional<SimulationParameters> _newSimulationParameters;
    SimulationParametersUpdateConfig _simulationParametersUpdateConfig = SimulationParametersUpdateConfig::All;

    Settings _settings;

    mutable std::mutex _mutexForSimulationData;
    std::shared_ptr<SimulationData> _cudaSimulationData;

    std::shared_ptr<RenderingData> _cudaRenderingData;
    std::shared_ptr<SelectionResult> _cudaSelectionResult;
    std::shared_ptr<DataTO> _cudaAccessTO;

    mutable std::mutex _mutexForStatistics;
    std::optional<std::chrono::steady_clock::time_point> _lastStatisticsUpdateTime;
    std::optional<RawStatisticsData> _statisticsData;
    StatisticsHistory _statisticsHistory;
    std::shared_ptr<SimulationStatistics> _cudaSimulationStatistics;
    MaxAgeBalancer _maxAgeBalancer;

    SimulationKernelsLauncher _simulationKernels;
    DataAccessKernelsLauncher _dataAccessKernels;
    GarbageCollectorKernelsLauncher _garbageCollectorKernels;
    RenderingKernelsLauncher _renderingKernels;
    EditKernelsLauncher _editKernels;
    StatisticsKernelsLauncher _statisticsKernels;
    TestKernelsLauncher _testKernels;
};
