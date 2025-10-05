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

#include "EngineInterface/MutationType.h"
#include "EngineInterface/ArraySizesForGpu.h"
#include "EngineInterface/ArraySizesForTO.h"
#include "EngineInterface/SettingsForSimulation.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/SimulationParametersUpdateConfig.h"
#include "EngineInterface/StatisticsRawData.h"
#include "EngineInterface/StatisticsHistory.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.cuh"
#include "TO.cuh"

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

    _SimulationCudaFacade(uint64_t timestep, SettingsForSimulation const& settings);
    ~_SimulationCudaFacade();

    void calcTimestep(uint64_t timesteps, bool forceUpdateStatistics);
    void applyCataclysm(int power);

    Ids getMaxIds() const;

    void drawVectorGraphics(float2 const& rectUpperLeft, float2 const& rectLowerRight, void* cudaResource, int2 const& imageSize, double zoom);
    // Return number of extracted objects
    NumRenderObjects copyBuffersFromCudaToOpenGL(RenderBuffers const& buffers);
    TO getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight);  // DataTO is unmanaged (i.e. must be deleted by the caller)
    TO getSelectedSimulationData(bool includeClusters);
    TO getInspectedSimulationData(std::vector<uint64_t> entityIds);
    TO getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight);
    void addAndSelectSimulationData(TO const& dataTO);
    void setSimulationData(TO const& dataTO);
    void removeSelectedObjects(bool includeClusters);
    void relaxSelectedObjects(bool includeClusters);
    void uniformVelocitiesForSelectedObjects(bool includeClusters);
    void makeSticky(bool includeClusters);
    void removeStickiness(bool includeClusters);
    void setBarrier(bool value, bool includeClusters);
    void changeInspectedSimulationData(TO const& changeDataTO);
    bool changeCreature(TO const& dataTO);  // dataTO only contains 1 genome

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

    void setGpuConstants(CudaSettings const& cudaConstants);
    SimulationParameters getSimulationParameters() const;
    void setSimulationParameters(
        SimulationParameters const& parameters,
        SimulationParametersUpdateConfig const& updateConfig = SimulationParametersUpdateConfig::All);

    ArraySizesForTO estimateCapacityNeededForTO() const;

    StatisticsRawData getStatisticsRawData();
    void updateStatistics();
    StatisticsHistory const& getStatisticsHistory() const;
    void setStatisticsHistory(StatisticsHistoryData const& data);

    void resetTimeIntervalStatistics();
    uint64_t getCurrentTimestep() const;
    void setCurrentTimestep(uint64_t timestep);

    void clear();

    void resizeArraysIfNecessary(ArraySizesForGpu const& sizeDelta = ArraySizesForGpu());

    // Simulated preview
    void initSettingsPreviewData();
    void newPreview(TO const& dataTO);
    void calcTimestepsForPreview(std::chrono::milliseconds const& duration, bool detailSimulation);
    void calcTimestepsForPreview(int numSteps, bool detailSimulation);
    uint64_t getCurrentTimestepForPreview();
    void setCurrentTimestepForPreview(uint64_t timestep);
    TO getPreviewData();

    // Only for tests
    void testOnly_mutate(uint64_t cellId, MutationType mutationType);
    void testOnly_mutationCheck(uint64_t cellId);
    void testOnly_createConnection(uint64_t cellId1, uint64_t cellId2);
    void testOnly_cleanupAfterTimestep();
    void testOnly_cleanupAfterDataManipulation();
    void testOnly_resizeArrays(ArraySizesForGpu const& sizeDelta);
    bool testOnly_areArraysValid();

private:
    void initCuda();

    void registerRenderBuffers(RenderBuffers const& buffers);

    void syncAndCheck();
    void copyDataTOtoGpu(TO const& cudaDataTO, TO const& dataTO);
    void copyDataTOtoHost(TO const& dataTO, TO const& cudaDataTO);
    void automaticResizeArrays();
    void resizeArrays(ArraySizesForGpu const& sizeDelta = ArraySizesForGpu());
    void checkAndProcessSimulationParameterChanges();

    SimulationData getSimulationDataPtrCopy() const;

    GpuInfo _gpuInfo;
    cudaGraphicsResource* _cudaResource = nullptr;

    mutable std::mutex _mutexForSimulationParameters;
    std::optional<SimulationParameters> _newSimulationParameters;
    SimulationParametersUpdateConfig _simulationParametersUpdateConfig = SimulationParametersUpdateConfig::All;

    SettingsForSimulation _settings;
    SettingsForSimulation _settingsForPreview;

    mutable std::mutex _mutexForSimulationData;
    std::shared_ptr<SimulationData> _cudaSimulationData;    // std::shared_ptr to prevent include in header
    std::shared_ptr<SimulationData> _cudaPreviewData;
    std::shared_ptr<RenderingData> _cudaRenderingData;
    std::shared_ptr<SelectionResult> _cudaSelectionResult;
    CudaTOProvider _cudaTOProvider;
    TOProvider _collectionTOProvider;

    mutable std::mutex _mutexForStatistics;
    std::optional<std::chrono::steady_clock::time_point> _lastStatisticsUpdateTime;
    std::optional<StatisticsRawData> _statisticsData;
    StatisticsHistory _statisticsHistory;
    std::shared_ptr<SimulationStatistics> _cudaSimulationStatistics;
    std::shared_ptr<SimulationStatistics> _cudaPreviewStatistics;
    MaxAgeBalancer _maxAgeBalancer;

    SimulationKernelsService _simulationKernels;
    DataAccessKernelsService _dataAccessKernels;
    GarbageCollectorKernelsService _garbageCollectorKernels;
    RenderingKernelsService _renderingKernels;
    EditKernelsService _editKernels;
    StatisticsKernelsService _statisticsKernels;
    TestKernelsService _testKernels;
};
