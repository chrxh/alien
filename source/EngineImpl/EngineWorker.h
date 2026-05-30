#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>

#if defined(_WIN32)
#include <windows.h>
#endif

#include <Base/Definitions.h>

#include <EngineInterface/ArraySizesForGpuEntities.h>
#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/Definitions.h>
#include <EngineInterface/GeometryBuffers.h>
#include <EngineInterface/PreviewDesc.h>
#include <EngineInterface/SelectionShallowData.h>
#include <EngineInterface/SettingsForSimulation.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>
#include <EngineInterface/SimulationParameters.h>
#include <EngineInterface/SimulationParametersUpdateConfig.h>
#include <EngineInterface/StatisticsHistory.h>
#include <EngineInterface/StatisticsRawData.h>

#include <EngineKernels/Definitions.h>

#include "Definitions.h"

struct ExceptionData
{
    mutable std::mutex mutex;
    std::optional<std::string> errorMessage;
};

struct TOs;

class EngineWorker
{
    friend class EngineWorkerGuard;

public:
    void newSimulation(uint64_t timestep, SettingsForSimulation const& _settings);
    void clear();

    std::string getGpuName() const;

    void tryCopyBuffersFromCudaToOpenGL(GeometryBuffers const& geometryBuffers, RealRect const& visibleWorldRect);

    bool isSyncSimulationWithRendering() const;
    void setSyncSimulationWithRendering(bool value);
    int getSyncSimulationWithRenderingRatio() const;
    void setSyncSimulationWithRenderingRatio(int value);

    Desc getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight);
    Desc getSelectedSimulationData(bool includeClusters);
    Desc getInspectedSimulationData(std::vector<uint64_t> objectsIds);
    StatisticsRawData getStatisticsRawData() const;
    StatisticsHistory const& getStatisticsHistory() const;
    void setStatisticsHistory(StatisticsHistoryData const& data);

    void addAndSelectSimulationData(Desc&& dataToUpdate);
    void setSimulationData(Desc const& dataToUpdate);
    void removeSelectedObjects(bool includeClusters);
    void relaxSelectedObjects(bool includeClusters);
    void uniformVelocitiesForSelectedObjects(bool includeClusters);
    void makeSticky(bool includeClusters);
    void removeStickiness(bool includeClusters);
    void setBarrier(bool value, bool includeClusters);
    void changeCell(ExtendedObjectDesc const& changedCell);
    void changeParticle(EnergyDesc const& changedParticle);
    int injectGenomeToSelectedCreatures(GenomeDesc const& genome);

    void calcTimesteps(uint64_t timesteps);
    void applyCataclysm(int power);

    void beginShutdown();  //caller should wait for termination of thread
    void endShutdown();

    int getTpsRestriction() const;
    void setTpsRestriction(int value);

    float getTps() const;
    uint64_t getCurrentTimestep() const;
    void setCurrentTimestep(uint64_t value);

    SimulationParameters getSimulationParameters() const;
    void setSimulationParameters(
        SimulationParameters const& parameters,
        SimulationParametersUpdateConfig const& updateConfig = SimulationParametersUpdateConfig::All);
    void setGpuSettings_async(CudaSettings const& gpuSettings);

    void applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius);

    void switchSelection(RealVector2D const& pos, float radius);
    void swapSelection(RealVector2D const& pos, float radius);
    SelectionShallowData getSelectionShallowData();
    void setSelection(RealVector2D const& startPos, RealVector2D const& endPos);
    void removeSelection();
    void updateSelection();
    void shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& updateData);
    void colorSelectedObjects(unsigned char color, bool includeClusters);
    void reconnectSelectedObjects();
    void setDetached(bool value);

    void runThreadLoop();
    void checkAndThrowException() const;
    void runSimulation();
    void pauseSimulation();
    bool isSimulationRunning() const;

    // Simulated preview
    Desc getPreviewData();
    void setPreviewData(Desc const& description);
    void calcTimestepsForPreview(std::chrono::milliseconds const& duration, bool detailSimulation);
    void calcTimestepsForPreview(int numSteps, bool detailSimulation);
    uint64_t getCurrentTimestepForPreview();
    void setCurrentTimestepForPreview(uint64_t timestep);

    // Only for tests
    void testOnly_mutate(uint64_t objectId);
    void testOnly_createConnection(uint64_t objectId1, uint64_t objectId2);
    void testOnly_createConnectionWithAbsAngle(uint64_t objectId1, uint64_t objectId2, float desiredDistance, float desiredAbsAngle1, float desiredAbsAngle2);
    void testOnly_cleanupAfterTimestep();
    void testOnly_cleanupAfterDataManipulation();
    void testOnly_resizeArrays(ArraySizesForGpuEntities const& sizeDelta);
    bool testOnly_isDataValid();
    void testOnly_calcTimestepWithCellFunctions();
    void testOnly_calcTimestepWithCellFunctionsForPreview(bool detailSimulation);
    void testOnly_zeroTransferData();

private:
    void resetTimeIntervalStatistics();
    void processJobs();

    void syncSimulationWithRenderingIfDesired();
    void waitAndAllowAccess(std::chrono::microseconds const& duration);
    void measureTPS();
    void slowdownTPS();

    CudaSimulationFacade _simulationCudaFacade;

    // Settings
    SettingsForSimulation _settings;

    // Sync
    std::atomic<bool> _syncSimulationWithRendering{false};
    std::atomic<int> _syncSimulationWithRenderingRatio{2};
    std::atomic<int> _accessState{0};  //0 = worker thread has access, 1 = require access from other thread, 2 = access granted to other thread
    std::atomic<bool> _isSimulationRunning{false};
    std::atomic<bool> _isPreviewRunning{false};
    std::atomic<bool> _isShutdown{false};
    ExceptionData _exceptionData;

    // Async jobs
    std::recursive_mutex _mutexForEngineWorkerGuard;
    mutable std::mutex _mutexForAsyncJobs;
    std::optional<CudaSettings> _updateGpuSettingsJob;

    struct ApplyForceJob
    {
        RealVector2D start;
        RealVector2D end;
        RealVector2D force;
        float radius;
    };
    std::vector<ApplyForceJob> _applyForceJobs;

    // Time step measurements
    std::atomic<int> _tpsRestriction{0};  //0 = no restriction
    std::atomic<float> _tps;
    int _timestepsSinceMeasurement = 0;
    std::optional<std::chrono::steady_clock::time_point> _measureTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _slowDownTimepoint;
    std::optional<std::chrono::microseconds> _slowDownOvershot;

    // Internals
    TOProvider _collectionTOProvider;
};

class EngineWorkerGuard
{
public:
    EngineWorkerGuard(EngineWorker* worker, std::optional<std::chrono::milliseconds> const& maxDuration = std::nullopt);
    ~EngineWorkerGuard();

    bool isTimeout() const;

private:
    void checkForException(ExceptionData const& exceptionData);

    EngineWorker* _worker;

    bool _isTimeout = false;
};
