#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>

#if defined(_WIN32)
#include <windows.h>
#endif
#include <GL/gl.h>

#include "Base/Definitions.h"

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/StatisticsData.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "EngineInterface/Settings.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/MutationType.h"
#include "EngineGpuKernels/Definitions.h"

#include "Definitions.h"

struct ExceptionData
{
    mutable std::mutex mutex;
    std::optional<std::string> errorMessage;
};

struct DataTO;

class EngineWorker
{
    friend class EngineWorkerGuard;
public:
    void newSimulation(uint64_t timestep, GeneralSettings const& generalSettings, SimulationParameters const& parameters);
    void clear();

    void setImageResource(void* image);
    std::string getGpuName() const;

    void tryDrawVectorGraphics(RealVector2D const& rectUpperLeft, RealVector2D const& rectLowerRight, IntVector2D const& imageSize, double zoom);
    std::optional<OverlayDescription>
    tryDrawVectorGraphicsAndReturnOverlay(RealVector2D const& rectUpperLeft, RealVector2D const& rectLowerRight, IntVector2D const& imageSize, double zoom);

    bool isSyncSimulationWithRendering() const;
    void setSyncSimulationWithRendering(bool value);
    int getSyncSimulationWithRenderingRatio() const;
    void setSyncSimulationWithRenderingRatio(int value);

    ClusteredDataDescription getClusteredSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight);
    DataDescription getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight);
    ClusteredDataDescription getSelectedClusteredSimulationData(bool includeClusters);
    DataDescription getSelectedSimulationData(bool includeClusters);
    DataDescription getInspectedSimulationData(std::vector<uint64_t> objectsIds);
    StatisticsData getStatistics() const;

    void addAndSelectSimulationData(DataDescription const& dataToUpdate);
    void setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate);
    void setSimulationData(DataDescription const& dataToUpdate);
    void removeSelectedObjects(bool includeClusters);
    void relaxSelectedObjects(bool includeClusters);
    void uniformVelocitiesForSelectedObjects(bool includeClusters);
    void makeSticky(bool includeClusters);
    void removeStickiness(bool includeClusters);
    void setBarrier(bool value, bool includeClusters);
    void changeCell(CellDescription const& changedCell);
    void changeParticle(ParticleDescription const& changedParticle);

    void calcTimesteps(uint64_t timesteps);
    void applyCataclysm(int power);

    void beginShutdown(); //caller should wait for termination of thread
    void endShutdown();

    int getTpsRestriction() const;
    void setTpsRestriction(int value);

    float getTps() const;
    uint64_t getCurrentTimestep() const;
    void setCurrentTimestep(uint64_t value);

    SimulationParameters getSimulationParameters() const;
    void setSimulationParameters(SimulationParameters const& parameters);
    void setGpuSettings_async(GpuSettings const& gpuSettings);

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
    void runSimulation();
    void pauseSimulation();
    bool isSimulationRunning() const;

    //for tests
    void testOnly_mutate(uint64_t cellId, MutationType mutationType);

private:
    DataTO provideTO(); 
    void resetTimeIntervalStatistics();
    void updateStatistics(bool afterMinDuration = false);
    void processJobs();

    void syncSimulationWithRenderingIfDesired();
    void waitAndAllowAccess(std::chrono::microseconds const& duration);
    void measureTPS();
    void slowdownTPS();

    CudaSimulationFacade _cudaSimulation;

    //settings
    Settings _settings;

    //sync
    std::atomic<bool> _syncSimulationWithRendering{false};
    std::atomic<int> _syncSimulationWithRenderingRatio{2};
    std::atomic<int> _accessState{0};  //0 = worker thread has access, 1 = require access from other thread, 2 = access granted to other thread
    std::atomic<bool> _isSimulationRunning{false};
    std::atomic<bool> _isShutdown{false};
    ExceptionData _exceptionData;

    //async jobs
    mutable std::mutex _mutexForAsyncJobs;
    std::optional<GpuSettings> _updateGpuSettingsJob;
    std::optional<GLuint> _imageResource;

    struct ApplyForceJob
    {
        RealVector2D start;
        RealVector2D end;
        RealVector2D force;
        float radius;
    };
    std::vector<ApplyForceJob> _applyForceJobs;

    //time step measurements
    std::atomic<int> _tpsRestriction{0};  //0 = no restriction
    std::atomic<float> _tps;
    int _timestepsSinceMeasurement = 0;
    std::optional<std::chrono::steady_clock::time_point> _measureTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _slowDownTimepoint;
    std::optional<std::chrono::microseconds> _slowDownOvershot;
  
    //internals
    void* _cudaResource;
    AccessDataTOCache _dataTOCache;
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
