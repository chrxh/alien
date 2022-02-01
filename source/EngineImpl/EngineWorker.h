#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/gl.h>

#include "Base/Definitions.h"

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/OverallStatistics.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "EngineInterface/FlowFieldSettings.h"
#include "EngineInterface/Settings.h"
#include "EngineInterface/SelectionShallowData.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineGpuKernels/Definitions.h"

#include "Definitions.h"

struct ExceptionData
{
    mutable std::mutex mutex;
    std::optional<std::string> errorMessage;
};

struct DataAccessTO;

class EngineWorker
{
    friend class EngineWorkerGuard;
public:
    void initCuda();

    void newSimulation(uint64_t timestep, Settings const& settings);
    void clear();

    void registerImageResource(void* image);

    void tryDrawVectorGraphics(RealVector2D const& rectUpperLeft, RealVector2D const& rectLowerRight, IntVector2D const& imageSize, double zoom);
    std::optional<OverlayDescription>
    tryDrawVectorGraphicsAndReturnOverlay(RealVector2D const& rectUpperLeft, RealVector2D const& rectLowerRight, IntVector2D const& imageSize, double zoom);

    ClusteredDataDescription getClusteredSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight);
    DataDescription getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight);
    ClusteredDataDescription getSelectedClusteredSimulationData(bool includeClusters);
    DataDescription getSelectedSimulationData(bool includeClusters);
    DataDescription getInspectedSimulationData(std::vector<uint64_t> entityIds);
    OverallStatistics getMonitorData() const;

    void addAndSelectSimulationData(DataDescription const& dataToUpdate);
    void setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate);
    void setSimulationData(DataDescription const& dataToUpdate);
    void removeSelectedEntities(bool includeClusters);
    void relaxSelectedEntities(bool includeClusters);
    void changeCell(CellDescription const& changedCell);
    void changeParticle(ParticleDescription const& changedParticle);

    void calcSingleTimestep();

    void beginShutdown(); //caller should wait for termination of thread
    void endShutdown();

    int getTpsRestriction() const;
    void setTpsRestriction(int value);

    float getTps() const;
    uint64_t getCurrentTimestep() const;
    void setCurrentTimestep(uint64_t value);

    void setSimulationParameters_async(SimulationParameters const& parameters);
    void setSimulationParametersSpots_async(SimulationParametersSpots const& spots);
    void setGpuSettings_async(GpuSettings const& gpuSettings);
    void setFlowFieldSettings_async(FlowFieldSettings const& flowFieldSettings);

    void applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius);

    void switchSelection(RealVector2D const& pos, float radius);
    void swapSelection(RealVector2D const& pos, float radius);
    SelectionShallowData getSelectionShallowData();
    void setSelection(RealVector2D const& startPos, RealVector2D const& endPos);
    void removeSelection();
    void updateSelection();
    void shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& updateData);
    void colorSelectedEntities(unsigned char color, bool includeClusters);
    void reconnectSelectedEntities();

    void runThreadLoop();
    void runSimulation();
    void pauseSimulation();
    bool isSimulationRunning() const;

private:
    DataAccessTO provideTO(); 
    void updateMonitorDataIntern();
    void processJobs();

    CudaSimulationAdapter _cudaSimulation;

    //sync
    mutable std::mutex _mutexForLoop;
    std::condition_variable _conditionForWorkerLoop;
    std::condition_variable _conditionForAccess;

    std::atomic<bool> _isSimulationRunning{false};
    std::atomic<bool> _isShutdown{false};
    std::atomic<int> _requireAccess{0}; // 1=require access, 2=access granted
    ExceptionData _exceptionData;

    //async jobs
    mutable std::mutex _mutexForAsyncJobs;
    std::optional<SimulationParameters> _updateSimulationParametersJob;
    std::optional<SimulationParametersSpots> _updateSimulationParametersSpotsJob;
    std::optional<GpuSettings> _updateGpuSettingsJob;
    std::optional<FlowFieldSettings> _flowFieldSettings;
    std::optional<GLuint> _imageResourceToRegister;

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
    std::optional<std::chrono::steady_clock::time_point> _timepoint;
    int _timestepsSinceTimepoint = 0;
  
    //settings
    Settings _settings;

    //monitor data
    std::optional<std::chrono::steady_clock::time_point> _lastMonitorUpdate;
    std::atomic<uint64_t> _timeStep{0};
    std::atomic<int> _numCells{0};
    std::atomic<int> _numParticles{0};
    std::atomic<int> _numTokens{0};
    std::atomic<double> _totalInternalEnergy{0.0};
    std::atomic<int> _numCreatedCells{0};
    std::atomic<int> _numSuccessfulAttacks{0};
    std::atomic<int> _numFailedAttacks{0};
    std::atomic<int> _numMuscleActivities{0};

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

    std::atomic<int>& _accessFlag;
    std::condition_variable& _conditionForWorkerLoop;

    bool _isTimeout = false;
};
