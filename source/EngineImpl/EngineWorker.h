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
#include "DllExport.h"

struct ExceptionData
{
    mutable std::mutex mutex;
    boost::optional<std::string> errorMessage;
};

struct DataAccessTO;

class EngineWorker
{
public:
    void initCuda();

    void newSimulation(uint64_t timestep, Settings const& settings, GpuSettings const& gpuSettings);
    void clear();

    void registerImageResource(GLuint image);

    void tryDrawVectorGraphics(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom);
    boost::optional<OverlayDescription> tryDrawVectorGraphicsAndReturnOverlay(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom);

    DataDescription getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight);
    DataDescription getSelectedSimulationData(bool includeClusters);
    DataDescription getInspectedSimulationData(std::vector<uint64_t> entityIds);
    OverallStatistics getMonitorData() const;

    void addAndSelectSimulationData(DataDescription const& dataToUpdate);
    void setSimulationData(DataDescription const& dataToUpdate);
    void removeSelectedEntities(bool includeClusters);
    void changeCell(CellDescription const& changedCell);

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

    void runThreadLoop();
    void runSimulation();
    void pauseSimulation();
    bool isSimulationRunning() const;

private:
    DataAccessTO provideTO(); 
    void updateMonitorDataIntern();
    void processJobs();

    CudaSimulation _cudaSimulation;

    //sync
    mutable std::mutex _mutexForLoop;
    std::condition_variable _conditionForWorkerLoop;
    std::condition_variable _conditionForAccess;

    std::atomic<bool> _isSimulationRunning{false};
    std::atomic<bool> _isShutdown{false};
    std::atomic<bool> _requireAccess{false};
    ExceptionData _exceptionData;

    //async jobs
    mutable std::mutex _mutexForAsyncJobs;
    boost::optional<SimulationParameters> _updateSimulationParametersJob;
    boost::optional<SimulationParametersSpots> _updateSimulationParametersSpotsJob;
    boost::optional<GpuSettings> _updateGpuSettingsJob;
    boost::optional<FlowFieldSettings> _flowFieldSettings;
    boost::optional<GLuint> _imageResourceToRegister;

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
    boost::optional<std::chrono::steady_clock::time_point> _timepoint;
    int _timestepsSinceTimepoint = 0;
  
    //settings
    Settings _settings;
    GpuSettings _gpuConstants;

    //monitor data
    boost::optional<std::chrono::steady_clock::time_point> _lastMonitorUpdate;
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