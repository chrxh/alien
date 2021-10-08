#pragma once

#include <mutex>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif
#include <GL/gl.h>

#include "Base/Definitions.h"

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuConstants.h"
#include "EngineInterface/OverallStatistics.h"
#include "EngineGpuKernels/Definitions.h"

#include "Definitions.h"
#include "DllExport.h"

class EngineWorker
{
public:
    ENGINEIMPL_EXPORT void initCuda();

    ENGINEIMPL_EXPORT void newSimulation(
        IntVector2D worldSize,
        int timestep,
        SimulationParameters const& parameters,
        GpuConstants const& gpuConstants);
    ENGINEIMPL_EXPORT void clear();

    ENGINEIMPL_EXPORT void registerImageResource(GLuint image);

    ENGINEIMPL_EXPORT void getVectorImage(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom);
    ENGINEIMPL_EXPORT DataDescription
    getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight);
    ENGINEIMPL_EXPORT OverallStatistics getMonitorData() const;

    ENGINEIMPL_EXPORT void updateData(DataChangeDescription const& dataToUpdate);

    ENGINEIMPL_EXPORT void calcSingleTimestep();

    ENGINEIMPL_EXPORT void beginShutdown(); //caller should wait for termination of thread
    ENGINEIMPL_EXPORT void endShutdown();

    ENGINEIMPL_EXPORT int getTpsRestriction() const;
    ENGINEIMPL_EXPORT void setTpsRestriction(int value);

    ENGINEIMPL_EXPORT float getTps() const;
    ENGINEIMPL_EXPORT uint64_t getCurrentTimestep() const;
    ENGINEIMPL_EXPORT void setCurrentTimestep(uint64_t value);

    ENGINEIMPL_EXPORT void setSimulationParameters_async(SimulationParameters const& parameters);
    ENGINEIMPL_EXPORT void setGpuSettings_async(GpuConstants const& gpuSettings);

    ENGINEIMPL_EXPORT void
    applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius);

    void runThreadLoop();
    void runSimulation();
    void pauseSimulation();
    bool isSimulationRunning() const;

private:
    void updateMonitorDataIntern();
    void processJobs();

    CudaSimulation _cudaSimulation;

    //sync
    mutable std::mutex _mutexForLoop;
    std::condition_variable _conditionForWorkerLoop;
    std::condition_variable _conditionForAccess;

    std::atomic<bool> _isSimulationRunning = false;
    std::atomic<bool> _isShutdown = false;
    std::atomic<bool> _requireAccess = false;

    //async jobs
    mutable std::mutex _mutexForAsyncJobs;
    boost::optional<SimulationParameters> _updateSimulationParametersJob;
    boost::optional<GpuConstants> _updateGpuSettingsJob;

    struct ApplyForceJob
    {
        RealVector2D start;
        RealVector2D end;
        RealVector2D force;
        float radius;
    };
    std::vector<ApplyForceJob> _applyForceJobs;

    //time step measurements
    std::atomic<int> _tpsRestriction = 0;   //0 = no restriction
    std::atomic<float> _tps;
    boost::optional<std::chrono::steady_clock::time_point> _timepoint;
    int _timestepsSinceTimepoint = 0;
  
    //settings
    IntVector2D _worldSize;
    SimulationParameters _parameters;
    GpuConstants _gpuConstants;

    //monitor data
    boost::optional<std::chrono::steady_clock::time_point> _lastMonitorUpdate;
    std::atomic<uint64_t> _timeStep = 0;
    std::atomic<int> _numCells = 0;
    std::atomic<int> _numParticles = 0;
    std::atomic<int> _numTokens = 0;
    std::atomic<double> _totalInternalEnergy = 0.0;
    std::atomic<int> _numCreatedCells = 0;
    std::atomic<int> _numSuccessfulAttacks = 0;
    std::atomic<int> _numFailedAttacks = 0;
    std::atomic<int> _numMuscleActivities = 0;

    //internals
    void* _cudaResource;
    AccessDataTOCache _dataTOCache;
};