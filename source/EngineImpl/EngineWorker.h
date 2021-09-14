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

    ENGINEIMPL_EXPORT void registerImageResource(GLuint image);

    ENGINEIMPL_EXPORT void getVectorImage(
        RealVector2D const& rectUpperLeft,
        RealVector2D const& rectLowerRight,
        IntVector2D const& imageSize,
        double zoom);

    ENGINEIMPL_EXPORT void updateData(DataChangeDescription const& dataToUpdate);

    ENGINEIMPL_EXPORT void calcNextTimestep();

    ENGINEIMPL_EXPORT void beginShutdown(); //caller should wait for termination of thread
    ENGINEIMPL_EXPORT void endShutdown();

    ENGINEIMPL_EXPORT int getTpsRestriction() const;
    ENGINEIMPL_EXPORT void setTpsRestriction(int value);

    ENGINEIMPL_EXPORT int getTps() const;

    void runThreadLoop();
    void runSimulation();
    void pauseSimulation();
    bool isSimulationRunning() const;

private:
    mutable std::mutex _mutexForLoop;
    std::condition_variable _conditionForWorkerLoop;
    std::condition_variable _conditionForAccess;

    std::atomic<bool> _isSimulationRunning = false;
    std::atomic<bool> _isShutdown = false;
    std::atomic<bool> _requireAccess = false;
    std::atomic<int> _tpsRestriction = 0;   //0 = no restriction
    std::atomic<int> _tps;
    boost::optional<std::chrono::steady_clock::time_point> _timepoint;
    int _timestepsSinceTimepoint = 0;
  
    CudaSimulation _cudaSimulation;
    void* _cudaResource;     //for rendering

    IntVector2D _worldSize;
    SimulationParameters _parameters;
    GpuConstants _gpuConstants;

    AccessDataTOCache _dataTOCache;
};