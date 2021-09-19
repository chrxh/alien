#include "EngineWorker.h"

#include <chrono>

#include "EngineInterface/ChangeDescriptions.h"
#include "EngineGpuKernels/AccessTOs.cuh"

#include "AccessDataTOCache.h"
#include "DataConverter.h"

namespace
{
    std::chrono::milliseconds const FrameTimeout(30);
    std::chrono::milliseconds const MonitorUpdate(30);

    class CudaAccess
    {
    public:
        CudaAccess(
            std::condition_variable& conditionForAccess,
            std::condition_variable& conditionForWorkerLoop,
            std::atomic<bool>& accessFlag,
            std::atomic<bool> const& isSimulationRunning,
            boost::optional<std::chrono::milliseconds> const& maxDuration = boost::none)
            : _accessFlag(accessFlag)
            , _conditionForWorkerLoop(conditionForWorkerLoop)
        {
            if (!isSimulationRunning.load()) {
                return;
            }
            std::mutex mutex;
            accessFlag.store(true);
            std::unique_lock<std::mutex> uniqueLock(mutex);
            if (maxDuration) {
                std::cv_status status = conditionForAccess.wait_for(uniqueLock, *maxDuration);
                _isTimeout = std::cv_status::timeout == status;
            } else {
                conditionForAccess.wait(uniqueLock);
            }
            conditionForWorkerLoop.notify_all();
        }

        ~CudaAccess()
        {
            _accessFlag.store(false);
            _conditionForWorkerLoop.notify_all();
        }

        bool isTimeout() const { return _isTimeout; }

    private:
        std::atomic<bool>& _accessFlag;
        std::condition_variable& _conditionForWorkerLoop;

        bool _isTimeout = false;
    };
}

void EngineWorker::initCuda()
{
    _CudaSimulation::initCuda();
}

void EngineWorker::newSimulation(
    IntVector2D size,
    int timestep,
    SimulationParameters const& parameters,
    GpuConstants const& gpuConstants)
{
    _worldSize = size;
    _parameters = parameters;
    _gpuConstants = gpuConstants;
    _dataTOCache = boost::make_shared<_AccessDataTOCache>(gpuConstants);
    _cudaSimulation = boost::make_shared<_CudaSimulation>(int2{size.x, size.y}, timestep, parameters, gpuConstants);
}

void EngineWorker::clear()
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);
    return _cudaSimulation->clear();
}

void EngineWorker::registerImageResource(GLuint image)
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);

    _cudaResource = _cudaSimulation->registerImageResource(image);
}

void EngineWorker::getVectorImage(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, FrameTimeout);

    if (!access.isTimeout()) {
        _cudaSimulation->getVectorImage(
            {rectUpperLeft.x, rectUpperLeft.y},
            {rectLowerRight.x, rectLowerRight.y},
            _cudaResource,
            {imageSize.x, imageSize.y},
            zoom);
    }
}

DataDescription EngineWorker::getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);

    DataAccessTO dataTO = _dataTOCache->getDataTO();
    _cudaSimulation->getSimulationData(
        {rectUpperLeft.x, rectUpperLeft.y}, int2{rectLowerRight.x, rectLowerRight.y}, dataTO);

    DataConverter converter(dataTO, _parameters, _gpuConstants);
    return converter.getDataDescription();
}

MonitorData EngineWorker::getMonitorData() const
{
    MonitorData result;
    result.timeStep = _timeStep.load();
    result.numCells = _numCells.load();
    result.numParticles = _numParticles.load();
    result.numTokens = _numTokens.load();
    result.totalInternalEnergy = _totalInternalEnergy.load();
    return result;
}

void EngineWorker::updateData(DataChangeDescription const& dataToUpdate)
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);

    DataAccessTO dataTO = _dataTOCache->getDataTO();
    _cudaSimulation->getSimulationData({0, 0}, int2{_worldSize.x, _worldSize.y}, dataTO);

    DataConverter converter(dataTO, _parameters, _gpuConstants);
    converter.updateData(dataToUpdate);

    _dataTOCache->releaseDataTO(dataTO);

    _cudaSimulation->setSimulationData({0, 0}, int2{_worldSize.x, _worldSize.y}, dataTO);
    updateMonitorDataIntern();
}

void EngineWorker::calcSingleTimestep()
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);

    _cudaSimulation->calcCudaTimestep();
    updateMonitorDataIntern();
}

void EngineWorker::beginShutdown()
{
    _isShutdown.store(true);
    _conditionForWorkerLoop.notify_all();
}

void EngineWorker::endShutdown()
{
    _isSimulationRunning = false;
    _isShutdown = false;
    _requireAccess = false;

    _cudaSimulation.reset();
}

int EngineWorker::getTpsRestriction() const
{
    auto result = _tpsRestriction.load();
    return result;
}

void EngineWorker::setTpsRestriction(int value)
{
    _tpsRestriction.store(value);
}

int EngineWorker::getTps() const
{
    return _tps.load();
}

uint64_t EngineWorker::getCurrentTimestep() const
{
    return _cudaSimulation->getCurrentTimestep();
}

void EngineWorker::setCurrentTimestep(uint64_t value)
{
    CudaAccess access(_conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning);
    _cudaSimulation->setCurrentTimestep(value);
}

void EngineWorker::setSimulationParameters_async(SimulationParameters const& parameters)
{
    {
        std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
        _updateSimulationParametersJob = parameters;
    }
    _conditionForWorkerLoop.notify_all();
}

void EngineWorker::runThreadLoop()
{
    std::unique_lock<std::mutex> uniqueLock(_mutexForLoop);
    boost::optional<std::chrono::steady_clock::time_point> startTimestepTime;
    while (true) {
        if (!_isSimulationRunning.load()) {

            //sleep...
            _tps.store(0);
            _conditionForWorkerLoop.wait(uniqueLock);
        }
        if (_isShutdown.load()) {
            return;
        }
        while (_requireAccess.load()) {
            _conditionForAccess.notify_all();
        }

        if (_isSimulationRunning.load()) {
            if (startTimestepTime && _tpsRestriction.load() > 0) {
                long long int actualDuration, desiredDuration;
                do {
                    auto tpsRestriction = _tpsRestriction.load();
                    desiredDuration = (0 != tpsRestriction) ? 1000000 / tpsRestriction : 0;
                    actualDuration = std::chrono::duration_cast<std::chrono::microseconds>(
                                         std::chrono::steady_clock::now() - *startTimestepTime)
                                         .count();

                    _conditionForAccess.notify_all();
                    //                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                } while (actualDuration < desiredDuration || _requireAccess.load());
            }

            auto timepoint = std::chrono::steady_clock::now();
            if (!_timepoint) {
                _timepoint = timepoint;
            } else {
                int duration = static_cast<int>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_timepoint).count());
                if (duration > 199) {
                    _timepoint = timepoint;
                    if (duration < 350) {
                        _tps.store(_timestepsSinceTimepoint * 5 * 200 / duration);
                    } else {
                        _tps.store(1000 / duration);
                    }
                    _timestepsSinceTimepoint = 0;
                }
            }

            startTimestepTime = std::chrono::steady_clock::now();
            _cudaSimulation->calcCudaTimestep();
            updateMonitorDataIntern();
            ++_timestepsSinceTimepoint;
        }

        std::unique_lock<std::mutex> asyncJobsLock(_mutexForAsyncJobs);
        if (_updateSimulationParametersJob) {
            _cudaSimulation->setSimulationParameters(*_updateSimulationParametersJob);
            _updateSimulationParametersJob = boost::none;
        }
    }
}

void EngineWorker::runSimulation()
{
    _isSimulationRunning.store(true);
    _conditionForWorkerLoop.notify_all();
}

void EngineWorker::pauseSimulation()
{
    _isSimulationRunning.store(false);
    _conditionForWorkerLoop.notify_all();
}

bool EngineWorker::isSimulationRunning() const
{
    return _isSimulationRunning.load();
}

void EngineWorker::updateMonitorDataIntern()
{
    auto now = std::chrono::steady_clock::now();
    if (!_lastMonitorUpdate || now - *_lastMonitorUpdate > MonitorUpdate) {

        auto data = _cudaSimulation->getMonitorData();
        _timeStep.store(data.timeStep);
        _numCells.store(data.numCells);
        _numParticles.store(data.numParticles);
        _numTokens.store(data.numTokens);
        _totalInternalEnergy.store(data.totalInternalEnergy);

        _lastMonitorUpdate = now;
    }
}
