#include "EngineWorker.h"

#include <chrono>

#include "EngineGpuKernels/AccessTOs.cuh"
#include "EngineInterface/ChangeDescriptions.h"
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
            ExceptionData const& exceptionData,
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
                checkForException(exceptionData);
            } else {
                std::cv_status status = conditionForAccess.wait_for(uniqueLock, std::chrono::milliseconds(5000));
                if (std::cv_status::timeout == status) {
                    checkForException(exceptionData);
                    throw std::runtime_error("GPU Timeout");
                }
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
        void checkForException(ExceptionData const& exceptionData)
        {
            std::unique_lock<std::mutex> uniqueLock(exceptionData.mutex);
            if (exceptionData.errorMessage) {
                throw std::runtime_error(*exceptionData.errorMessage);
            }
        }

        std::atomic<bool>& _accessFlag;
        std::condition_variable& _conditionForWorkerLoop;

        bool _isTimeout = false;
    };
}

void EngineWorker::initCuda()
{
    _CudaSimulation::initCuda();
}

void EngineWorker::newSimulation(uint64_t timestep, Settings const& settings, GpuSettings const& gpuSettings)
{
    _settings = settings;
    _gpuConstants = gpuSettings;
    _dataTOCache = boost::make_shared<_AccessDataTOCache>(gpuSettings);
    _cudaSimulation = boost::make_shared<_CudaSimulation>(timestep, settings, gpuSettings);
}

void EngineWorker::clear()
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);
    return _cudaSimulation->clear();
}

void EngineWorker::registerImageResource(GLuint image)
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);

    _cudaResource = _cudaSimulation->registerImageResource(image);
}

void EngineWorker::getVectorImage(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    CudaAccess access(
        _conditionForAccess,
        _conditionForWorkerLoop,
        _requireAccess,
        _isSimulationRunning,
        _exceptionData,
        FrameTimeout);

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
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);

    auto arraySizes = _cudaSimulation->getArraySizes();
    DataAccessTO dataTO =
        _dataTOCache->getDataTO({arraySizes.cellArraySize, arraySizes.particleArraySize, arraySizes.tokenArraySize});
    _cudaSimulation->getSimulationData(
        {rectUpperLeft.x, rectUpperLeft.y}, int2{rectLowerRight.x, rectLowerRight.y}, dataTO);

    DataConverter converter(dataTO, _settings.simulationParameters, _gpuConstants);
    return converter.getDataDescription();
}

OverallStatistics EngineWorker::getMonitorData() const
{
    OverallStatistics result;
    result.timeStep = _timeStep.load();
    result.numCells = _numCells.load();
    result.numParticles = _numParticles.load();
    result.numTokens = _numTokens.load();
    result.totalInternalEnergy = _totalInternalEnergy.load();
    result.numCreatedCells = _numCreatedCells.load();
    result.numSuccessfulAttacks = _numSuccessfulAttacks.load();
    result.numFailedAttacks = _numFailedAttacks.load();
    result.numMuscleActivities = _numMuscleActivities.load();
    return result;
}

void EngineWorker::updateData(DataChangeDescription const& dataToUpdate)
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);

    int numCells = 0;
    int numParticles = 0;
    int numTokens = 0;
    for (auto const& cell : dataToUpdate.cells) {
        if (cell.isAdded()) {
            ++numCells;
            if (cell->tokens.getOptionalValue()) {
                numTokens += toInt(cell->tokens.getValue().size());
            }
        }
    }
    for (auto const& particle : dataToUpdate.particles) {
        if (particle.isAdded()) {
            ++numParticles;
        }
    }
    _cudaSimulation->resizeArraysIfNecessary({numCells, numParticles, numTokens});

    auto arraySizes = _cudaSimulation->getArraySizes();
    DataAccessTO dataTO =
        _dataTOCache->getDataTO({arraySizes.cellArraySize, arraySizes.particleArraySize, arraySizes.tokenArraySize});
    int2 worldSize{_settings.generalSettings.worldSizeX, _settings.generalSettings.worldSizeY};
    _cudaSimulation->getSimulationData({0, 0}, worldSize, dataTO);
//    _cudaSimulation->getSimulationData({0, 0}, {0, 0}, dataTO);

    DataConverter converter(dataTO, _settings.simulationParameters, _gpuConstants);
    converter.updateData(dataToUpdate);

    _dataTOCache->releaseDataTO(dataTO);

    _cudaSimulation->setSimulationData({0, 0}, worldSize, dataTO);
    updateMonitorDataIntern();
}

void EngineWorker::calcSingleTimestep()
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);

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

float EngineWorker::getTps() const
{
    return _tps.load();
}

uint64_t EngineWorker::getCurrentTimestep() const
{
    return _cudaSimulation->getCurrentTimestep();
}

void EngineWorker::setCurrentTimestep(uint64_t value)
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);
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

void EngineWorker::setSimulationParametersSpots_async(SimulationParametersSpots const& spots)
{
    {
        std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
        _updateSimulationParametersSpotsJob = spots;
    }
    _conditionForWorkerLoop.notify_all();
}

void EngineWorker::setGpuSettings_async(GpuSettings const& gpuSettings)
{
    {
        std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
        _updateGpuSettingsJob = gpuSettings;
    }
    _conditionForWorkerLoop.notify_all();
}

void EngineWorker::setFlowFieldSettings_async(FlowFieldSettings const& flowFieldSettings)
{
    {
        std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
        _flowFieldSettings = flowFieldSettings;
    }
    _conditionForWorkerLoop.notify_all();
}

void EngineWorker::applyForce_async(
    RealVector2D const& start,
    RealVector2D const& end,
    RealVector2D const& force,
    float radius)
{
    {
        std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
        _applyForceJobs.emplace_back(ApplyForceJob{start, end, force, radius});
    }
    _conditionForWorkerLoop.notify_all();
}

void EngineWorker::switchSelection(RealVector2D const& pos, float radius)
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);
    _cudaSimulation->switchSelection(SwitchSelectionData{{pos.x, pos.y}, radius});
}

SelectionShallowData EngineWorker::getSelectionShallowData()
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);
    return _cudaSimulation->getSelectionShallowData();
}

void EngineWorker::setSelection(RealVector2D const& startPos, RealVector2D const& endPos)
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);
    _cudaSimulation->setSelection(SetSelectionData{{startPos.x, startPos.y}, {endPos.x, endPos.y}});
}

void EngineWorker::moveSelection(RealVector2D const& displacement)
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);
    _cudaSimulation->shallowUpdateSelection(ShallowUpdateSelectionData{{displacement.x, displacement.y}, {0, 0}});
}

void EngineWorker::accelerateSelection(RealVector2D const& velDelta)
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);
    _cudaSimulation->shallowUpdateSelection(ShallowUpdateSelectionData{{0, 0}, {velDelta.x, velDelta.y}});
}

void EngineWorker::removeSelection()
{
    CudaAccess access(
        _conditionForAccess, _conditionForWorkerLoop, _requireAccess, _isSimulationRunning, _exceptionData);
    _cudaSimulation->removeSelection();
}

void EngineWorker::runThreadLoop()
{
    try {
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
                            _tps.store(toFloat(_timestepsSinceTimepoint) * 5 * 200 / duration);
                        } else {
                            _tps.store(1000.0f / duration);
                        }
                        _timestepsSinceTimepoint = 0;
                    }
                }

                startTimestepTime = std::chrono::steady_clock::now();
                _cudaSimulation->calcCudaTimestep();
                updateMonitorDataIntern();
                ++_timestepsSinceTimepoint;
            }
            processJobs();
        }
    } catch (std::exception const& e) {
        std::unique_lock<std::mutex> uniqueLock(_exceptionData.mutex);
        _exceptionData.errorMessage = e.what();
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
        _numCreatedCells.store(data.numCreatedCells);
        _numSuccessfulAttacks.store(data.numSuccessfulAttacks);
        _numFailedAttacks.store(data.numFailedAttacks);
        _numMuscleActivities.store(data.numMuscleActivities);

        _lastMonitorUpdate = now;
    }
}

void EngineWorker::processJobs()
{
    std::unique_lock<std::mutex> asyncJobsLock(_mutexForAsyncJobs);
    if (_updateSimulationParametersJob) {
        _cudaSimulation->setSimulationParameters(*_updateSimulationParametersJob);
        _updateSimulationParametersJob = boost::none;
    }
    if (_updateSimulationParametersSpotsJob) {
        _cudaSimulation->setSimulationParametersSpots(*_updateSimulationParametersSpotsJob);
        _updateSimulationParametersSpotsJob = boost::none;
    }
    if (_updateGpuSettingsJob) {
        _cudaSimulation->setGpuConstants(*_updateGpuSettingsJob);
        _updateGpuSettingsJob = boost::none;
    }
    if (_flowFieldSettings) {
        _cudaSimulation->setFlowFieldSettings(*_flowFieldSettings);
        _flowFieldSettings = boost::none;
    }
    if (!_applyForceJobs.empty()) {
        for (auto const& applyForceJob : _applyForceJobs) {
            _cudaSimulation->applyForce(
                {{applyForceJob.start.x, applyForceJob.start.y},
                 {applyForceJob.end.x, applyForceJob.end.y},
                 {applyForceJob.force.x, applyForceJob.force.y},
                 applyForceJob.radius,
                 false});
        }
        _applyForceJobs.clear();
    }
}
