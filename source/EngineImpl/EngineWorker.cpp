#include "EngineWorker.h"

#include <chrono>
#include <thread>

#include "EngineGpuKernels/AccessTOs.cuh"
#include "EngineGpuKernels/CudaSimulationFacade.cuh"
#include "AccessDataTOCache.h"
#include "DataConverter.h"

namespace
{
    std::chrono::milliseconds const FrameTimeout(500);
    std::chrono::milliseconds const MonitorUpdate(30);
}

void EngineWorker::initCuda()
{
    _CudaSimulationFacade::initCuda();
}

void EngineWorker::newSimulation(uint64_t timestep, Settings const& settings)
{
    _accessState = 0;
    _settings = settings;
    _dataTOCache = std::make_shared<_AccessDataTOCache>(settings.gpuSettings);
    _cudaSimulation = std::make_shared<_CudaSimulationFacade>(timestep, settings);

    if (_imageResourceToRegister) {
        _cudaResource = _cudaSimulation->registerImageResource(*_imageResourceToRegister);
        _imageResourceToRegister = std::nullopt;
    }
    updateMonitorDataIntern();
}

void EngineWorker::clear()
{
    EngineWorkerGuard access(this);
    return _cudaSimulation->clear();
}

void EngineWorker::registerImageResource(void* image)
{
    GLuint imageId = reinterpret_cast<uintptr_t>(image);
    if (!_cudaSimulation) {

        //cuda is not initialized yet => register image resource later
        _imageResourceToRegister = imageId;
    } else {

        EngineWorkerGuard access(this);
        _cudaResource = _cudaSimulation->registerImageResource(imageId);
    }
}

void EngineWorker::tryDrawVectorGraphics(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    EngineWorkerGuard access(this, FrameTimeout);


    if (!access.isTimeout()) {
        _cudaSimulation->drawVectorGraphics(
            {rectUpperLeft.x, rectUpperLeft.y},
            {rectLowerRight.x, rectLowerRight.y},
            _cudaResource,
            {imageSize.x, imageSize.y},
            zoom);
    }
}

std::optional<OverlayDescription> EngineWorker::tryDrawVectorGraphicsAndReturnOverlay(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    EngineWorkerGuard access(this, FrameTimeout);

    if (!access.isTimeout()) {
        _cudaSimulation->drawVectorGraphics(
            {rectUpperLeft.x, rectUpperLeft.y},
            {rectLowerRight.x, rectLowerRight.y},
            _cudaResource,
            {imageSize.x, imageSize.y},
            zoom);

        auto arraySizes = _cudaSimulation->getArraySizes();
        DataAccessTO dataTO = _dataTOCache->getDataTO({arraySizes.cellArraySize, arraySizes.particleArraySize});

        _cudaSimulation->getOverlayData(
            {toInt(rectUpperLeft.x), toInt(rectUpperLeft.y)},
            int2{toInt(rectLowerRight.x), toInt(rectLowerRight.y)},
            dataTO);

        DataConverter converter(_settings.simulationParameters);
        auto result = converter.convertAccessTOtoOverlayDescription(dataTO);
        _dataTOCache->releaseDataTO(dataTO);

        return result;
    }
    return std::nullopt;
}

ClusteredDataDescription EngineWorker::getClusteredSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    EngineWorkerGuard access(this);

    auto arraySizes = _cudaSimulation->getArraySizes();
    DataAccessTO dataTO =
        _dataTOCache->getDataTO({arraySizes.cellArraySize, arraySizes.particleArraySize});
    _cudaSimulation->getSimulationData(
        {rectUpperLeft.x, rectUpperLeft.y}, int2{rectLowerRight.x, rectLowerRight.y}, dataTO);

    DataConverter converter(_settings.simulationParameters);

    auto result = converter.convertAccessTOtoClusteredDataDescription(dataTO);
    _dataTOCache->releaseDataTO(dataTO);

    return result;
}

DataDescription EngineWorker::getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    EngineWorkerGuard access(this);

    auto arraySizes = _cudaSimulation->getArraySizes();
    DataAccessTO dataTO = _dataTOCache->getDataTO({arraySizes.cellArraySize, arraySizes.particleArraySize});
    _cudaSimulation->getSimulationData({rectUpperLeft.x, rectUpperLeft.y}, int2{rectLowerRight.x, rectLowerRight.y}, dataTO);

    DataConverter converter(_settings.simulationParameters);

    auto result = converter.convertAccessTOtoDataDescription(dataTO);
    _dataTOCache->releaseDataTO(dataTO);

    return result;
}

ClusteredDataDescription EngineWorker::getSelectedClusteredSimulationData(bool includeClusters)
{
    EngineWorkerGuard access(this);

    auto arraySizes = _cudaSimulation->getArraySizes();
    DataAccessTO dataTO = _dataTOCache->getDataTO({arraySizes.cellArraySize, arraySizes.particleArraySize});
    _cudaSimulation->getSelectedSimulationData(includeClusters, dataTO);

    DataConverter converter(_settings.simulationParameters);

    auto result = converter.convertAccessTOtoClusteredDataDescription(dataTO);
    _dataTOCache->releaseDataTO(dataTO);

    return result;
}

DataDescription EngineWorker::getSelectedSimulationData(bool includeClusters)
{
    EngineWorkerGuard access(this);

    auto arraySizes = _cudaSimulation->getArraySizes();
    DataAccessTO dataTO =
        _dataTOCache->getDataTO({arraySizes.cellArraySize, arraySizes.particleArraySize});
    _cudaSimulation->getSelectedSimulationData(includeClusters, dataTO);

    DataConverter converter(_settings.simulationParameters);

    auto result = converter.convertAccessTOtoDataDescription(dataTO);
    _dataTOCache->releaseDataTO(dataTO);

    return result;
}

DataDescription EngineWorker::getInspectedSimulationData(std::vector<uint64_t> entityIds)
{
    EngineWorkerGuard access(this);

    auto arraySizes = _cudaSimulation->getArraySizes();
    DataAccessTO dataTO =
        _dataTOCache->getDataTO({arraySizes.cellArraySize, arraySizes.particleArraySize});
    _cudaSimulation->getInspectedSimulationData(entityIds, dataTO);

    DataConverter converter(_settings.simulationParameters);

    auto result = converter.convertAccessTOtoDataDescription(dataTO, DataConverter::SortTokens::Yes);
    _dataTOCache->releaseDataTO(dataTO);

    return result;
}

MonitorData EngineWorker::getMonitorData() const
{
    std::lock_guard guard(_mutexForStatistics);

    return _lastStatistics;
}

namespace
{
    struct NumberOfEntities
    {
        int cells = 0;
        int particles = 0;
        int tokens = 0;
    };
    NumberOfEntities getNumberOfEntities(ClusteredDataDescription const& data)
    {
        NumberOfEntities result;
        for (auto const& cluster : data.clusters) {
            result.cells += cluster.cells.size();
        }
        result.particles = data.particles.size();
        return result;
    }
    NumberOfEntities getNumberOfEntities(DataDescription const& data)
    {
        NumberOfEntities result;
        result.cells = data.cells.size();
        result.particles = data.particles.size();
        return result;
    }
}

void EngineWorker::addAndSelectSimulationData(DataDescription const& dataToUpdate)
{
    auto numberOfEntities = getNumberOfEntities(dataToUpdate);

    EngineWorkerGuard access(this);

    _cudaSimulation->resizeArraysIfNecessary({numberOfEntities.cells, numberOfEntities.particles});

    DataAccessTO dataTO = provideTO();

    DataConverter converter(_settings.simulationParameters);
    converter.convertDataDescriptionToAccessTO(dataTO, dataToUpdate);

    _cudaSimulation->addAndSelectSimulationData(dataTO);
    updateMonitorDataIntern();

    _dataTOCache->releaseDataTO(dataTO);
}

void EngineWorker::setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate)
{
    auto numberOfEntities = getNumberOfEntities(dataToUpdate);

    EngineWorkerGuard access(this);

    _cudaSimulation->resizeArraysIfNecessary({numberOfEntities.cells, numberOfEntities.particles});

    DataAccessTO dataTO = provideTO();

    DataConverter converter(_settings.simulationParameters);
    converter.convertClusteredDataDescriptionToAccessTO(dataTO, dataToUpdate);

    _cudaSimulation->setSimulationData(dataTO);
    updateMonitorDataIntern();

    _dataTOCache->releaseDataTO(dataTO);
}

void EngineWorker::setSimulationData(DataDescription const& dataToUpdate)
{
    auto numberOfEntities = getNumberOfEntities(dataToUpdate);

    EngineWorkerGuard access(this);

    _cudaSimulation->resizeArraysIfNecessary({numberOfEntities.cells, numberOfEntities.particles});

    DataAccessTO dataTO = provideTO();

    DataConverter converter(_settings.simulationParameters);
    converter.convertDataDescriptionToAccessTO(dataTO, dataToUpdate);

    _cudaSimulation->setSimulationData(dataTO);
    updateMonitorDataIntern();

    _dataTOCache->releaseDataTO(dataTO);
}

void EngineWorker::removeSelectedEntities(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _cudaSimulation->removeSelectedEntities(includeClusters);
    updateMonitorDataIntern();
}

void EngineWorker::relaxSelectedEntities(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _cudaSimulation->relaxSelectedEntities(includeClusters);
}

void EngineWorker::uniformVelocitiesForSelectedEntities(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _cudaSimulation->uniformVelocitiesForSelectedEntities(includeClusters);
}

void EngineWorker::makeSticky(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _cudaSimulation->makeSticky(includeClusters);
}

void EngineWorker::removeStickiness(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _cudaSimulation->removeStickiness(includeClusters);
}

void EngineWorker::setBarrier(bool value, bool includeClusters)
{
    EngineWorkerGuard access(this);

    _cudaSimulation->setBarrier(value, includeClusters);
}

void EngineWorker::changeCell(CellDescription const& changedCell)
{
    EngineWorkerGuard access(this);

    auto dataTO = provideTO();

    DataConverter converter(_settings.simulationParameters);
    converter.convertCellDescriptionToAccessTO(dataTO, changedCell);

    _cudaSimulation->changeInspectedSimulationData(dataTO);

    _dataTOCache->releaseDataTO(dataTO);
}

void EngineWorker::changeParticle(ParticleDescription const& changedParticle)
{
    EngineWorkerGuard access(this);

    auto dataTO = provideTO();

    DataConverter converter(_settings.simulationParameters);
    converter.convertParticleDescriptionToAccessTO(dataTO, changedParticle);

    _cudaSimulation->changeInspectedSimulationData(dataTO);

    _dataTOCache->releaseDataTO(dataTO);
}

void EngineWorker::calcSingleTimestep()
{
    EngineWorkerGuard access(this);

    _cudaSimulation->calcTimestep();
    updateMonitorDataIntern();
}

void EngineWorker::beginShutdown()
{
    _isShutdown.store(true);
}

void EngineWorker::endShutdown()
{
    _isSimulationRunning = false;
    _isShutdown = false;
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
    EngineWorkerGuard access(this);
    _cudaSimulation->setCurrentTimestep(value);
    resetProcessMonitorData();
}

void EngineWorker::setSimulationParameters_async(SimulationParameters const& parameters)
{
    std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
    _updateSimulationParametersJob = parameters;
}

void EngineWorker::setSimulationParametersSpots_async(SimulationParametersSpots const& spots)
{
    std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
    _updateSimulationParametersSpotsJob = spots;
}

void EngineWorker::setGpuSettings_async(GpuSettings const& gpuSettings)
{
    std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
    _updateGpuSettingsJob = gpuSettings;
}

void EngineWorker::setFlowFieldSettings_async(FlowFieldSettings const& flowFieldSettings)
{
    std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
    _flowFieldSettings = flowFieldSettings;
}

void EngineWorker::applyForce_async(
    RealVector2D const& start,
    RealVector2D const& end,
    RealVector2D const& force,
    float radius)
{
    std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
    _applyForceJobs.emplace_back(ApplyForceJob{start, end, force, radius});
}

void EngineWorker::switchSelection(RealVector2D const& pos, float radius)
{
    EngineWorkerGuard access(this);
    _cudaSimulation->switchSelection(PointSelectionData{{pos.x, pos.y}, radius});
}

void EngineWorker::swapSelection(RealVector2D const& pos, float radius)
{
    EngineWorkerGuard access(this);
    _cudaSimulation->swapSelection(PointSelectionData{{pos.x, pos.y}, radius});
}

SelectionShallowData EngineWorker::getSelectionShallowData()
{
    EngineWorkerGuard access(this);
    return _cudaSimulation->getSelectionShallowData();
}

void EngineWorker::setSelection(RealVector2D const& startPos, RealVector2D const& endPos)
{
    EngineWorkerGuard access(this);
    _cudaSimulation->setSelection(AreaSelectionData{{startPos.x, startPos.y}, {endPos.x, endPos.y}});
}

void EngineWorker::removeSelection()
{
    EngineWorkerGuard access(this);
    _cudaSimulation->removeSelection();

    updateMonitorDataIntern();
}

void EngineWorker::updateSelection()
{
    EngineWorkerGuard access(this);
    _cudaSimulation->updateSelection();
}

void EngineWorker::shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& updateData)
{
    EngineWorkerGuard access(this);
    _cudaSimulation->shallowUpdateSelectedEntities(updateData);

    updateMonitorDataIntern();
}

void EngineWorker::colorSelectedEntities(unsigned char color, bool includeClusters)
{
    EngineWorkerGuard access(this);
    _cudaSimulation->colorSelectedEntities(color, includeClusters);

    updateMonitorDataIntern();
}

void EngineWorker::reconnectSelectedEntities()
{
    EngineWorkerGuard access(this);
    _cudaSimulation->reconnectSelectedEntities();
}

void EngineWorker::runThreadLoop()
{
    try {
        std::mutex mutexForLoop;
        std::unique_lock<std::mutex> lockForLoop(mutexForLoop);

        while (!_isShutdown.load()) {

            if (_accessState == 0) {
                if (_isSimulationRunning.load()) {
                    _cudaSimulation->calcTimestep();
                    if (++_monitorCounter == 3) {  //for performance reasons...
                        updateMonitorDataIntern(true);
                        _monitorCounter = 0;
                    }
                }
                measureTPS();
                slowdownTPS();
            }

            processJobs();

            if (_accessState == 1) {
                _accessState = 2;
            }
        }
    } catch (std::exception const& e) {
        std::unique_lock<std::mutex> uniqueLock(_exceptionData.mutex);
        _exceptionData.errorMessage = e.what();
    }
}

void EngineWorker::runSimulation()
{
    _isSimulationRunning.store(true);
}

void EngineWorker::pauseSimulation()
{
    EngineWorkerGuard access(this);
    _isSimulationRunning.store(false);
}

bool EngineWorker::isSimulationRunning() const
{
    return _isSimulationRunning.load();
}

DataAccessTO EngineWorker::provideTO()
{
    auto arraySizes = _cudaSimulation->getArraySizes();
    return _dataTOCache->getDataTO({arraySizes.cellArraySize, arraySizes.particleArraySize});
}

void EngineWorker::resetProcessMonitorData()
{
    std::lock_guard guard(_mutexForStatistics);
    _cudaSimulation->resetProcessMonitorData();
}

void EngineWorker::updateMonitorDataIntern(bool afterMinDuration)
{
    auto now = std::chrono::steady_clock::now();
    if (!afterMinDuration  || !_lastMonitorUpdate || now - *_lastMonitorUpdate > MonitorUpdate) {

        std::lock_guard guard(_mutexForStatistics);
        _lastStatistics = _cudaSimulation->getMonitorData();
        _lastMonitorUpdate = now;
    }
}

void EngineWorker::processJobs()
{
    std::unique_lock<std::mutex> asyncJobsLock(_mutexForAsyncJobs);
    if (_updateSimulationParametersJob) {
        _cudaSimulation->setSimulationParameters(*_updateSimulationParametersJob);
        _updateSimulationParametersJob = std::nullopt;
    }
    if (_updateSimulationParametersSpotsJob) {
        _cudaSimulation->setSimulationParametersSpots(*_updateSimulationParametersSpotsJob);
        _updateSimulationParametersSpotsJob = std::nullopt;
    }
    if (_updateGpuSettingsJob) {
        _cudaSimulation->setGpuConstants(*_updateGpuSettingsJob);
        _updateGpuSettingsJob = std::nullopt;
    }
    if (_flowFieldSettings) {
        _cudaSimulation->setFlowFieldSettings(*_flowFieldSettings);
        _flowFieldSettings = std::nullopt;
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

void EngineWorker::waitAndAllowAccess(std::chrono::microseconds const& duration)
{
    auto startTimepoint = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTimepoint) < duration) {
        if (_accessState == 1) {
            _accessState = 2;
        }
    }
}

void EngineWorker::measureTPS()
{
    if (_isSimulationRunning.load()) {
        auto timepoint = std::chrono::steady_clock::now();
        if (!_measureTimepoint) {
            _measureTimepoint = timepoint;
        } else {
            int duration = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(timepoint - *_measureTimepoint).count());
            if (duration > 199) {
                _measureTimepoint = timepoint;
                if (duration < 350) {
                    _tps.store(toFloat(_timestepsSinceMeasurement) * 5 * 200 / duration);
                } else {
                    _tps.store(1000.0f / duration);
                }
                _timestepsSinceMeasurement = 0;
            }
        }
        ++_timestepsSinceMeasurement;
    } else {
        _tps.store(0);
    }
}

void EngineWorker::slowdownTPS()
{
    if (_slowDownTimepoint) {
        auto timestepDuration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - *_slowDownTimepoint);
        if (_slowDownOvershot) {
            timestepDuration += *_slowDownOvershot;
        }
        auto tpsRestriction = _tpsRestriction.load();
        if (_isSimulationRunning.load() && tpsRestriction > 0) {
            auto desiredDuration = std::chrono::microseconds(1000000 / tpsRestriction);
            if (desiredDuration > timestepDuration) {
                waitAndAllowAccess(desiredDuration - timestepDuration);
            } else {
            }
            _slowDownOvershot = std::min(std::max(timestepDuration - desiredDuration, std::chrono::microseconds(0)), desiredDuration);
        }
    }
    _slowDownTimepoint = std::chrono::steady_clock::now();
}

EngineWorkerGuard::EngineWorkerGuard(EngineWorker* worker, std::optional<std::chrono::milliseconds> const& maxDuration)
    : _worker(worker)
{
    worker->_accessState = 1;

    auto startTimepoint = std::chrono::steady_clock::now();
    while (worker->_accessState == 1) {
        auto timePassed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTimepoint);
        if (maxDuration) {
            if (timePassed > *maxDuration) {
                break;
            }
        } else {
            if (timePassed > std::chrono::seconds(5)) {
                _isTimeout = true;
                throw std::runtime_error("GPU Timeout");
            }
        }
    }

    checkForException(worker->_exceptionData);
}

EngineWorkerGuard::~EngineWorkerGuard()
{
    _worker->_accessState = 0;
}

bool EngineWorkerGuard::isTimeout() const
{
    return _isTimeout;
}

void EngineWorkerGuard::checkForException(ExceptionData const& exceptionData)
{
    std::unique_lock<std::mutex> uniqueLock(exceptionData.mutex);
    if (exceptionData.errorMessage) {
        throw std::runtime_error(*exceptionData.errorMessage);
    }
}
