#include "EngineWorker.h"

#include <chrono>

#include "EngineGpuKernels/TOs.cuh"
#include "EngineGpuKernels/SimulationCudaFacade.cuh"
#include "AccessDataTOCache.h"
#include "DescriptionConverter.h"

namespace
{
    std::chrono::milliseconds const FrameTimeout(500);
}

void EngineWorker::newSimulation(uint64_t timestep, GeneralSettings const& generalSettings, SimulationParameters const& parameters)
{
    _accessState = 0;
    _settings.generalSettings = generalSettings;
    _settings.simulationParameters = parameters;
    _dataTOCache = std::make_shared<_AccessDataTOCache>();
    _simulationCudaFacade = std::make_shared<_SimulationCudaFacade>(timestep, _settings);

    if (_imageResource) {
        _cudaResource = _simulationCudaFacade->registerImageResource(*_imageResource);
    }
}

void EngineWorker::clear()
{
    EngineWorkerGuard access(this);
    return _simulationCudaFacade->clear();
}

void EngineWorker::setImageResource(void* image)
{
    GLuint imageId = reinterpret_cast<uintptr_t>(image);
    _imageResource = imageId;

    if (_simulationCudaFacade) {
        EngineWorkerGuard access(this);
        _cudaResource = _simulationCudaFacade->registerImageResource(imageId);
    }
}

std::string EngineWorker::getGpuName() const
{
    return _SimulationCudaFacade::checkAndReturnGpuInfo().gpuModelName;
}

void EngineWorker::tryDrawVectorGraphics(
    RealVector2D const& rectUpperLeft,
    RealVector2D const& rectLowerRight,
    IntVector2D const& imageSize,
    double zoom)
{
    EngineWorkerGuard access(this, FrameTimeout);

    if (!access.isTimeout()) {
        _simulationCudaFacade->drawVectorGraphics(
            {rectUpperLeft.x, rectUpperLeft.y},
            {rectLowerRight.x, rectLowerRight.y},
            _cudaResource,
            {imageSize.x, imageSize.y},
            zoom);
        syncSimulationWithRenderingIfDesired();
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
        _simulationCudaFacade->drawVectorGraphics(
            {rectUpperLeft.x, rectUpperLeft.y},
            {rectLowerRight.x, rectLowerRight.y},
            _cudaResource,
            {imageSize.x, imageSize.y},
            zoom);

        DataTO dataTO = provideTO();

        _simulationCudaFacade->getOverlayData(
            {toInt(rectUpperLeft.x), toInt(rectUpperLeft.y)},
            int2{toInt(rectLowerRight.x), toInt(rectLowerRight.y)},
            dataTO);

        DescriptionConverter converter(_settings.simulationParameters);
        auto result = converter.convertTOtoOverlayDescription(dataTO);

        syncSimulationWithRenderingIfDesired();
        return result;
    }
    return std::nullopt;
}

bool EngineWorker::isSyncSimulationWithRendering() const
{
    return _syncSimulationWithRendering;
}

void EngineWorker::setSyncSimulationWithRendering(bool value)
{
    _syncSimulationWithRendering = value;
}

int EngineWorker::getSyncSimulationWithRenderingRatio() const
{
    return _syncSimulationWithRenderingRatio;
}

void EngineWorker::setSyncSimulationWithRenderingRatio(int value)
{
    _syncSimulationWithRenderingRatio = value;
}

ClusteredDataDescription EngineWorker::getClusteredSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    EngineWorkerGuard access(this);

    DataTO dataTO = provideTO();
    
    _simulationCudaFacade->getSimulationData(
        {rectUpperLeft.x, rectUpperLeft.y}, int2{rectLowerRight.x, rectLowerRight.y}, dataTO);

    DescriptionConverter converter(_settings.simulationParameters);

    auto result = converter.convertTOtoClusteredDataDescription(dataTO);
    return result;
}

DataDescription EngineWorker::getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    EngineWorkerGuard access(this);

    DataTO dataTO = provideTO();
    
    _simulationCudaFacade->getSimulationData({rectUpperLeft.x, rectUpperLeft.y}, int2{rectLowerRight.x, rectLowerRight.y}, dataTO);

    DescriptionConverter converter(_settings.simulationParameters);

    auto result = converter.convertTOtoDataDescription(dataTO);
    return result;
}

ClusteredDataDescription EngineWorker::getSelectedClusteredSimulationData(bool includeClusters)
{
    EngineWorkerGuard access(this);

    DataTO dataTO = provideTO();
    
    _simulationCudaFacade->getSelectedSimulationData(includeClusters, dataTO);

    DescriptionConverter converter(_settings.simulationParameters);

    auto result = converter.convertTOtoClusteredDataDescription(dataTO);
    return result;
}

DataDescription EngineWorker::getSelectedSimulationData(bool includeClusters)
{
    EngineWorkerGuard access(this);

    DataTO dataTO = provideTO();
    
    _simulationCudaFacade->getSelectedSimulationData(includeClusters, dataTO);

    DescriptionConverter converter(_settings.simulationParameters);

    auto result = converter.convertTOtoDataDescription(dataTO);

    return result;
}

DataDescription EngineWorker::getInspectedSimulationData(std::vector<uint64_t> objectsIds)
{
    EngineWorkerGuard access(this);

    DataTO dataTO = provideTO();
    
    _simulationCudaFacade->getInspectedSimulationData(objectsIds, dataTO);

    DescriptionConverter converter(_settings.simulationParameters);

    auto result = converter.convertTOtoDataDescription(dataTO);
    return result;
}

RawStatisticsData EngineWorker::getRawStatistics() const
{
    return _simulationCudaFacade->getRawStatistics();
}

StatisticsHistory const& EngineWorker::getStatisticsHistory() const
{
    return _simulationCudaFacade->getStatisticsHistory();
}

void EngineWorker::setStatisticsHistory(StatisticsHistoryData const& data)
{
    _simulationCudaFacade->setStatisticsHistory(data);
}

void EngineWorker::addAndSelectSimulationData(DataDescription const& dataToUpdate)
{
    DescriptionConverter converter(_settings.simulationParameters);

    auto arraySizes = converter.getArraySizes(dataToUpdate);

    EngineWorkerGuard access(this);

    _simulationCudaFacade->resizeArraysIfNecessary(arraySizes);

    DataTO dataTO = provideTO();

    converter.convertDescriptionToTO(dataTO, dataToUpdate);

    _simulationCudaFacade->addAndSelectSimulationData(dataTO);
}

void EngineWorker::setClusteredSimulationData(ClusteredDataDescription const& dataToUpdate)
{
    DescriptionConverter converter(_settings.simulationParameters);

    EngineWorkerGuard access(this);

    _simulationCudaFacade->resizeArraysIfNecessary(converter.getArraySizes(dataToUpdate));

    DataTO dataTO = provideTO();

    converter.convertDescriptionToTO(dataTO, dataToUpdate);

    _simulationCudaFacade->setSimulationData(dataTO);
}

void EngineWorker::setSimulationData(DataDescription const& dataToUpdate)
{
    DescriptionConverter converter(_settings.simulationParameters);

    EngineWorkerGuard access(this);

    _simulationCudaFacade->resizeArraysIfNecessary(converter.getArraySizes(dataToUpdate));

    DataTO dataTO = provideTO();
    converter.convertDescriptionToTO(dataTO, dataToUpdate);

    _simulationCudaFacade->setSimulationData(dataTO);
}

void EngineWorker::removeSelectedObjects(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->removeSelectedObjects(includeClusters);
}

void EngineWorker::relaxSelectedObjects(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->relaxSelectedObjects(includeClusters);
}

void EngineWorker::uniformVelocitiesForSelectedObjects(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->uniformVelocitiesForSelectedObjects(includeClusters);
}

void EngineWorker::makeSticky(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->makeSticky(includeClusters);
}

void EngineWorker::removeStickiness(bool includeClusters)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->removeStickiness(includeClusters);
}

void EngineWorker::setBarrier(bool value, bool includeClusters)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->setBarrier(value, includeClusters);
}

void EngineWorker::changeCell(CellDescription const& changedCell)
{
    EngineWorkerGuard access(this);

    auto dataTO = provideTO();

    DescriptionConverter converter(_settings.simulationParameters);
    converter.convertDescriptionToTO(dataTO, changedCell);

    _simulationCudaFacade->changeInspectedSimulationData(dataTO);
}

void EngineWorker::changeParticle(ParticleDescription const& changedParticle)
{
    EngineWorkerGuard access(this);

    auto dataTO = provideTO();

    DescriptionConverter converter(_settings.simulationParameters);
    converter.convertDescriptionToTO(dataTO, changedParticle);

    _simulationCudaFacade->changeInspectedSimulationData(dataTO);
}

void EngineWorker::calcTimesteps(uint64_t timesteps)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->calcTimestep(timesteps, true);
}

void EngineWorker::applyCataclysm(int power)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->applyCataclysm(power);
}

void EngineWorker::beginShutdown()
{
    _isShutdown.store(true);
}

void EngineWorker::endShutdown()
{
    _isSimulationRunning = false;
    _isShutdown = false;
    _simulationCudaFacade.reset();
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
    return _simulationCudaFacade->getCurrentTimestep();
}

void EngineWorker::setCurrentTimestep(uint64_t value)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->setCurrentTimestep(value);
    resetTimeIntervalStatistics();
}

SimulationParameters EngineWorker::getSimulationParameters() const
{
    return _simulationCudaFacade->getSimulationParameters();
}

void EngineWorker::setSimulationParameters(SimulationParameters const& parameters)
{
    _simulationCudaFacade->setSimulationParameters(parameters);
}

void EngineWorker::setGpuSettings_async(GpuSettings const& gpuSettings)
{
    std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
    _updateGpuSettingsJob = gpuSettings;
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
    _simulationCudaFacade->switchSelection(PointSelectionData{{pos.x, pos.y}, radius});
}

void EngineWorker::swapSelection(RealVector2D const& pos, float radius)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->swapSelection(PointSelectionData{{pos.x, pos.y}, radius});
}

SelectionShallowData EngineWorker::getSelectionShallowData(RealVector2D const& refPos)
{
    EngineWorkerGuard access(this);
    return _simulationCudaFacade->getSelectionShallowData({refPos.x, refPos.y});
}

void EngineWorker::setSelection(RealVector2D const& startPos, RealVector2D const& endPos)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->setSelection(AreaSelectionData{{startPos.x, startPos.y}, {endPos.x, endPos.y}});
}

void EngineWorker::removeSelection()
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->removeSelection();
}

void EngineWorker::updateSelection()
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->updateSelection();
}

void EngineWorker::shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& updateData)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->shallowUpdateSelectedObjects(updateData);
}

void EngineWorker::colorSelectedObjects(unsigned char color, bool includeClusters)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->colorSelectedObjects(color, includeClusters);
}

void EngineWorker::reconnectSelectedObjects()
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->reconnectSelectedObjects();
}

void EngineWorker::setDetached(bool value)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->setDetached(value);
}

void EngineWorker::runThreadLoop()
{
    try {
        std::mutex mutexForLoop;
        std::unique_lock<std::mutex> lockForLoop(mutexForLoop);

        while (!_isShutdown.load()) {

            if (!_syncSimulationWithRendering && _accessState == 0) {
                if (_isSimulationRunning.load()) {
                    _simulationCudaFacade->calcTimestep(1, false);
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

void EngineWorker::testOnly_mutate(uint64_t cellId, MutationType mutationType)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_mutate(cellId, mutationType);
}

DataTO EngineWorker::provideTO()
{
    return _dataTOCache->getDataTO(_simulationCudaFacade->getArraySizes());
}

void EngineWorker::resetTimeIntervalStatistics()
{
    _simulationCudaFacade->resetTimeIntervalStatistics();
}

void EngineWorker::processJobs()
{
    std::unique_lock<std::mutex> asyncJobsLock(_mutexForAsyncJobs);
    if (_updateGpuSettingsJob) {
        _simulationCudaFacade->setGpuConstants(*_updateGpuSettingsJob);
        _updateGpuSettingsJob = std::nullopt;
    }
    if (!_applyForceJobs.empty()) {
        for (auto const& applyForceJob : _applyForceJobs) {
            _simulationCudaFacade->applyForce(
                {{applyForceJob.start.x, applyForceJob.start.y},
                 {applyForceJob.end.x, applyForceJob.end.y},
                 {applyForceJob.force.x, applyForceJob.force.y},
                 applyForceJob.radius,
                 false});
        }
        _applyForceJobs.clear();
    }
}

void EngineWorker::syncSimulationWithRenderingIfDesired()
{
    if (_syncSimulationWithRendering && _isSimulationRunning) {
        for (int i = 0; i < _syncSimulationWithRenderingRatio; ++i) {
            calcTimesteps(1);
            measureTPS();
            slowdownTPS();
        }
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
    checkForException(worker->_exceptionData);

    worker->_accessState = 1;

    auto startTimepoint = std::chrono::steady_clock::now();
    while (worker->_accessState == 1) {
        auto timePassed = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTimepoint);
        if (maxDuration) {
            if (timePassed > *maxDuration) {
                _isTimeout = true;
                break;
            }
        } else {
            if (timePassed > std::chrono::seconds(7)) {
                _isTimeout = true;
                throw std::runtime_error("GPU worker thread is not reachable.");
            }
        }
    }

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
        throw std::runtime_error("GPU worker thread is in an invalid state.");
    }
}
