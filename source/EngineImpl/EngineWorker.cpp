#include "EngineWorker.h"

#include <algorithm>
#include <chrono>

#include <Base/ExitScopeGuard.h>

#include <EngineInterface/DescEditService.h>
#include <EngineInterface/GeometryBuffers.h>
#include <EngineInterface/Ids.h>
#include <EngineInterface/NumberGenerator.h>

#include <EngineGpuKernels/TOProvider.cuh>
#include <EngineGpuKernels/TOs.cuh>

#include "DescConverterService.h"

#include "SimulationCudaFacade.cuh"

namespace
{
    std::chrono::milliseconds const FrameTimeout(500);
}

void EngineWorker::newSimulation(uint64_t timestep, SettingsForSimulation const& settings)
{
    _accessState = 0;
    _settings = settings;
    _collectionTOProvider = std::make_shared<_TOProvider>();
    _simulationCudaFacade = std::make_shared<_SimulationCudaFacade>(timestep, settings);
}

void EngineWorker::clear()
{
    EngineWorkerGuard access(this);
    return _simulationCudaFacade->clear();
}

std::string EngineWorker::getGpuName() const
{
    return _SimulationCudaFacade::checkAndReturnGpuInfo().gpuModelName;
}

void EngineWorker::tryCopyBuffersFromCudaToOpenGL(GeometryBuffers const& geometryBuffers, RealRect const& visibleWorldRect)
{
    EngineWorkerGuard access(this, FrameTimeout);

    if (!access.isTimeout()) {
        _simulationCudaFacade->copyBuffersFromCudaToOpenGL(geometryBuffers, visibleWorldRect);
        syncSimulationWithRenderingIfDesired();
    }
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

Desc EngineWorker::getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    EngineWorkerGuard access(this);

    auto dataTO = _simulationCudaFacade->getSimulationData({rectUpperLeft.x, rectUpperLeft.y}, int2{rectLowerRight.x, rectLowerRight.y});
    ExitScopeGuard guard([&dataTO]() { _TOProvider::destroyUnmanagedDataTO(dataTO); });

    return DescConverterService::get().convertTOtoDescription(dataTO);
}

Desc EngineWorker::getSelectedSimulationData(bool includeClusters)
{
    EngineWorkerGuard access(this);

    auto dataTO = _simulationCudaFacade->getSelectedSimulationData(includeClusters);

    return DescConverterService::get().convertTOtoDescription(dataTO);
}

Desc EngineWorker::getInspectedSimulationData(std::vector<uint64_t> objectsIds)
{
    EngineWorkerGuard access(this);

    auto dataTO = _simulationCudaFacade->getInspectedSimulationData(objectsIds);

    return DescConverterService::get().convertTOtoDescription(dataTO);
}

StatisticsRawData EngineWorker::getStatisticsRawData() const
{
    return _simulationCudaFacade->getStatisticsRawData();
}

StatisticsHistory const& EngineWorker::getStatisticsHistory() const
{
    return _simulationCudaFacade->getStatisticsHistory();
}

void EngineWorker::setStatisticsHistory(StatisticsHistoryData const& data)
{
    _simulationCudaFacade->setStatisticsHistory(data);
}

void EngineWorker::addAndSelectSimulationData(Desc&& dataToUpdate)
{
    EngineWorkerGuard access(this);

    auto maxIds = _simulationCudaFacade->getMaxIds();
    NumberGenerator::get().adaptMaxIds(maxIds);

    dataToUpdate.assignNewEntityIds();

    auto dataTO = DescConverterService::get().convertDescriptionToTO(dataToUpdate);

    _simulationCudaFacade->addAndSelectSimulationData(dataTO);
}

void EngineWorker::setSimulationData(Desc const& dataToUpdate)
{
    if (!dataToUpdate.hasUniqueIds()) {
        throw std::runtime_error("Object ids are not unique.");
    }

    EngineWorkerGuard access(this);

    auto dataTO = DescConverterService::get().convertDescriptionToTO(dataToUpdate);

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

void EngineWorker::changeCell(ExtendedObjectDesc const& changedCell)
{
    EngineWorkerGuard access(this);

    auto dataTO = DescConverterService::get().convertDescriptionToTO(changedCell);

    _simulationCudaFacade->changeInspectedSimulationData(dataTO);
}

void EngineWorker::changeParticle(EnergyDesc const& changedParticle)
{
    EngineWorkerGuard access(this);

    auto dataTO = DescConverterService::get().convertDescriptionToTO(changedParticle);

    _simulationCudaFacade->changeInspectedSimulationData(dataTO);
}

int EngineWorker::injectGenomeToSelectedCreatures(GenomeDesc const& genome)
{
    EngineWorkerGuard access(this);

    auto dataTO = DescConverterService::get().convertDescriptionToTO(genome);

    return _simulationCudaFacade->injectGenomeToSelectedCreatures(dataTO);
}

void EngineWorker::calcTimesteps(uint64_t timesteps)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->calcTimesteps(timesteps, true);
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
    if (_simulationCudaFacade == nullptr) {
        return 0ull;
    }
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

void EngineWorker::setSimulationParameters(SimulationParameters const& parameters, SimulationParametersUpdateConfig const& updateConfig)
{
    _simulationCudaFacade->setSimulationParameters(parameters, updateConfig);
}

void EngineWorker::setGpuSettings_async(CudaSettings const& gpuSettings)
{
    std::unique_lock<std::mutex> uniqueLock(_mutexForAsyncJobs);
    _updateGpuSettingsJob = gpuSettings;
}

void EngineWorker::applyForce_async(RealVector2D const& start, RealVector2D const& end, RealVector2D const& force, float radius)
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

SelectionShallowData EngineWorker::getSelectionShallowData()
{
    EngineWorkerGuard access(this);
    return _simulationCudaFacade->getSelectionShallowData();
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
                    _simulationCudaFacade->calcTimesteps(1, false);
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

Desc EngineWorker::getPreviewData()
{
    EngineWorkerGuard access(this);

    auto preview = _simulationCudaFacade->getPreviewData();
    ExitScopeGuard guard([&preview]() { _TOProvider::destroyUnmanagedDataTO(preview); });

    return DescConverterService::get().convertTOtoDescription(preview);
}

void EngineWorker::setPreviewData(Desc const& description)
{
    if (!description.hasUniqueIds()) {
        throw std::runtime_error("Cell ids are not unique.");
    }

    EngineWorkerGuard access(this);

    auto dataTO = DescConverterService::get().convertDescriptionToTO(description);

    for (uint64_t i = 0; i < *dataTO.numObjects; ++i) {
        if (dataTO.objects[i].type == ObjectType_Cell) {
            dataTO.objects[i].typeData.cell.lastUpdate = 0;
        }
    }

    _simulationCudaFacade->newPreview(dataTO);
}

void EngineWorker::calcTimestepsForPreview(std::chrono::milliseconds const& duration, bool detailSimulation)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->calcTimestepsForPreview(duration, detailSimulation);
}

void EngineWorker::calcTimestepsForPreview(int numSteps, bool detailSimulation)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->calcTimestepsForPreview(numSteps, detailSimulation);
}

uint64_t EngineWorker::getCurrentTimestepForPreview()
{
    return _simulationCudaFacade->getCurrentTimestepForPreview();
}

void EngineWorker::setCurrentTimestepForPreview(uint64_t timestep)
{
    _simulationCudaFacade->setCurrentTimestepForPreview(timestep);
}

void EngineWorker::testOnly_mutate(uint64_t objectId)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_mutate(objectId);
}

void EngineWorker::testOnly_createConnection(uint64_t objectId1, uint64_t objectId2)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_createConnection(objectId1, objectId2);
}

void EngineWorker::testOnly_cleanupAfterTimestep()
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_cleanupAfterTimestep();
}

void EngineWorker::testOnly_cleanupAfterDataManipulation()
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_cleanupAfterDataManipulation();
}

void EngineWorker::testOnly_resizeArrays(ArraySizesForGpuEntities const& sizeDelta)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_resizeArrays(sizeDelta);
}

bool EngineWorker::testOnly_arePointersValid()
{
    EngineWorkerGuard access(this);
    return _simulationCudaFacade->testOnly_arePointersValid();
}

void EngineWorker::testOnly_calcTimestepWithCellFunctions()
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_calcTimestepWithCellTypeFunctions();
}

void EngineWorker::testOnly_calcTimestepWithCellFunctionsForPreview(bool detailSimulation)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_calcTimestepWithCellTypeFunctionsForPreview(detailSimulation);
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
    _worker->_mutexForEngineWorkerGuard.lock();
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
    _worker->_mutexForEngineWorkerGuard.unlock();
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
