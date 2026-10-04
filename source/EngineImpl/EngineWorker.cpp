#include "EngineWorker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ranges>
#include <unordered_map>
#include <unordered_set>

#include <Base/ExitScopeGuard.h>
#include <Base/GlobalSettings.h>
#include <Base/KernelProfiler.h>
#include <Base/KernelTracer.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>

#include <Base/Ids.h>
#include <Base/NumberGenerator.h>

#include <Data/DescEditService.h>

#include <EngineInterface/GeometryBuffers.h>

#include <EngineKernels/TOProvider.cuh>
#include <EngineKernels/TOs.cuh>

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

namespace
{
    // Every domain contributes its own objects together with their creatures and genomes; ghost copies are dropped
    ContentDesc mergeDomainData(std::vector<TOs> const& dataTOs)
    {
        ContentDesc result;
        std::unordered_set<uint64_t> creatureIds;
        std::unordered_set<uint64_t> genomeIds;
        for (auto const& dataTO : dataTOs) {
            std::unordered_set<uint64_t> ghostIds;
            for (uint64_t i = 0; i < *dataTO.numObjects; ++i) {
                if (dataTO.objects[i].isGhost()) {
                    ghostIds.insert(dataTO.objects[i].id);
                }
            }
            auto domainData = DescConverterService::get().convertTOtoDescription(dataTO);

            std::unordered_set<uint64_t> referencedCreatureIds;
            for (auto& object : domainData._objects) {
                if (ghostIds.contains(object._id)) {
                    continue;
                }
                if (object.getObjectType() == ObjectType_Cell) {
                    referencedCreatureIds.insert(object.getCellRef()._creatureId);
                }
                result._objects.emplace_back(std::move(object));
            }
            for (auto& energy : domainData._energies) {
                result._energies.emplace_back(std::move(energy));
            }

            std::unordered_set<uint64_t> referencedGenomeIds;
            for (auto& creature : domainData._creatures) {
                if (referencedCreatureIds.contains(creature._id) && creatureIds.insert(creature._id).second) {
                    referencedGenomeIds.insert(creature._genomeId);
                    result._creatures.emplace_back(std::move(creature));
                }
            }
            for (auto& genome : domainData._genomes) {
                if (referencedGenomeIds.contains(genome._id) && genomeIds.insert(genome._id).second) {
                    result._genomes.emplace_back(std::move(genome));
                }
            }
        }
        return result;
    }
}

ContentDesc EngineWorker::getSimulationData(IntVector2D const& rectUpperLeft, IntVector2D const& rectLowerRight)
{
    std::vector<TOs> dataTOs;
    {
        EngineWorkerGuard access(this);

        dataTOs = _simulationCudaFacade->getSimulationData({rectUpperLeft.x, rectUpperLeft.y}, int2{rectLowerRight.x, rectLowerRight.y});
    }
    ExitScopeGuard guard([&dataTOs]() {
        for (auto& dataTO : dataTOs) {
            _TOProvider::destroyUnmanagedDataTO(dataTO);
        }
    });

    if (dataTOs.size() == 1) {
        return DescConverterService::get().convertTOtoDescription(dataTOs.front());
    }
    return mergeDomainData(dataTOs);
}

ContentDesc EngineWorker::getSelectedSimulationData(bool includeClusters)
{
    EngineWorkerGuard access(this);

    auto dataTO = _simulationCudaFacade->getSelectedSimulationData(includeClusters);

    return DescConverterService::get().convertTOtoDescription(dataTO);
}

ContentDesc EngineWorker::getInspectedSimulationData(std::vector<uint64_t> objectsIds)
{
    EngineWorkerGuard access(this);

    auto dataTO = _simulationCudaFacade->getInspectedSimulationData(objectsIds);

    return DescConverterService::get().convertTOtoDescription(dataTO);
}

StatisticsHistory const& EngineWorker::getStatisticsHistory() const
{
    return _simulationCudaFacade->getStatisticsHistory();
}

void EngineWorker::setStatisticsHistory(StatisticsHistoryData const& data)
{
    _simulationCudaFacade->setStatisticsHistory(data);
}

StatisticsEntry EngineWorker::getStatisticsEntry() const
{
    return _simulationCudaFacade->getStatisticsEntry();
}

void EngineWorker::addAndSelectSimulationData(ContentDesc&& dataToUpdate)
{
    EngineWorkerGuard access(this);

    auto maxIds = _simulationCudaFacade->getMaxIds();
    NumberGenerator::get().adaptMaxIds(maxIds);

    dataToUpdate.assignNewEntityIds();

    auto dataTO = DescConverterService::get().convertDescriptionToTO(dataToUpdate);

    _simulationCudaFacade->addAndSelectSimulationData(dataTO);
}

void EngineWorker::setSimulationData(ContentDesc const& dataToUpdate)
{
    if (!dataToUpdate.hasUniqueIds()) {
        throw AlienException("Object ids are not unique.");
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

void EngineWorker::setStatic(bool value, bool includeClusters)
{
    EngineWorkerGuard access(this);

    _simulationCudaFacade->setStatic(value, includeClusters);
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

    // Developer aid: ALIEN_CHECK_DOMAINS=n validates a simulation split into domains every n time steps
    static auto const checkInterval = [] {
        auto value = std::getenv("ALIEN_CHECK_DOMAINS");
        return value ? std::max<uint64_t>(1, std::strtoull(value, nullptr, 10)) : uint64_t(0);
    }();
    if (checkInterval == 0) {
        _simulationCudaFacade->calcTimesteps(timesteps, true);
        return;
    }
    for (uint64_t calculated = 0; calculated < timesteps;) {
        auto chunk = std::min(checkInterval, timesteps - calculated);
        _simulationCudaFacade->calcTimesteps(chunk, true);
        calculated += chunk;

        auto errors = testOnly_getDomainConsistencyErrors();
        for (auto const& error : errors) {
            log(Priority::Important, "domain consistency at time step " + std::to_string(getCurrentTimestep()) + ": " + error);
        }
        if (!errors.empty()) {
            throw std::runtime_error("The domains of the simulation are inconsistent.");
        }
    }
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
}

SimulationParameters EngineWorker::getSimulationParameters() const
{
    return _simulationCudaFacade->getSimulationParameters();
}

void EngineWorker::setSimulationParameters(SimulationParameters const& parameters, SimulationParametersUpdateConfig const& updateConfig)
{
    _simulationCudaFacade->setSimulationParameters(parameters, updateConfig);
}

void EngineWorker::setDebugMode(bool value)
{
    EngineWorkerGuard access(this);

    GlobalSettings::get().setDebugMode(value);
    if (value) {
        KernelProfiler::get().init(Const::ProfileFilename);
        KernelTracer::get().init(Const::TraceFilename);
    } else {
        KernelProfiler::get().close();
        KernelTracer::get().close();
    }
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

void EngineWorker::glueSelectedObjects(bool includeClusters)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->glueSelectedObjects(includeClusters);
}

void EngineWorker::cutConnections(RealVector2D const& start, RealVector2D const& end, bool onlySelected, bool includeClusters)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->cutConnections({start.x, start.y}, {end.x, end.y}, onlySelected, includeClusters);
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
    } catch (AlienException const& e) {
        std::unique_lock<std::mutex> uniqueLock(_exceptionData.mutex);
        _exceptionData.errorMessage = std::string(e.what()) + "\nCallstack:\n" + e.getCallstack();
    } catch (std::exception const& e) {
        std::unique_lock<std::mutex> uniqueLock(_exceptionData.mutex);
        _exceptionData.errorMessage = e.what();
    } catch (...) {
        std::unique_lock<std::mutex> uniqueLock(_exceptionData.mutex);
        _exceptionData.errorMessage = "An unknown exception occurred in the GPU worker thread.";
    }
}

void EngineWorker::checkAndThrowException() const
{
    std::unique_lock<std::mutex> uniqueLock(_exceptionData.mutex);
    if (_exceptionData.errorMessage) {
        throw std::runtime_error(*_exceptionData.errorMessage);
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

ContentDesc EngineWorker::getPreviewData()
{
    EngineWorkerGuard access(this);

    auto preview = _simulationCudaFacade->getPreviewData();
    ExitScopeGuard guard([&preview]() { _TOProvider::destroyUnmanagedDataTO(preview); });

    return DescConverterService::get().convertTOtoDescription(preview);
}

void EngineWorker::setPreviewData(ContentDesc const& description)
{
    if (!description.hasUniqueIds()) {
        throw std::runtime_error("Cell ids are not unique.");
    }

    EngineWorkerGuard access(this);

    auto dataTO = DescConverterService::get().convertDescriptionToTO(description);

    auto numObjects = *dataTO.numObjects;
    for (uint64_t i = 0; i < numObjects; ++i) {
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

void EngineWorker::testOnly_voidUnreachableNodes(uint64_t objectId)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_voidUnreachableNodes(objectId);
}

void EngineWorker::testOnly_removeUnusedGenes(uint64_t objectId)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_removeUnusedGenes(objectId);
}

void EngineWorker::testOnly_removeGeneCycles(uint64_t objectId)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_removeGeneCycles(objectId);
}

void EngineWorker::testOnly_limitGenesWithSeparation(uint64_t objectId)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_limitGenesWithSeparation(objectId);
}

void EngineWorker::testOnly_createConnection(uint64_t objectId1, uint64_t objectId2)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_createConnection(objectId1, objectId2);
}

void EngineWorker::testOnly_createConnectionWithAbsAngle(
    uint64_t objectId1,
    uint64_t objectId2,
    float desiredDistance,
    float desiredAbsAngle1,
    float desiredAbsAngle2)
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_createConnectionWithAbsAngle(objectId1, objectId2, desiredDistance, desiredAbsAngle1, desiredAbsAngle2);
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

bool EngineWorker::testOnly_isDataValid()
{
    EngineWorkerGuard access(this);
    return _simulationCudaFacade->testOnly_isDataValid();
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

void EngineWorker::testOnly_zeroTransferData()
{
    EngineWorkerGuard access(this);
    _simulationCudaFacade->testOnly_zeroTransferData();
}

void EngineWorker::testOnly_syncNumberGenerator()
{
    EngineWorkerGuard access(this);
    auto maxIds = _simulationCudaFacade->getMaxIds();
    NumberGenerator::get().adaptMaxIds(maxIds);
}

namespace
{
    struct DomainView
    {
        ContentDesc data;
        std::unordered_set<uint64_t> ghostIds;
        std::unordered_map<uint64_t, ObjectDesc const*> objectById;
    };

    float calcTorusDistance(RealVector2D const& pos1, RealVector2D const& pos2, IntVector2D const& worldSize)
    {
        auto dx = std::abs(pos1.x - pos2.x);
        auto dy = std::abs(pos1.y - pos2.y);
        dx = (std::min)(dx, toFloat(worldSize.x) - dx);
        dy = (std::min)(dy, toFloat(worldSize.y) - dy);
        return std::sqrt(dx * dx + dy * dy);
    }

    std::vector<std::string> checkDomainConsistency(std::vector<TOs> const& dataTOs, IntVector2D const& worldSize)
    {
        auto constexpr MaxGhostDeviation = 2.0f;
        auto constexpr MaxErrors = 100;

        std::vector<std::string> result;
        auto addError = [&](std::string const& error) {
            if (result.size() < MaxErrors) {
                result.emplace_back(error);
            }
        };

        std::vector<DomainView> views(dataTOs.size());
        for (auto const& [dataTO, view] : std::views::zip(dataTOs, views)) {
            for (uint64_t i = 0; i < *dataTO.numObjects; ++i) {
                if (dataTO.objects[i].isGhost()) {
                    view.ghostIds.insert(dataTO.objects[i].id);
                }
            }
            view.data = DescConverterService::get().convertTOtoDescription(dataTO);
            for (auto const& object : view.data._objects) {
                view.objectById.emplace(object._id, &object);
            }
        }

        std::unordered_map<uint64_t, int> ownerById;
        for (auto const& [domainIndex, view] : std::views::enumerate(views)) {
            for (auto const& object : view.data._objects) {
                if (view.ghostIds.contains(object._id)) {
                    continue;
                }
                auto [iter, inserted] = ownerById.emplace(object._id, toInt(domainIndex));
                if (!inserted) {
                    addError(
                        "object " + std::to_string(object._id) + " is owned by domain " + std::to_string(iter->second) + " and domain "
                        + std::to_string(domainIndex));
                }
            }
        }

        for (auto const& [domainIndex, view] : std::views::enumerate(views)) {
            for (auto const& object : view.data._objects) {
                auto isGhost = view.ghostIds.contains(object._id);
                auto ownerIter = ownerById.find(object._id);
                if (ownerIter == ownerById.end()) {
                    addError("ghost " + std::to_string(object._id) + " in domain " + std::to_string(domainIndex) + " has no owner");
                    continue;
                }
                if (isGhost) {
                    auto const& original = *views.at(ownerIter->second).objectById.at(object._id);
                    if (calcTorusDistance(object._pos, original._pos, worldSize) > MaxGhostDeviation) {
                        addError("ghost " + std::to_string(object._id) + " in domain " + std::to_string(domainIndex) + " deviates from its original");
                    }
                    continue;
                }
                for (auto const& connection : object._connections) {
                    if (!view.objectById.contains(connection._objectId)) {
                        addError(
                            "object " + std::to_string(object._id) + " in domain " + std::to_string(domainIndex) + " is connected to the absent object "
                            + std::to_string(connection._objectId));
                        continue;
                    }
                    auto partnerOwnerIter = ownerById.find(connection._objectId);
                    if (partnerOwnerIter == ownerById.end()) {
                        continue;
                    }
                    auto const& partner = *views.at(partnerOwnerIter->second).objectById.at(connection._objectId);
                    if (!partner.isConnectedTo(object._id)) {
                        auto describe = [](ObjectDesc const& o) {
                            auto result = std::to_string(o._id) + " (type " + std::to_string(o.getObjectType()) + " at " + std::to_string(o._pos.x) + ", "
                                + std::to_string(o._pos.y) + ", " + std::to_string(o._connections.size()) + " connections";
                            if (o.getObjectType() == ObjectType_Cell) {
                                auto const& cell = o.getCellRef();
                                result += ", creature " + std::to_string(cell._creatureId) + ", state " + std::to_string(cell._cellState)
                                    + (cell._headCell ? ", head" : "") + (cell._constructor ? ", constructor" : "");
                            }
                            return result + ")";
                        };
                        addError(
                            "connection " + describe(object) + " in domain " + std::to_string(domainIndex) + " - " + describe(partner)
                            + " is missing in domain " + std::to_string(partnerOwnerIter->second));
                    }
                }
            }
        }
        return result;
    }
}

std::vector<std::string> EngineWorker::testOnly_getDomainConsistencyErrors()
{
    std::vector<TOs> dataTOs;
    {
        EngineWorkerGuard access(this);

        // After two syncs the ghosts are fresh and all changes across domains and their replies have been delivered
        _simulationCudaFacade->syncDomains();
        _simulationCudaFacade->syncDomains();
        dataTOs = _simulationCudaFacade->getSimulationData({-10, -10}, {_settings.worldSizeX + 10, _settings.worldSizeY + 10});
    }
    ExitScopeGuard guard([&dataTOs]() {
        for (auto& dataTO : dataTOs) {
            _TOProvider::destroyUnmanagedDataTO(dataTO);
        }
    });
    return checkDomainConsistency(dataTOs, {_settings.worldSizeX, _settings.worldSizeY});
}


void EngineWorker::processJobs()
{
    std::unique_lock<std::mutex> asyncJobsLock(_mutexForAsyncJobs);
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
