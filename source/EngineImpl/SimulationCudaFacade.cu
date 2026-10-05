#include "SimulationCudaFacade.cuh"

#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <ranges>
#include <set>

#include <cuda/helper_cuda.h>
#include <cuda_runtime.h>

#include <Base/AlienExceptions.h>
#include <Base/GlobalSettings.h>
#include <Base/KernelProfiler.h>
#include <Base/LoggingService.h>
#include <Base/Macros.h>
#include <Base/StringHelper.h>

#include <Base/Ids.h>

#include <Data/SimulationParameters.h>
#include <Data/SpaceCalculator.h>

#include <EngineInterface/InspectedEntityIds.h>
#include <EngineInterface/KernelLaunchSettings.h>

#include <iomanip>
#include <sstream>
#include <EngineKernels/Base.cuh>
#include <EngineKernels/ConstantMemory.cuh>
#include <EngineKernels/CudaGeometryBuffers.cuh>

#include <EngineKernels/CudaMemoryManager.cuh>
#include <EngineKernels/CudaTOProvider.cuh>
#include <EngineKernels/DataAccessKernels.cuh>
#include <EngineKernels/EditKernels.cuh>
#include <EngineKernels/Entities.cuh>
#include <EngineKernels/GarbageCollectorKernels.cuh>
#include <EngineKernels/GeometryKernels.cuh>
#include <EngineKernels/ObjectGrid.cuh>
#include <EngineKernels/SelectionResult.cuh>
#include <EngineKernels/SimulationData.cuh>
#include <EngineKernels/SimulationKernels.cuh>
#include <EngineKernels/SimulationStatistics.cuh>
#include <EngineKernels/StatisticsKernels.cuh>
#include <EngineKernels/TOProvider.cuh>
#include <EngineKernels/TOs.cuh>

#include "DataAccessKernelsService.cuh"
#include "DomainSyncService.cuh"
#include "EditKernelsService.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "GeometryKernelsService.cuh"
#include "KernelLaunchSettingsService.cuh"
#include "SelectionKernelsService.cuh"
#include "SimulationKernelsService.cuh"
#include "SimulationParametersUpdateService.cuh"
#include "StatisticsKernelsService.cuh"
#include "StatisticsService.cuh"
#include "TestKernelsService.cuh"

namespace
{
    auto constexpr EvolutionStatisticsUpdateInterval = 10;
    ArraySizesForGpuEntities const PreviewCapacityGpu{10000, 10000, 10000000};
    ArraySizesForTOs const PreviewCapacityTO{1000, 1000, 1000, 10000, 10000, 10000, 10000000};

    // Width of the border around the strip of a domain whose objects are mirrored from the neighbor domains
    auto constexpr DomainHaloWidth = 32.0f;
}

_SimulationCudaFacade::_SimulationCudaFacade(uint64_t timestep, SettingsForSimulation const& settings)
{
    initCuda();
    CudaMemoryManager::getInstance().reset();

    initSettingsPreviewData();

    _settings = settings;
    setSimulationParameters(settings.simulationParameters);
    setKernelLaunchSettings(KernelLaunchSettingsService::get().deriveFromDevice(_gpuInfo.deviceNumber));

    log(Priority::Important, "initialize simulation");

    _cudaPreviewData = std::make_shared<SimulationData>();
    _cudaGeometryBuffers = std::make_shared<CudaGeometryBuffers>();
    _cudaSelectionResult = std::make_shared<SelectionResult>();
    _collectionTOProvider = std::make_shared<_TOProvider>();
    _cudaTOProvider = std::make_shared<_CudaTOProvider>();
    _cudaPreviewStatistics = std::make_shared<SimulationStatistics>();

    _simulationTimestep = timestep;
    initDomains();
    _cudaPreviewData->init({_settingsForPreview.worldSizeX, _settingsForPreview.worldSizeY}, 0);
    _cudaPreviewStatistics->init();
    _cudaSelectionResult->init();

    GarbageCollectorKernelsService::get().init();
    SelectionKernelsService::get().init();
    StatisticsKernelsService::get().init();
    TestKernelsService::get().init();
    GeometryKernelsService::get().init();
    EditKernelsService::get().init();
    DataAccessKernelsService::get().init();
    SimulationKernelsService::get().init();

    // Default array sizes for empty simulation (will be resized later if not sufficient)
    for (auto const& domain : _domains) {
        domain.data->resizeObjectsAndTempObjects({100000, 100000, 10000000});
    }
    _cudaPreviewData->resizeObjectsAndTempObjects(PreviewCapacityGpu);

    auto memory = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();
    log(Priority::Important, std::to_string(memory / (1024 * 1024)) + " MB GPU memory used");
}

_SimulationCudaFacade::~_SimulationCudaFacade() noexcept
{
    auto cudaContextWasInvalid = CudaContextState::get().isInvalid();
    if (cudaContextWasInvalid) {
        log(Priority::Unimportant, "skip CUDA shutdown because the CUDA context is invalid");
    } else {
        try {
            DomainSyncService::get().shutdown(_domains);
            for (auto const& domain : _domains) {
                activateDevice(domain);
                domain.data->free();
                domain.statistics->free();
            }
            activateDevice(getMainDomain());
            _cudaPreviewData->free();
            _cudaPreviewStatistics->free();
            _cudaSelectionResult->free();

            SimulationKernelsService::get().shutdown();
            DataAccessKernelsService::get().shutdown();
            EditKernelsService::get().shutdown();
            GeometryKernelsService::get().shutdown();
            TestKernelsService::get().shutdown();
            StatisticsKernelsService::get().shutdown();
            SelectionKernelsService::get().shutdown();
            GarbageCollectorKernelsService::get().shutdown();

            auto const resetResult = cudaDeviceReset();
            if (resetResult != cudaSuccess) {
                log(Priority::Important, std::string("skip CUDA device reset cleanup: ") + cudaGetErrorString(resetResult));
            }
        } catch (std::exception const& e) {
            log(Priority::Important, "skip CUDA shutdown: " + std::string(e.what()));
        } catch (...) {
            log(Priority::Important, "skip CUDA shutdown");
        }
    }
    CudaMemoryManager::getInstance().reset();
    log(Priority::Important, "simulation closed");
}

void _SimulationCudaFacade::copyBuffersFromCudaToOpenGL(GeometryBuffers const& geometryBuffers, RealRect const& visibleWorldRect)
{
    checkAndProcessSimulationParameterChanges();

    KernelProfiler::CategoryScope profilerScope(KernelCategory::Rendering);
    auto simulationData = getSimulationDataPtrCopy();

    GeometryKernelsService::get().correctPositionsForRendering(_settings, simulationData, visibleWorldRect);
    auto numRenderObjects = GeometryKernelsService::get().getNumRenderObjects(_settings, simulationData, visibleWorldRect);
    geometryBuffers->updateNumObjects(numRenderObjects);

    if (GlobalSettings::get().isInterop() && GeometryKernelsService::get().checkForInterop()) {
        _cudaGeometryBuffers->registerBuffers(geometryBuffers);
        GeometryKernelsService::get().extractObjectData(_settings, simulationData, *_cudaGeometryBuffers, visibleWorldRect, true);
        syncAndCheck();
    } else {
        _cudaGeometryBuffers->allocateBuffersForNoInterop(numRenderObjects);
        GeometryKernelsService::get().extractObjectData(_settings, simulationData, *_cudaGeometryBuffers, visibleWorldRect, false);
        syncAndCheck();
        _cudaGeometryBuffers->copyToOpenGL(geometryBuffers, numRenderObjects);
    }

    GeometryKernelsService::get().restorePositions(_settings, simulationData);
    syncAndCheck();
}

void _SimulationCudaFacade::calcTimesteps(uint64_t timesteps, bool forceUpdateStatistics)
{
    calcTimestepsInternal(timesteps, forceUpdateStatistics, false);
}

void _SimulationCudaFacade::syncDomains()
{
    if (!isDecomposed()) {
        return;
    }
    DomainSyncService::get().sync(_domains, _settings.kernelLaunchSettings, [this](Domain& domain, ArraySizesForGpuEntities const& sizeDelta) {
        resizeArraysIfNecessary(domain, sizeDelta);
    });
    activateDevice(getMainDomain());
}

void _SimulationCudaFacade::applyCataclysm(int power)
{
    for (int i = 0; i < power; ++i) {
        EditKernelsService::get().applyCataclysm(_settings.kernelLaunchSettings, getSimulationDataPtrCopy());
        syncAndCheck();
        resizeArraysIfNecessary();
    }
}

std::vector<TOs> _SimulationCudaFacade::getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight)
{
    std::vector<TOs> result;
    for (auto const& domain : _domains) {
        activateDevice(domain);
        auto simulationData = getSimulationDataPtrCopy(domain);
        auto capacities = DataAccessKernelsService::get().estimateCapacityNeededForTO(_settings.kernelLaunchSettings, simulationData);
        auto cudaTO = _cudaTOProvider->provideDataTO(capacities);
        DataAccessKernelsService::get().getData(_settings.kernelLaunchSettings, simulationData, rectUpperLeft, rectLowerRight, cudaTO);
        syncAndCheck();

        auto to = _collectionTOProvider->provideNewUnmanagedDataTO(cudaTO.capacities);
        copyDataTOtoHost(to, cudaTO);
        result.emplace_back(to);
    }
    activateDevice(getMainDomain());
    return result;
}

TOs _SimulationCudaFacade::getSelectedSimulationData(bool includeClusters)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    DataAccessKernelsService::get().getSelectedData(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), includeClusters, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

TOs _SimulationCudaFacade::getInspectedSimulationData(std::vector<uint64_t> entityIds)
{
    InspectedEntityIds ids;
    if (entityIds.size() > Const::MaxInspectedObjects) {
        return TOs{};
    }
    for (int i = 0; i < entityIds.size(); ++i) {
        ids.values[i] = entityIds.at(i);
    }
    if (entityIds.size() < Const::MaxInspectedObjects) {
        ids.values[entityIds.size()] = Const::MaxInspectedObjects_Break;
    }

    auto cudaTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    DataAccessKernelsService::get().getInspectedData(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), ids, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

TOs _SimulationCudaFacade::getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    DataAccessKernelsService::get().getOverlayData(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), rectUpperLeft, rectLowerRight, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

void _SimulationCudaFacade::addAndSelectSimulationData(TOs const& to)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(to.capacities);
    copyDataTOtoGpu(cudaTO, to);

    auto sizeDelta = DataAccessKernelsService::get().estimateCapacityNeededForGpu(_settings.kernelLaunchSettings, cudaTO);
    resizeArraysIfNecessary(sizeDelta);

    SelectionKernelsService::get().removeSelection(_settings.kernelLaunchSettings, getSimulationDataPtrCopy());
    DataAccessKernelsService::get().addData(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), cudaTO, true);
    syncAndCheck();
    updateStatistics();
}

// Every domain receives the whole world and keeps its own objects and the ghosts it needs
void _SimulationCudaFacade::setSimulationData(TOs const& to)
{
    for (auto& domain : _domains) {
        activateDevice(domain);
        auto cudaTO = _cudaTOProvider->provideDataTO(to.capacities);
        copyDataTOtoGpu(cudaTO, to);

        auto sizeDelta = DataAccessKernelsService::get().estimateCapacityNeededForGpu(_settings.kernelLaunchSettings, cudaTO);
        resizeArraysIfNecessary(domain, sizeDelta);

        auto simulationData = getSimulationDataPtrCopy(domain);
        DataAccessKernelsService::get().clearData(_settings.kernelLaunchSettings, simulationData);
        DataAccessKernelsService::get().addData(_settings.kernelLaunchSettings, simulationData, cudaTO, false);
        syncAndCheck();
    }
    if (isDecomposed()) {
        DomainSyncService::get().distribute(_domains, _settings.kernelLaunchSettings);
    }
    activateDevice(getMainDomain());

    _accumulatedLineageValues.clear();
    updateStatistics();
}

void _SimulationCudaFacade::removeSelectedObjects(bool includeClusters)
{
    EditKernelsService::get().removeSelectedObjects(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::relaxSelectedObjects(bool includeClusters)
{
    EditKernelsService::get().relaxSelectedObjects(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

Ids _SimulationCudaFacade::getMaxIds() const
{
    Ids result;
    for (auto const& domain : _domains) {
        activateDevice(domain);
        auto ids = domain.data->primaryNumberGen.getIds_host();
        result.entityId = std::max(result.entityId, ids.entityId);
        result.lineageId = std::max(result.lineageId, ids.lineageId);
    }
    activateDevice(getMainDomain());
    return result;
}

void _SimulationCudaFacade::uniformVelocitiesForSelectedObjects(bool includeClusters)
{
    EditKernelsService::get().uniformVelocities(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::makeSticky(bool includeClusters)
{
    EditKernelsService::get().makeSticky(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::removeStickiness(bool includeClusters)
{
    EditKernelsService::get().removeStickiness(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::setStatic(bool value, bool includeClusters)
{
    EditKernelsService::get().setStatic(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), value, includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::changeInspectedSimulationData(TOs const& changeTO)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(changeTO.capacities);
    copyDataTOtoGpu(cudaTO, changeTO);

    EditKernelsService::get().changeSimulationData(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), cudaTO);
    syncAndCheck();

    updateStatistics();

    resizeArraysIfNecessary();
}

int _SimulationCudaFacade::injectGenomeToSelectedCreatures(TOs const& to)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(to.capacities);
    copyDataTOtoGpu(cudaTO, to);

    auto result = EditKernelsService::get().injectGenomeToSelectedCreatures(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), cudaTO);
    syncAndCheck();

    updateStatistics();

    resizeArraysIfNecessary();

    return result;
}

void _SimulationCudaFacade::applyForce(ApplyForceData const& applyData)
{
    EditKernelsService::get().applyForce(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), applyData);
    syncAndCheck();
}

void _SimulationCudaFacade::switchSelection(PointSelectionData const& pointData)
{
    SelectionKernelsService::get().switchSelection(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::swapSelection(PointSelectionData const& pointData)
{
    SelectionKernelsService::get().swapSelection(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::setSelection(AreaSelectionData const& selectionData)
{
    SelectionKernelsService::get().setSelection(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), selectionData);
    syncAndCheck();
}

SelectionShallowData _SimulationCudaFacade::getSelectionShallowData()
{
    EditKernelsService::get().getSelectionShallowData(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), *_cudaSelectionResult);
    syncAndCheck();
    return _cudaSelectionResult->getSelectionShallowData();
}

void _SimulationCudaFacade::shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& shallowUpdateData)
{
    EditKernelsService::get().shallowUpdateSelectedObjects(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), shallowUpdateData);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::removeSelection()
{
    SelectionKernelsService::get().removeSelection(_settings.kernelLaunchSettings, getSimulationDataPtrCopy());
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::updateSelection()
{
    SelectionKernelsService::get().updateSelection(_settings.kernelLaunchSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::colorSelectedObjects(unsigned char color, bool includeClusters)
{
    EditKernelsService::get().colorSelectedCells(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), color, includeClusters);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::reconnectSelectedObjects()
{
    EditKernelsService::get().reconnect(_settings.kernelLaunchSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::glueSelectedObjects(bool includeClusters)
{
    EditKernelsService::get().glueSelectedObjects(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::cutConnections(float2 const& start, float2 const& end, bool onlySelected, bool includeClusters)
{
    EditKernelsService::get().cutConnections(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), start, end, onlySelected, includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::setDetached(bool value)
{
    EditKernelsService::get().setDetached(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), value);
    syncAndCheck();
}

void _SimulationCudaFacade::setKernelLaunchSettings(KernelLaunchSettings const& launchSettings)
{
    _settings.kernelLaunchSettings = launchSettings;
}

SimulationParameters _SimulationCudaFacade::getSimulationParameters() const
{
    std::lock_guard lock(_mutexForSimulationParameters);
    return _newSimulationParameters ? *_newSimulationParameters : _settings.simulationParameters;
}

void _SimulationCudaFacade::setSimulationParameters(SimulationParameters const& parameters, SimulationParametersUpdateConfig const& updateConfig)
{
    std::lock_guard lock(_mutexForSimulationParameters);
    _newSimulationParameters = parameters;
    _simulationParametersUpdateConfig = updateConfig;
}

ArraySizesForTOs _SimulationCudaFacade::estimateCapacityNeededForTO() const
{
    return DataAccessKernelsService::get().estimateCapacityNeededForTO(_settings.kernelLaunchSettings, getSimulationDataPtrCopy());
}

namespace
{
    // The domains count their own objects, so their statistics add up
    StatisticsEntry mergeStatisticsEntries(std::vector<StatisticsEntry> const& entries)
    {
        if (entries.size() == 1) {
            return entries.front();
        }
        StatisticsEntry result;
        std::map<uint32_t, LineageStatisticsEntry> lineageEntryById;
        for (auto const& entry : entries) {
            auto& objectStatistics = result.objectStatistics;
            objectStatistics.numSolidObjects += entry.objectStatistics.numSolidObjects;
            objectStatistics.numFluidObjects += entry.objectStatistics.numFluidObjects;
            objectStatistics.numFreeCellObjects += entry.objectStatistics.numFreeCellObjects;
            objectStatistics.numCellObjects += entry.objectStatistics.numCellObjects;
            objectStatistics.numEnergyParticles += entry.objectStatistics.numEnergyParticles;
            objectStatistics.totalInternalEnergy += entry.objectStatistics.totalInternalEnergy;

            for (auto const& lineageEntry : entry.lineageEntries) {
                auto [iter, inserted] = lineageEntryById.try_emplace(lineageEntry.lineageId, lineageEntry);
                if (inserted) {
                    continue;
                }
                auto& merged = iter->second;
                merged.colorBitset |= lineageEntry.colorBitset;
                merged.numCreatures += lineageEntry.numCreatures;
                merged.numGenomes += lineageEntry.numGenomes;
                merged.sumCreatureCells += lineageEntry.sumCreatureCells;
                merged.sumCreatureGenerations += lineageEntry.sumCreatureGenerations;
                merged.sumGenomeNodes += lineageEntry.sumGenomeNodes;
                merged.sumMutationRates += lineageEntry.sumMutationRates;
                merged.sumCreatureEnergy += lineageEntry.sumCreatureEnergy;
                if (merged.representativeCellId == 0) {
                    merged.representativeCellId = lineageEntry.representativeCellId;
                }
                merged.numCreatedCreatures += lineageEntry.numCreatedCreatures;
                merged.totalMutations += lineageEntry.totalMutations;
                merged.totalAttackedEnergy += lineageEntry.totalAttackedEnergy;
                merged.totalMuscleActivity += lineageEntry.totalMuscleActivity;
            }
        }
        for (auto const& lineageEntry : lineageEntryById | std::views::values) {
            result.lineageEntries.emplace_back(lineageEntry);
        }
        return result;
    }
}

void _SimulationCudaFacade::updateStatistics()
{
    std::vector<StatisticsEntry> entries;
    for (auto const& domain : _domains) {
        activateDevice(domain);
        StatisticsKernelsService::get().updateStatistics(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(domain), *domain.statistics);
        syncAndCheck();
        entries.emplace_back(domain.statistics->getStatisticsEntry());
        if (isDecomposed()) {
            for (auto const& drainedEntry : domain.statistics->getDrainedAccumulatorEntries()) {
                auto& values = _accumulatedLineageValues[drainedEntry.lineageId];
                values.numCreatedCreatures += drainedEntry.numCreatedCreatures;
                values.totalMutations += drainedEntry.totalMutations;
                values.totalAttackedEnergy += drainedEntry.totalAttackedEnergy;
                values.totalMuscleActivity += drainedEntry.totalMuscleActivity;
            }
        }
    }
    activateDevice(getMainDomain());

    auto statisticsEntry = mergeStatisticsEntries(entries);
    if (isDecomposed()) {
        applyAccumulatedLineageValues(statisticsEntry);
    }
    {
        std::lock_guard lock(_mutexForStatistics);
        _statisticsEntry = statisticsEntry;
    }
    StatisticsService::get().addDataPoint(_statisticsHistory, statisticsEntry, getCurrentTimestep());
}

// The accumulated values of extinct lineages are discarded, the statistics history keeps them
void _SimulationCudaFacade::applyAccumulatedLineageValues(StatisticsEntry& statisticsEntry)
{
    std::unordered_map<uint32_t, AccumulatedLineageValues> livingLineageValues;
    for (auto& lineageEntry : statisticsEntry.lineageEntries) {
        auto values = _accumulatedLineageValues[lineageEntry.lineageId];
        lineageEntry.numCreatedCreatures = values.numCreatedCreatures;
        lineageEntry.totalMutations = values.totalMutations;
        lineageEntry.totalAttackedEnergy = values.totalAttackedEnergy;
        lineageEntry.totalMuscleActivity = values.totalMuscleActivity;
        livingLineageValues.emplace(lineageEntry.lineageId, values);
    }
    _accumulatedLineageValues = std::move(livingLineageValues);
}

StatisticsHistory const& _SimulationCudaFacade::getStatisticsHistory() const
{
    return _statisticsHistory;
}

StatisticsEntry _SimulationCudaFacade::getStatisticsEntry()
{
    std::lock_guard lock(_mutexForStatistics);
    if (_statisticsEntry) {
        return *_statisticsEntry;
    } else {
        return StatisticsEntry();
    }
}

void _SimulationCudaFacade::setStatisticsHistory(StatisticsHistoryData const& data)
{
    StatisticsService::get().setStatisticsHistory(_statisticsHistory, data, getCurrentTimestep());
}

uint64_t _SimulationCudaFacade::getCurrentTimestep() const
{
    std::lock_guard lock(_mutexForSimulationData);
    return _simulationTimestep;
}

void _SimulationCudaFacade::setCurrentTimestep(uint64_t timestep)
{
    {
        std::lock_guard lock(_mutexForSimulationData);
        for (auto const& domain : _domains) {
            activateDevice(domain);
            copyToDevice(domain.data->timestep, &timestep);  // Update GPU timestep
        }
        activateDevice(getMainDomain());
        _simulationTimestep = timestep;
    }
    StatisticsService::get().resetTime(_statisticsHistory, timestep);
}

void _SimulationCudaFacade::clear()
{
    for (auto const& domain : _domains) {
        activateDevice(domain);
        DataAccessKernelsService::get().clearData(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(domain));
        syncAndCheck();
    }
    activateDevice(getMainDomain());
    _accumulatedLineageValues.clear();
}

void _SimulationCudaFacade::resizeArraysIfNecessary(ArraySizesForGpuEntities const& sizeDelta)
{
    for (auto& domain : _domains) {
        activateDevice(domain);
        resizeArraysIfNecessary(domain, sizeDelta);
    }
    activateDevice(getMainDomain());
}

void _SimulationCudaFacade::resizeArraysIfNecessary(Domain& domain, ArraySizesForGpuEntities const& sizeDelta)
{
    if (domain.data->shouldResize(sizeDelta)) {
        resizeArrays(domain, sizeDelta);
    }
}

void _SimulationCudaFacade::initSettingsPreviewData()
{
    _settingsForPreview.simulationParameters.friction.baseValue = 0.01f;
    _settingsForPreview.simulationParameters.maxVelocity.value = 0.02f;
    for (int i = 0; i < MAX_COLORS; ++i) {
        _settingsForPreview.simulationParameters.radiationType1_strength.baseValue[i] = 0.0f;
        _settingsForPreview.simulationParameters.radiationType2_strength.value[i] = 0.0f;
    }
    _settingsForPreview.worldSizeX = PREVIEW_WIDTH;
    _settingsForPreview.worldSizeY = PREVIEW_HEIGHT;
    _settingsForPreview.kernelLaunchSettings.numBlocks = 16;
}

void _SimulationCudaFacade::newPreview(TOs const& to)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(to.capacities);
    copyDataTOtoGpu(cudaTO, to);

    DataAccessKernelsService::get().clearData(_settings.kernelLaunchSettings, *_cudaPreviewData);
    DataAccessKernelsService::get().addData(_settings.kernelLaunchSettings, *_cudaPreviewData, cudaTO, false);
    syncAndCheck();
}

void _SimulationCudaFacade::calcTimestepsForPreview(std::chrono::milliseconds const& duration, bool detailSimulation)
{
    CHECK_FOR_DEVICE_ERRORS(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settingsForPreview.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));

    auto startTimepoint = std::chrono::steady_clock::now();
    do {

        SimulationKernelsService::get().calcTimestepForPreview(
            _settingsForPreview, *_cudaPreviewData, *_cudaPreviewStatistics, _previewTimestep, false, detailSimulation);
        syncAndCheck();

        ++_previewTimestep;  // SimulationData::timestep is already updated in the kernels
    } while (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTimepoint) < duration);

    CHECK_FOR_DEVICE_ERRORS(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void _SimulationCudaFacade::calcTimestepsForPreview(int numSteps, bool detailSimulation)
{
    CHECK_FOR_DEVICE_ERRORS(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settingsForPreview.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));

    for (int i = 0; i < numSteps; ++i) {
        SimulationKernelsService::get().calcTimestepForPreview(
            _settingsForPreview, *_cudaPreviewData, *_cudaPreviewStatistics, _previewTimestep, false, detailSimulation);
        syncAndCheck();

        ++_previewTimestep;  // SimulationData::timestep is already updated in the kernels
    }

    CHECK_FOR_DEVICE_ERRORS(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

uint64_t _SimulationCudaFacade::getCurrentTimestepForPreview()
{
    return _previewTimestep;
}

void _SimulationCudaFacade::setCurrentTimestepForPreview(uint64_t timestep)
{
    _previewTimestep = timestep;
    copyToDevice(_cudaPreviewData->timestep, &timestep);  // Update GPU timestep
}

TOs _SimulationCudaFacade::getPreviewData()
{
    auto cudaTO = _cudaTOProvider->provideDataTO(PreviewCapacityTO);
    DataAccessKernelsService::get().getData(
        _settings.kernelLaunchSettings, *_cudaPreviewData, {-10, -10}, {_settingsForPreview.worldSizeX + 10, _settingsForPreview.worldSizeY + 10}, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideNewUnmanagedDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

void _SimulationCudaFacade::testOnly_mutate(uint64_t objectId)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_mutate(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), *getMainDomain().statistics, objectId);
    syncAndCheck();

    resizeArraysIfNecessary();
}

void _SimulationCudaFacade::testOnly_voidUnreachableNodes(uint64_t objectId)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_voidUnreachableNodes(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), objectId);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_removeUnusedGenes(uint64_t objectId)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_removeUnusedGenes(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), objectId);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_removeGeneCycles(uint64_t objectId)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_removeGeneCycles(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), objectId);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_limitGenesWithSeparation(uint64_t objectId)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_limitGenesWithSeparation(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), objectId);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_createConnection(uint64_t objectId1, uint64_t objectId2)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_createConnection(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(), objectId1, objectId2);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_createConnectionWithAbsAngle(
    uint64_t objectId1,
    uint64_t objectId2,
    float desiredDistance,
    float desiredAbsAngle1,
    float desiredAbsAngle2)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_createConnectionWithAbsAngle(
        _settings.kernelLaunchSettings, getSimulationDataPtrCopy(), objectId1, objectId2, desiredDistance, desiredAbsAngle1, desiredAbsAngle2);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_cleanupAfterTimestep()
{
    checkAndProcessSimulationParameterChanges();
    GarbageCollectorKernelsService::get().cleanupAfterTimestep(_settings.kernelLaunchSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_cleanupAfterDataManipulation()
{
    checkAndProcessSimulationParameterChanges();
    GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(_settings.kernelLaunchSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_resizeArrays(ArraySizesForGpuEntities const& sizeDelta)
{
    checkAndProcessSimulationParameterChanges();
    resizeArrays(sizeDelta);
    syncAndCheck();
}

bool _SimulationCudaFacade::testOnly_isDataValid()
{
    checkAndProcessSimulationParameterChanges();
    auto result = TestKernelsService::get().testOnly_isDataValid(_settings.kernelLaunchSettings, getSimulationDataPtrCopy());
    syncAndCheck();
    return result;
}

void _SimulationCudaFacade::testOnly_calcTimestepWithCellTypeFunctions()
{
    calcTimestepsInternal(1, true, true);

    CHECK(testOnly_isDataValid());
}

void _SimulationCudaFacade::testOnly_calcTimestepWithCellTypeFunctionsForPreview(bool detailSimulation)
{
    CHECK_FOR_DEVICE_ERRORS(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settingsForPreview.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));

    SimulationKernelsService::get().calcTimestepForPreview(
        _settingsForPreview, *_cudaPreviewData, *_cudaPreviewStatistics, _previewTimestep, true, detailSimulation);
    syncAndCheck();

    ++_previewTimestep;  // SimulationData::timestep is already updated in the kernels

    CHECK_FOR_DEVICE_ERRORS(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void _SimulationCudaFacade::initCuda()
{
    log(Priority::Important, "initialize CUDA");
    _gpuInfo = checkAndReturnGpuInfo();

    auto result = cudaSetDevice(_gpuInfo.deviceNumber);
    if (result != cudaSuccess) {
        throw std::runtime_error("CUDA device could not be initialized.");
    }

    cudaGetLastError();  // Reset error code
    CudaContextState::get().reset();

    log(Priority::Important, "device " + std::to_string(_gpuInfo.deviceNumber) + " selected");
}

auto _SimulationCudaFacade::checkAndReturnGpuInfo() -> GpuInfo
{
    static std::optional<GpuInfo> cachedResult;
    if (cachedResult) {
        return *cachedResult;
    }
    cachedResult = GpuInfo();

    int numberOfDevices;
    CHECK_FOR_DEVICE_ERRORS(cudaGetDeviceCount(&numberOfDevices));
    if (numberOfDevices < 1) {
        throw std::runtime_error("No CUDA device found.");
    }
    {
        std::stringstream stream;
        if (1 == numberOfDevices) {
            stream << "1 CUDA device found";
        } else {
            stream << numberOfDevices << " CUDA devices found";
        }
        log(Priority::Important, stream.str());
    }

    int highestComputeCapability = 0;
    for (int deviceNumber = 0; deviceNumber < numberOfDevices; ++deviceNumber) {
        cudaDeviceProp prop;
        CHECK_FOR_DEVICE_ERRORS(cudaGetDeviceProperties(&prop, deviceNumber));

        std::stringstream stream;
        stream << "device " << deviceNumber << ": " << prop.name << " with compute capability " << prop.major << "." << prop.minor;
        log(Priority::Important, stream.str());

        int computeCapability = prop.major * 100 + prop.minor;
        if (computeCapability > highestComputeCapability) {
            cachedResult->deviceNumber = deviceNumber;
            highestComputeCapability = computeCapability;
            cachedResult->gpuModelName = prop.name;
        }
    }
#if !defined(USE_HIP)
    // The CUDA compute-capability heuristic (major*100+minor, gated at >= 705)
    // is meaningless for AMD GPUs, where prop.major/minor encode the gfx arch.
    // On HIP accept the selected device unconditionally.
    if (highestComputeCapability < 705) {
        throw std::runtime_error("No CUDA device with compute capability of 7.5 or higher found.");
    }
#endif

    return *cachedResult;
}

void _SimulationCudaFacade::syncAndCheck()
{
    cudaDeviceSynchronize();
    CHECK_FOR_DEVICE_ERRORS(cudaGetLastError());
}

void _SimulationCudaFacade::copyDataTOtoGpu(TOs const& cudaTO, TOs const& to)
{
    copyToDevice(cudaTO.numObjects, to.numObjects);
    copyToDevice(cudaTO.numEnergyParticles, to.numEnergyParticles);
    copyToDevice(cudaTO.numGenomes, to.numGenomes);
    copyToDevice(cudaTO.numCreatures, to.numCreatures);
    copyToDevice(cudaTO.numGenes, to.numGenes);
    copyToDevice(cudaTO.numNodes, to.numNodes);
    copyToDevice(cudaTO.heapSize, to.heapSize);

    copyToDevice(cudaTO.objects, to.objects, *to.numObjects);
    copyToDevice(cudaTO.energyParticles, to.energyParticles, *to.numEnergyParticles);
    copyToDevice(cudaTO.genomes, to.genomes, *to.numGenomes);
    copyToDevice(cudaTO.creatures, to.creatures, *to.numCreatures);
    copyToDevice(cudaTO.genes, to.genes, *to.numGenes);
    copyToDevice(cudaTO.nodes, to.nodes, *to.numNodes);
    copyToDevice(cudaTO.heap, to.heap, *to.heapSize);
}

void _SimulationCudaFacade::copyDataTOtoHost(TOs const& to, TOs const& cudaTO)
{
    copyToHost(to.numObjects, cudaTO.numObjects);
    copyToHost(to.numEnergyParticles, cudaTO.numEnergyParticles);
    copyToHost(to.numGenomes, cudaTO.numGenomes);
    copyToHost(to.numCreatures, cudaTO.numCreatures);
    copyToHost(to.numGenes, cudaTO.numGenes);
    copyToHost(to.numNodes, cudaTO.numNodes);
    copyToHost(to.heapSize, cudaTO.heapSize);

    copyToHost(to.objects, cudaTO.objects, *to.numObjects);
    copyToHost(to.energyParticles, cudaTO.energyParticles, *to.numEnergyParticles);
    copyToHost(to.genomes, cudaTO.genomes, *to.numGenomes);
    copyToHost(to.creatures, cudaTO.creatures, *to.numCreatures);
    copyToHost(to.genes, cudaTO.genes, *to.numGenes);
    copyToHost(to.nodes, cudaTO.nodes, *to.numNodes);
    copyToHost(to.heap, cudaTO.heap, *to.heapSize);
}

void _SimulationCudaFacade::testOnly_zeroTransferData()
{
    auto cudaTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    CHECK_FOR_DEVICE_ERRORS(cudaMemset(cudaTO.objects, 0, sizeof(ObjectTO) * cudaTO.capacities.objects));
    CHECK_FOR_DEVICE_ERRORS(cudaMemset(cudaTO.energyParticles, 0, sizeof(EnergyTO) * cudaTO.capacities.energyParticles));
    CHECK_FOR_DEVICE_ERRORS(cudaMemset(cudaTO.creatures, 0, sizeof(CreatureTO) * cudaTO.capacities.creatures));
    CHECK_FOR_DEVICE_ERRORS(cudaMemset(cudaTO.genomes, 0, sizeof(GenomeTO) * cudaTO.capacities.genomes));
    CHECK_FOR_DEVICE_ERRORS(cudaMemset(cudaTO.genes, 0, sizeof(GeneTO) * cudaTO.capacities.genes));
    CHECK_FOR_DEVICE_ERRORS(cudaMemset(cudaTO.nodes, 0, sizeof(NodeTO) * cudaTO.capacities.nodes));
    CHECK_FOR_DEVICE_ERRORS(cudaMemset(cudaTO.heap, 0, sizeof(uint8_t) * cudaTO.capacities.heap));
}

void _SimulationCudaFacade::calcTimestepsInternal(uint64_t timesteps, bool forceUpdateStatistics, bool forceCellFunctionExecution)
{
    KernelProfiler::CategoryScope profilerScope(KernelCategory::Simulation);

    static int counter = 0;

    for (uint64_t i = 0; i < timesteps; ++i) {
        checkAndProcessSimulationParameterChanges();

        syncDomains();

        auto timestep = getCurrentTimestep();
        reportProfilingContext();
        for (auto const& domain : _domains) {
            activateDevice(domain);
            SimulationKernelsService::get().launchTimestep(_settings, *domain.data, *domain.statistics, timestep, forceCellFunctionExecution);
        }
        for (auto const& domain : _domains) {
            activateDevice(domain);
            SimulationKernelsService::get().finishTimestep(_settings, *domain.data);
        }
        {
            std::lock_guard lock(_mutexForSimulationData);
            ++_simulationTimestep;  // SimulationData::timestep is already updated in the kernels
        }
        for (auto const& domain : _domains) {
            activateDevice(domain);
            syncAndCheck();
        }
        activateDevice(getMainDomain());

        // Make check after every 10th call
        if (++counter % 10 == 0) {
            counter = 0;
            resizeArraysIfNecessary();
        }

        {
            std::lock_guard lock(_mutexForSimulationParameters);
            auto readExternalEnergy = [this] {
                auto result = 0.0;
                for (auto const& domain : _domains) {
                    activateDevice(domain);
                    result += copyToHost(domain.data->externalEnergy);
                }
                activateDevice(getMainDomain());
                return result;
            };
            if (SimulationParametersUpdateService::get().updateSimulationParametersAfterTimestep(_settings, readExternalEnergy, getCurrentTimestep())) {
                copySimulationParametersToDevices(_settings.simulationParameters);
            }
            if (isDecomposed()) {
                auto isCellFunctionStep = [&](uint64_t value) { return forceCellFunctionExecution || value % TIMESTEPS_PER_CELL_FUNCTION == 0; };
                DomainSyncService::get().updateExternalEnergyShares(
                    _domains, _settings.simulationParameters.externalEnergy.value, isCellFunctionStep(timestep), isCellFunctionStep(timestep + 1));
                activateDevice(getMainDomain());
            }
        }
        if (getCurrentTimestep() % EvolutionStatisticsUpdateInterval == 0) {
            updateStatistics();
        }
    }
    if (forceUpdateStatistics) {
        updateStatistics();
    }
}

void _SimulationCudaFacade::resizeArrays(ArraySizesForGpuEntities const& sizeDelta)
{
    for (auto& domain : _domains) {
        activateDevice(domain);
        resizeArrays(domain, sizeDelta);
    }
    activateDevice(getMainDomain());
}

void _SimulationCudaFacade::resizeArrays(Domain& domain, ArraySizesForGpuEntities const& sizeDelta)
{
    log(Priority::Important, "resize arrays");

    auto const& simulationData = domain.data;
    simulationData->resizeTempObjects(sizeDelta);

    if (!simulationData->isEmpty()) {
        GarbageCollectorKernelsService::get().copyArrays(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(domain));
        syncAndCheck();

        simulationData->resizeObjectsByMatchingTempObjects();

        GarbageCollectorKernelsService::get().swapArrays(_settings.kernelLaunchSettings, getSimulationDataPtrCopy(domain));
        syncAndCheck();
    } else {
        simulationData->resizeObjectsByMatchingTempObjects();
    }

    auto cellArraySize = simulationData->entities.objects.getCapacity_host();
    auto particleArraySize = simulationData->entities.energies.getCapacity_host();
    auto auxiliaryDataSize = simulationData->entities.heap.getCapacity_host();

    CHECK_FOR_DEVICE_ERRORS(cudaGetLastError());

    log(Priority::Unimportant, "cell array capacity: " + StringHelper::format(cellArraySize));
    log(Priority::Unimportant, "particle array capacity: " + StringHelper::format(particleArraySize));
    log(Priority::Unimportant, "heap capacity: " + StringHelper::format(auxiliaryDataSize));

    auto const memorySizeAfter = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();
    log(Priority::Important, std::to_string(memorySizeAfter / (1024 * 1024)) + " MB GPU memory used");
}

void _SimulationCudaFacade::reportProfilingContext()
{
    if (!KernelProfiler::get().isEnabled()) {
        return;
    }

    auto& profiler = KernelProfiler::get();
    profiler.setReportEntry("gpu", _gpuInfo.gpuModelName);

    cudaDeviceProp prop;
    auto const hasDeviceProperties = cudaGetDeviceProperties(&prop, _gpuInfo.deviceNumber) == cudaSuccess;
    if (hasDeviceProperties) {
        profiler.setReportEntry(
            "compute capability / SMs", std::to_string(prop.major) + "." + std::to_string(prop.minor) + " / " + std::to_string(prop.multiProcessorCount));
        profiler.setReportEntry("total GPU memory [MB]", std::to_string(prop.totalGlobalMem / (1024 * 1024)));
    }
    size_t freeMemory = 0;
    size_t totalMemory = 0;
    if (cudaMemGetInfo(&freeMemory, &totalMemory) == cudaSuccess) {
        profiler.setReportEntry("free GPU memory [MB]", std::to_string(freeMemory / (1024 * 1024)));
    }
    profiler.setReportEntry("acquired GPU memory [MB]", std::to_string(CudaMemoryManager::getInstance().getSizeOfAcquiredMemory() / (1024 * 1024)));

    profiler.setReportEntry("numBlocks", std::to_string(_settings.kernelLaunchSettings.numBlocks));
    profiler.setReportEntry("world size", std::to_string(_settings.worldSizeX) + " x " + std::to_string(_settings.worldSizeY));
    profiler.setReportEntry("smoothing length", std::to_string(_settings.simulationParameters.smoothingLength.value));

    auto const& entities = getMainDomain().data->entities;
    profiler.setReportEntry(
        "objects (used / capacity)", std::to_string(entities.objects.getNumEntries_host()) + " / " + std::to_string(entities.objects.getCapacity_host()));
    profiler.setReportEntry(
        "energy particles", std::to_string(entities.energies.getNumEntries_host()) + " / " + std::to_string(entities.energies.getCapacity_host()));
    profiler.setReportEntry(
        "heap (used / capacity)", std::to_string(entities.heap.getNumEntries_host()) + " / " + std::to_string(entities.heap.getCapacity_host()));

    // Objects per block is what the block-partitioned kernels actually loop over, so it decides their cost.
    auto const numBlocks = std::max(1, _settings.kernelLaunchSettings.numBlocks);
    profiler.setReportEntry("objects per block", std::to_string(entities.objects.getNumEntries_host() / static_cast<uint64_t>(numBlocks)));

#if !defined(USE_HIP)
    // The occupancy query would need a HIP counterpart; the kernels only misbehave on NVIDIA hardware anyway.
    // Blackwell keeps a single block resident per SM for the fluid kernels even though the budgets below allow far
    // more, which is what made them collapse there. The table shows what each block size buys on the machine at
    // hand and therefore why KernelLaunchSettingsService picked the entry reported as "fluid warps per block".
    if (hasDeviceProperties) {
        std::string residentWarpsByWarpsPerBlock;
        auto warpsPerBlock = 0;
        for (auto const& residentWarps : KernelLaunchSettingsService::get().calcFluidResidentWarps()) {
            ++warpsPerBlock;
            residentWarpsByWarpsPerBlock += std::to_string(warpsPerBlock) + ":" + std::to_string(residentWarps) + " ";
        }
        profiler.setReportEntry("fluid resident warps/SM", residentWarpsByWarpsPerBlock);

        cudaFuncAttributes attributes;
        if (cudaFuncGetAttributes(&attributes, cudaNextTimestep_physics_calcFluidForces) == cudaSuccess) {
            profiler.setReportEntry("fluid kernel registers", std::to_string(attributes.numRegs));
            profiler.setReportEntry("fluid kernel shared [B]", std::to_string(attributes.sharedSizeBytes));
            profiler.setReportEntry("fluid kernel local [B]", std::to_string(attributes.localSizeBytes));
            profiler.setReportEntry("fluid kernel binary / ptx", std::to_string(attributes.binaryVersion) + " / " + std::to_string(attributes.ptxVersion));
            profiler.setReportEntry("fluid warps per block", std::to_string(_settings.kernelLaunchSettings.fluidWarpsPerBlock));
        }

        profiler.setReportEntry("SM budget: threads", std::to_string(prop.maxThreadsPerMultiProcessor));
        profiler.setReportEntry("SM budget: blocks", std::to_string(prop.maxBlocksPerMultiProcessor));
        profiler.setReportEntry("SM budget: registers", std::to_string(prop.regsPerMultiprocessor));
        profiler.setReportEntry("SM budget: shared [B]", std::to_string(prop.sharedMemPerMultiprocessor));
        profiler.setReportEntry("SM budget: reserved shared [B]", std::to_string(prop.reservedSharedMemPerBlock));
    }
#endif
}

void _SimulationCudaFacade::checkAndProcessSimulationParameterChanges()
{
    std::lock_guard lock(_mutexForSimulationParameters);
    if (_newSimulationParameters) {
        _settings.simulationParameters = SimulationParametersUpdateService::get().integrateChanges(
            _settings.simulationParameters, *_newSimulationParameters, _simulationParametersUpdateConfig);
        copySimulationParametersToDevices(_settings.simulationParameters);
        _newSimulationParameters.reset();

        for (auto const& domain : _domains) {
            activateDevice(domain);
            SimulationKernelsService::get().prepareForSimulationParametersChanges(_settings, getSimulationDataPtrCopy(domain));
        }
        activateDevice(getMainDomain());
    }
}

void _SimulationCudaFacade::copySimulationParametersToDevices(SimulationParameters const& parameters)
{
    std::set<int> devices;
    for (auto const& domain : _domains) {
        devices.insert(domain.device);
    }
    devices.insert(_gpuInfo.deviceNumber);
    for (auto const& device : devices) {
        CHECK_FOR_DEVICE_ERRORS(cudaSetDevice(device));
        CHECK_FOR_DEVICE_ERRORS(cudaMemcpyToSymbol(cudaSimulationParameters, &parameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
    }
    if (!_domains.empty()) {
        activateDevice(getMainDomain());
    }
}

// The operations based on this copy only support a simulation that is not split into domains
SimulationData _SimulationCudaFacade::getSimulationDataPtrCopy() const
{
    checkNotDecomposed("This operation");
    std::lock_guard lock(_mutexForSimulationData);
    return *getMainDomain().data;
}

SimulationData _SimulationCudaFacade::getSimulationDataPtrCopy(Domain const& domain) const
{
    std::lock_guard lock(_mutexForSimulationData);
    return *domain.data;
}

Domain& _SimulationCudaFacade::getMainDomain()
{
    return _domains.front();
}

Domain const& _SimulationCudaFacade::getMainDomain() const
{
    return _domains.front();
}

void _SimulationCudaFacade::initDomains()
{
    _domains.clear();

    auto numDomains = GlobalSettings::get().getNumDomains();
    if (numDomains > DomainLayout::MaxDomains) {
        throw std::runtime_error("At most " + std::to_string(DomainLayout::MaxDomains) + " domains are supported.");
    }
    auto devices = GlobalSettings::get().getDomainDevices();
    if (devices.empty()) {
        devices.emplace_back(_gpuInfo.deviceNumber);
    }
    int numDevices = 0;
    CHECK_FOR_DEVICE_ERRORS(cudaGetDeviceCount(&numDevices));
    for (auto const& device : devices) {
        if (device < 0 || device >= numDevices) {
            auto existingDevices = numDevices == 1 ? std::string("only GPU 0 exists") : "the GPUs are numbered from 0 to " + std::to_string(numDevices - 1);
            throw std::runtime_error("GPU " + std::to_string(device) + " does not exist, " + existingDevices + ".");
        }
    }

    for (int index = 0; index < numDomains; ++index) {
        Domain domain;
        domain.index = index;
        domain.device = devices.at(index % devices.size());
        activateDevice(domain);
        domain.data = std::make_shared<SimulationData>();
        domain.statistics = std::make_shared<SimulationStatistics>();
        domain.data->domain = DomainContext{.index = index, .numDomains = numDomains};
        domain.data->init({_settings.worldSizeX, _settings.worldSizeY}, _simulationTimestep);
        domain.statistics->init();
        if (numDomains > 1) {
            domain.data->primaryNumberGen.setIdPartition(numDomains, index);
            domain.statistics->enableDraining();
        }
        _domains.emplace_back(domain);
    }
    if (numDomains > 1) {
        DomainSyncService::get().init(_domains, {_settings.worldSizeX, _settings.worldSizeY}, DomainHaloWidth);
        log(Priority::Important, "world split into " + std::to_string(numDomains) + " domains");
    }
    activateDevice(getMainDomain());
}

bool _SimulationCudaFacade::isDecomposed() const
{
    return _domains.size() > 1;
}

void _SimulationCudaFacade::checkNotDecomposed(std::string const& operation) const
{
    if (isDecomposed()) {
        throw std::runtime_error(operation + " is not supported for a simulation split into several domains.");
    }
}

void _SimulationCudaFacade::activateDevice(Domain const& domain) const
{
    CHECK_FOR_DEVICE_ERRORS(cudaSetDevice(domain.device));
}
