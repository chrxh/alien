#include <functional>
#include <iostream>
#include <list>

#include <cuda/helper_cuda.h>
#include <cuda_runtime.h>

#include <Base/Exceptions.h>
#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>
#include <Base/Macros.h>

#include <EngineInterface/CudaSettings.h>
#include <EngineInterface/Ids.h>
#include <EngineInterface/InspectedEntityIds.h>
#include <EngineInterface/SimulationParameters.h>
#include <EngineInterface/SpaceCalculator.h>

#include "Base.cuh"
#include "ConstantMemory.cuh"
#include "CudaGeometryBuffers.cuh"
#include "CudaMemoryManager.cuh"
#include "CudaTOProvider.cuh"
#include "DataAccessKernels.cuh"
#include "DataAccessKernelsService.cuh"
#include "EditKernels.cuh"
#include "EditKernelsService.cuh"
#include "GarbageCollectorKernels.cuh"
#include "GarbageCollectorKernelsService.cuh"
#include "GeometryKernels.cuh"
#include "GeometryKernelsService.cuh"
#include "Map.cuh"
#include "MaxAgeBalancer.cuh"
#include "Objects.cuh"
#include "SelectionResult.cuh"
#include "SimulationCudaFacade.cuh"

#include "SelectionKernelsService.cuh"
#include "SimulationData.cuh"
#include "SimulationKernelsService.cuh"
#include "SimulationParametersUpdateService.cuh"
#include "SimulationStatistics.cuh"
#include "StatisticsKernels.cuh"
#include "StatisticsKernelsService.cuh"
#include "StatisticsService.cuh"
#include "TO.cuh"
#include "TOProvider.cuh"
#include "TestKernelsService.cuh"

namespace
{
    std::chrono::milliseconds const StatisticsUpdate(30);
    ArraySizesForGpu const PreviewCapacityGpu{10000, 10000, 10000000};
    ArraySizesForTO const PreviewCapacityTO{1000, 1000, 1000, 10000, 10000, 10000, 10000000};
}

_SimulationCudaFacade::_SimulationCudaFacade(uint64_t timestep, SettingsForSimulation const& settings)
{
    initCuda();
    CudaMemoryManager::getInstance().reset();

    initSettingsPreviewData();

    _settings = settings;
    setSimulationParameters(settings.simulationParameters);
    setGpuConstants(settings.cudaSettings);

    log(Priority::Important, "initialize simulation");

    _cudaSimulationData = std::make_shared<SimulationData>();
    _cudaPreviewData = std::make_shared<SimulationData>();
    _cudaGeometryBuffers = std::make_shared<CudaGeometryBuffers>();
    _cudaSelectionResult = std::make_shared<SelectionResult>();
    _collectionTOProvider = std::make_shared<_TOProvider>();
    _cudaTOProvider = std::make_shared<_CudaTOProvider>();
    _cudaSimulationStatistics = std::make_shared<SimulationStatistics>();
    _cudaPreviewStatistics = std::make_shared<SimulationStatistics>();
    _maxAgeBalancer = std::make_shared<_MaxAgeBalancer>();

    _cudaSimulationData->init({_settings.worldSizeX, _settings.worldSizeY}, timestep);
    _cudaPreviewData->init({_settingsForPreview.worldSizeX, _settingsForPreview.worldSizeY}, 0);
    _cudaSimulationStatistics->init();
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
    _cudaSimulationData->resizeObjectsAndTempObjects({100000, 100000, 10000000});
    _cudaPreviewData->resizeObjectsAndTempObjects(PreviewCapacityGpu);

    auto memory = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();
    log(Priority::Important, std::to_string(memory / (1024 * 1024)) + " MB GPU memory used");
}

_SimulationCudaFacade::~_SimulationCudaFacade()
{
    _cudaSimulationData->free();
    _cudaPreviewData->free();
    _cudaSimulationStatistics->free();
    _cudaSelectionResult->free();

    SimulationKernelsService::get().shutdown();
    DataAccessKernelsService::get().shutdown();
    EditKernelsService::get().shutdown();
    GeometryKernelsService::get().shutdown();
    TestKernelsService::get().shutdown();
    StatisticsKernelsService::get().shutdown();
    SelectionKernelsService::get().shutdown();
    GarbageCollectorKernelsService::get().shutdown();

    _cudaTOProvider.reset();
    _collectionTOProvider.reset();

    CHECK_FOR_CUDA_ERROR(cudaDeviceReset());
    log(Priority::Important, "simulation closed");
}

void _SimulationCudaFacade::copyBuffersFromCudaToOpenGL(GeometryBuffers const& geometryBuffers, RealRect const& visibleWorldRect)
{
    checkAndProcessSimulationParameterChanges();
    auto simulationData = getSimulationDataPtrCopy();

    GeometryKernelsService::get().correctPositionsForRendering(_settings, simulationData, visibleWorldRect);
    auto numRenderObjects = GeometryKernelsService::get().getNumRenderObjects(_settings, simulationData, visibleWorldRect);
    geometryBuffers->updateNumObjects(numRenderObjects);

    if (GlobalSettings::get().isInterop()) {
        _cudaGeometryBuffers->registerBuffers(geometryBuffers);
        GeometryKernelsService::get().extractObjectData(_settings, simulationData, *_cudaGeometryBuffers, visibleWorldRect);
        syncAndCheck();
    } else {
        _cudaGeometryBuffers->allocateBuffersForNoInterop(numRenderObjects);
        GeometryKernelsService::get().extractObjectData(_settings, simulationData, *_cudaGeometryBuffers, visibleWorldRect);
        syncAndCheck();
        _cudaGeometryBuffers->copyToOpenGL(geometryBuffers, numRenderObjects);
    }

    GeometryKernelsService::get().restorePositions(_settings, simulationData);
    syncAndCheck();
}

void _SimulationCudaFacade::calcTimestep(uint64_t timesteps, bool forceUpdateStatistics)
{
    static int counter = 0;

    for (uint64_t i = 0; i < timesteps; ++i) {
        checkAndProcessSimulationParameterChanges();

        auto simulationData = getSimulationDataPtrCopy();
        SimulationKernelsService::get().calcTimestep(_settings, simulationData, *_cudaSimulationStatistics);
        {
            std::lock_guard lock(_mutexForSimulationData);
            ++_simulationTimestep;  // SimulationData::timestep is already updated in the kernels
        }
        syncAndCheck();

        //make check after every 10th call
        if (++counter % 10 == 0) {
            counter = 0;
            resizeArraysIfNecessary();
        }

        auto statistics = getStatisticsRawData();
        {
            std::lock_guard lock(_mutexForSimulationParameters);
            if (SimulationParametersUpdateService::get().updateSimulationParametersAfterTimestep(_settings, _maxAgeBalancer, simulationData, getCurrentTimestep(), statistics)) {
                CHECK_FOR_CUDA_ERROR(
                    cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
            }
        }
        auto now = std::chrono::steady_clock::now();
        if (!_lastStatisticsUpdateTime || now - *_lastStatisticsUpdateTime > StatisticsUpdate) {
            _lastStatisticsUpdateTime = now;
            updateStatistics();
        }
    }
    if (forceUpdateStatistics) {
        updateStatistics();
    }
}

void _SimulationCudaFacade::applyCataclysm(int power)
{
    for (int i = 0; i < power; ++i) {
        EditKernelsService::get().applyCataclysm(_settings.cudaSettings, getSimulationDataPtrCopy());
        syncAndCheck();
        resizeArraysIfNecessary();
    }
}

TO _SimulationCudaFacade::getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    DataAccessKernelsService::get().getData(_settings.cudaSettings, getSimulationDataPtrCopy(), rectUpperLeft, rectLowerRight, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideNewUnmanagedDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

TO _SimulationCudaFacade::getSelectedSimulationData(bool includeClusters)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    DataAccessKernelsService::get().getSelectedData(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

TO _SimulationCudaFacade::getInspectedSimulationData(std::vector<uint64_t> entityIds)
{
    InspectedEntityIds ids;
    if (entityIds.size() > Const::MaxInspectedObjects) {
        return TO{};
    }
    for (int i = 0; i < entityIds.size(); ++i) {
        ids.values[i] = entityIds.at(i);
    }
    if (entityIds.size() < Const::MaxInspectedObjects) {
        ids.values[entityIds.size()] = Const::MaxInspectedObjects_Break;
    }

    auto cudaTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    DataAccessKernelsService::get().getInspectedData(_settings.cudaSettings, getSimulationDataPtrCopy(), ids, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

TO _SimulationCudaFacade::getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    DataAccessKernelsService::get().getOverlayData(_settings.cudaSettings, getSimulationDataPtrCopy(), rectUpperLeft, rectLowerRight, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

TO _SimulationCudaFacade::getGenomeOfCreature(uint64_t creatureId, bool& found)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    found = DataAccessKernelsService::get().getGenomeOfCreature(_settings.cudaSettings, getSimulationDataPtrCopy(), creatureId, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

void _SimulationCudaFacade::addAndSelectSimulationData(TO const& to)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(to.capacities);
    copyDataTOtoGpu(cudaTO, to);

    auto sizeDelta = DataAccessKernelsService::get().estimateCapacityNeededForGpu(_settings.cudaSettings, cudaTO);
    resizeArraysIfNecessary(sizeDelta);

    SelectionKernelsService::get().removeSelection(_settings.cudaSettings, getSimulationDataPtrCopy());
    DataAccessKernelsService::get().addData(_settings.cudaSettings, getSimulationDataPtrCopy(), cudaTO, true);
    syncAndCheck();
    updateStatistics();
}

void _SimulationCudaFacade::setSimulationData(TO const& to)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(to.capacities);
    copyDataTOtoGpu(cudaTO, to);

    auto sizeDelta = DataAccessKernelsService::get().estimateCapacityNeededForGpu(_settings.cudaSettings, cudaTO);
    resizeArraysIfNecessary(sizeDelta);

    DataAccessKernelsService::get().clearData(_settings.cudaSettings, getSimulationDataPtrCopy());
    DataAccessKernelsService::get().addData(_settings.cudaSettings, getSimulationDataPtrCopy(), cudaTO, false);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::removeSelectedObjects(bool includeClusters)
{
    EditKernelsService::get().removeSelectedObjects(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::relaxSelectedObjects(bool includeClusters)
{
    EditKernelsService::get().relaxSelectedObjects(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

Ids _SimulationCudaFacade::getMaxIds() const
{
    return _cudaSimulationData->primaryNumberGen.getIds_host();
}

void _SimulationCudaFacade::uniformVelocitiesForSelectedObjects(bool includeClusters)
{
    EditKernelsService::get().uniformVelocities(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::makeSticky(bool includeClusters)
{
    EditKernelsService::get().makeSticky(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::removeStickiness(bool includeClusters)
{
    EditKernelsService::get().removeStickiness(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::setBarrier(bool value, bool includeClusters)
{
    EditKernelsService::get().setBarrier(_settings.cudaSettings, getSimulationDataPtrCopy(), value, includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::changeInspectedSimulationData(TO const& changeTO)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(changeTO.capacities);
    copyDataTOtoGpu(cudaTO, changeTO);

    EditKernelsService::get().changeSimulationData(_settings.cudaSettings, getSimulationDataPtrCopy(), cudaTO);
    syncAndCheck();

    updateStatistics();

    resizeArraysIfNecessary();
}

bool _SimulationCudaFacade::changeCreature(TO const& to)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(to.capacities);
    copyDataTOtoGpu(cudaTO, to);

    auto result = EditKernelsService::get().changeCreature(_settings.cudaSettings, getSimulationDataPtrCopy(), cudaTO);
    syncAndCheck();

    updateStatistics();

    resizeArraysIfNecessary();

    return result;
}

void _SimulationCudaFacade::applyForce(ApplyForceData const& applyData)
{
    EditKernelsService::get().applyForce(_settings.cudaSettings, getSimulationDataPtrCopy(), applyData);
    syncAndCheck();
}

void _SimulationCudaFacade::switchSelection(PointSelectionData const& pointData)
{
    SelectionKernelsService::get().switchSelection(_settings.cudaSettings, getSimulationDataPtrCopy(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::swapSelection(PointSelectionData const& pointData)
{
    SelectionKernelsService::get().swapSelection(_settings.cudaSettings, getSimulationDataPtrCopy(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::setSelection(AreaSelectionData const& selectionData)
{
    SelectionKernelsService::get().setSelection(_settings.cudaSettings, getSimulationDataPtrCopy(), selectionData);
    syncAndCheck();
}

SelectionShallowData _SimulationCudaFacade::getSelectionShallowData()
{
    EditKernelsService::get().getSelectionShallowData(_settings.cudaSettings, getSimulationDataPtrCopy(), *_cudaSelectionResult);
    syncAndCheck();
    return _cudaSelectionResult->getSelectionShallowData();
}

void _SimulationCudaFacade::shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& shallowUpdateData)
{
    EditKernelsService::get().shallowUpdateSelectedObjects(_settings.cudaSettings, getSimulationDataPtrCopy(), shallowUpdateData);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::removeSelection()
{
    SelectionKernelsService::get().removeSelection(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::updateSelection()
{
    SelectionKernelsService::get().updateSelection(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::colorSelectedObjects(unsigned char color, bool includeClusters)
{
    EditKernelsService::get().colorSelectedCells(_settings.cudaSettings, getSimulationDataPtrCopy(), color, includeClusters);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::reconnectSelectedObjects()
{
    EditKernelsService::get().reconnect(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::setDetached(bool value)
{
    EditKernelsService::get().setDetached(_settings.cudaSettings, getSimulationDataPtrCopy(), value);
    syncAndCheck();
}

void _SimulationCudaFacade::setGpuConstants(CudaSettings const& gpuConstants)
{
    _settings.cudaSettings = gpuConstants;
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

ArraySizesForTO _SimulationCudaFacade::estimateCapacityNeededForTO() const
{
    return DataAccessKernelsService::get().estimateCapacityNeededForTO(_settings.cudaSettings, getSimulationDataPtrCopy());
}

StatisticsRawData _SimulationCudaFacade::getStatisticsRawData()
{
    std::lock_guard lock(_mutexForStatistics);
    if (_statisticsData) {
        return *_statisticsData;
    } else {
        return StatisticsRawData();
    }
}

void _SimulationCudaFacade::updateStatistics()
{
    StatisticsKernelsService::get().updateStatistics(_settings.cudaSettings, getSimulationDataPtrCopy(), *_cudaSimulationStatistics);
    syncAndCheck();

    {
        std::lock_guard lock(_mutexForStatistics);
        _statisticsData = _cudaSimulationStatistics->getStatistics();
    }
    StatisticsService::get().addDataPoint(_statisticsHistory, _statisticsData->timeline, getCurrentTimestep());
}

StatisticsHistory const& _SimulationCudaFacade::getStatisticsHistory() const
{
    return _statisticsHistory;
}

void _SimulationCudaFacade::setStatisticsHistory(StatisticsHistoryData const& data)
{
    StatisticsService::get().rewriteHistory(_statisticsHistory, data, getCurrentTimestep());
}

void _SimulationCudaFacade::resetTimeIntervalStatistics()
{
    _cudaSimulationStatistics->resetAccumulatedStatistics();
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
        copyToDevice(_cudaSimulationData->timestep, &timestep); // Update GPU timestep
        _simulationTimestep = timestep;
    }
    StatisticsService::get().resetTime(_statisticsHistory, timestep);
}

void _SimulationCudaFacade::clear()
{
    DataAccessKernelsService::get().clearData(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::resizeArraysIfNecessary(ArraySizesForGpu const& sizeDelta)
{
    if (_cudaSimulationData->shouldResize(sizeDelta)) {
        resizeArrays(sizeDelta);
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
    _settingsForPreview.cudaSettings.numBlocks = 16;
}

void _SimulationCudaFacade::newPreview(TO const& to)
{
    auto cudaTO = _cudaTOProvider->provideDataTO(to.capacities);
    copyDataTOtoGpu(cudaTO, to);

    DataAccessKernelsService::get().clearData(_settings.cudaSettings, *_cudaPreviewData);
    DataAccessKernelsService::get().addData(_settings.cudaSettings, *_cudaPreviewData, cudaTO, false);
    syncAndCheck();
}

void _SimulationCudaFacade::calcTimestepsForPreview(std::chrono::milliseconds const& duration, bool detailSimulation)
{
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settingsForPreview.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));

    auto startTimepoint = std::chrono::steady_clock::now();
    do {

        SimulationKernelsService::get().calcTimestepForPreview(_settingsForPreview, *_cudaPreviewData, *_cudaPreviewStatistics, detailSimulation);
        syncAndCheck();

        ++_previewTimestep; // SimulationData::timestep is already updated in the kernels
    } while (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTimepoint) < duration);

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void _SimulationCudaFacade::calcTimestepsForPreview(int numSteps, bool detailSimulation)
{
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settingsForPreview.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));

    for (int i = 0; i < numSteps; ++i) {
        SimulationKernelsService::get().calcTimestepForPreview(_settingsForPreview, *_cudaPreviewData, *_cudaPreviewStatistics, detailSimulation);
        syncAndCheck();

        ++_previewTimestep; // SimulationData::timestep is already updated in the kernels
    }

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

uint64_t _SimulationCudaFacade::getCurrentTimestepForPreview()
{
    return _previewTimestep;
}

void _SimulationCudaFacade::setCurrentTimestepForPreview(uint64_t timestep)
{
    _previewTimestep = timestep;
    copyToDevice(_cudaPreviewData->timestep, &timestep);    // Update GPU timestep
}

TO _SimulationCudaFacade::getPreviewData()
{
    auto cudaTO = _cudaTOProvider->provideDataTO(PreviewCapacityTO);
    DataAccessKernelsService::get().getData(
        _settings.cudaSettings, *_cudaPreviewData, {-10, -10}, {_settingsForPreview.worldSizeX + 10, _settingsForPreview.worldSizeY + 10}, cudaTO);
    syncAndCheck();

    auto to = _collectionTOProvider->provideNewUnmanagedDataTO(cudaTO.capacities);
    copyDataTOtoHost(to, cudaTO);

    return to;
}

void _SimulationCudaFacade::testOnly_mutate(uint64_t cellId, MutationType mutationType)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_mutate(_settings.cudaSettings, getSimulationDataPtrCopy(), cellId, mutationType);
    syncAndCheck();

    resizeArraysIfNecessary();
}

void _SimulationCudaFacade::testOnly_mutationCheck(uint64_t cellId)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_mutationCheck(_settings.cudaSettings, getSimulationDataPtrCopy(), cellId);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_createConnection(uint64_t cellId1, uint64_t cellId2)
{
    checkAndProcessSimulationParameterChanges();
    TestKernelsService::get().testOnly_createConnection(_settings.cudaSettings, getSimulationDataPtrCopy(), cellId1, cellId2);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_cleanupAfterTimestep()
{
    checkAndProcessSimulationParameterChanges();
    GarbageCollectorKernelsService::get().cleanupAfterTimestep(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_cleanupAfterDataManipulation()
{
    checkAndProcessSimulationParameterChanges();
    GarbageCollectorKernelsService::get().cleanupAfterDataManipulation(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_resizeArrays(ArraySizesForGpu const& sizeDelta)
{
    checkAndProcessSimulationParameterChanges();
    resizeArrays(sizeDelta);
    syncAndCheck();
}

bool _SimulationCudaFacade::testOnly_arePointersValid()
{
    checkAndProcessSimulationParameterChanges();
    auto result = TestKernelsService::get().testOnly_arePointersValid(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
    return result;
}

NumRenderObjects _SimulationCudaFacade::testOnly_getNumRenderObjects()
{
    checkAndProcessSimulationParameterChanges();
    auto simulationData = getSimulationDataPtrCopy();
    RealRect visibleWorldRect = {{0, 0}, {static_cast<float>(_settings.worldSizeX), static_cast<float>(_settings.worldSizeY)}};
    auto result = GeometryKernelsService::get().getNumRenderObjects(_settings, simulationData, visibleWorldRect);
    syncAndCheck();
    return result;
}

std::vector<SelectedObjectVertexData> _SimulationCudaFacade::testOnly_getSelectedObjectData()
{
    checkAndProcessSimulationParameterChanges();
    auto simulationData = getSimulationDataPtrCopy();
    auto result = GeometryKernelsService::get().testOnly_getSelectedObjectData(_settings, simulationData);
    syncAndCheck();
    return result;
}

std::vector<ConnectionArrowVertexData> _SimulationCudaFacade::testOnly_getConnectionArrowData()
{
    checkAndProcessSimulationParameterChanges();
    auto simulationData = getSimulationDataPtrCopy();
    auto result = GeometryKernelsService::get().testOnly_getConnectionArrowData(_settings, simulationData);
    syncAndCheck();
    return result;
}

void _SimulationCudaFacade::initCuda()
{
    log(Priority::Important, "initialize CUDA");
    _gpuInfo = checkAndReturnGpuInfo();

    auto result = cudaSetDevice(_gpuInfo.deviceNumber);
    if (result != cudaSuccess) {
        throw std::runtime_error("CUDA device could not be initialized.");
    }

    cudaGetLastError();  //reset error code

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
    CHECK_FOR_CUDA_ERROR(cudaGetDeviceCount(&numberOfDevices));
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
        CHECK_FOR_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceNumber));

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
    if (highestComputeCapability < 600) {
        throw std::runtime_error("No CUDA device with compute capability of 6.0 or higher found.");
    }

    return *cachedResult;
}

void _SimulationCudaFacade::syncAndCheck()
{
    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());
}

void _SimulationCudaFacade::copyDataTOtoGpu(TO const& cudaTO, TO const& to)
{
    copyToDevice(cudaTO.numCells, to.numCells);
    copyToDevice(cudaTO.numParticles, to.numParticles);
    copyToDevice(cudaTO.numGenomes, to.numGenomes);
    copyToDevice(cudaTO.numCreatures, to.numCreatures);
    copyToDevice(cudaTO.numGenes, to.numGenes);
    copyToDevice(cudaTO.numNodes, to.numNodes);
    copyToDevice(cudaTO.heapSize, to.heapSize);

    copyToDevice(cudaTO.cells, to.cells, *to.numCells);
    copyToDevice(cudaTO.particles, to.particles, *to.numParticles);
    copyToDevice(cudaTO.genomes, to.genomes, *to.numGenomes);
    copyToDevice(cudaTO.creatures, to.creatures, *to.numCreatures);
    copyToDevice(cudaTO.genes, to.genes, *to.numGenes);
    copyToDevice(cudaTO.nodes, to.nodes, *to.numNodes);
    copyToDevice(cudaTO.heap, to.heap, *to.heapSize);
}

void _SimulationCudaFacade::copyDataTOtoHost(TO const& to, TO const& cudaTO)
{
    copyToHost(to.numCells, cudaTO.numCells);
    copyToHost(to.numParticles, cudaTO.numParticles);
    copyToHost(to.numGenomes, cudaTO.numGenomes);
    copyToHost(to.numCreatures, cudaTO.numCreatures);
    copyToHost(to.numGenes, cudaTO.numGenes);
    copyToHost(to.numNodes, cudaTO.numNodes);
    copyToHost(to.heapSize, cudaTO.heapSize);

    copyToHost(to.cells, cudaTO.cells, *to.numCells);
    copyToHost(to.particles, cudaTO.particles, *to.numParticles);
    copyToHost(to.genomes, cudaTO.genomes, *to.numGenomes);
    copyToHost(to.creatures, cudaTO.creatures, *to.numCreatures);
    copyToHost(to.genes, cudaTO.genes, *to.numGenes);
    copyToHost(to.nodes, cudaTO.nodes, *to.numNodes);
    copyToHost(to.heap, cudaTO.heap, *to.heapSize);
}

void _SimulationCudaFacade::resizeArrays(ArraySizesForGpu const& sizeDelta)
{
    log(Priority::Important, "resize arrays");

    _cudaSimulationData->resizeTempObjects(sizeDelta);

    if (!_cudaSimulationData->isEmpty()) {
        GarbageCollectorKernelsService::get().copyArrays(_settings.cudaSettings, getSimulationDataPtrCopy());
        syncAndCheck();

        _cudaSimulationData->resizeObjectsByMatchingTempObjects();

        GarbageCollectorKernelsService::get().swapArrays(_settings.cudaSettings, getSimulationDataPtrCopy());
        syncAndCheck();
    } else {
        _cudaSimulationData->resizeObjectsByMatchingTempObjects();
    }

    auto cellArraySize = _cudaSimulationData->objects.cells.getCapacity_host();
    auto particleArraySize = _cudaSimulationData->objects.particles.getCapacity_host();
    auto auxiliaryDataSize = _cudaSimulationData->objects.heap.getCapacity_host();

    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    log(Priority::Unimportant, "cell array capacity: " + std::to_string(cellArraySize));
    log(Priority::Unimportant, "particle array capacity: " + std::to_string(particleArraySize));
    log(Priority::Unimportant, "heap capacity: " + std::to_string(auxiliaryDataSize));

    auto const memorySizeAfter = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();
    log(Priority::Important, std::to_string(memorySizeAfter / (1024 * 1024)) + " MB GPU memory used");
}

void _SimulationCudaFacade::checkAndProcessSimulationParameterChanges()
{
    std::lock_guard lock(_mutexForSimulationParameters);
    if (_newSimulationParameters) {
        _settings.simulationParameters = SimulationParametersUpdateService::get().integrateChanges(
            _settings.simulationParameters, *_newSimulationParameters, _simulationParametersUpdateConfig);
        CHECK_FOR_CUDA_ERROR(
            cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
        _newSimulationParameters.reset();

        if (_cudaSimulationData) {
            SimulationKernelsService::get().prepareForSimulationParametersChanges(_settings, getSimulationDataPtrCopy());
        }
    }
}

SimulationData _SimulationCudaFacade::getSimulationDataPtrCopy() const
{
    std::lock_guard lock(_mutexForSimulationData    );
    return *_cudaSimulationData;
}
