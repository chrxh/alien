#include "SimulationCudaFacade.cuh"

#include <functional>
#include <iostream>
#include <list>

#include <cuda_runtime.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>

#include "Base/Exceptions.h"
#include "Base/LoggingService.h"
#include "Base/Macros.h"

#include "EngineInterface/InspectedEntityIds.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/CudaSettings.h"
#include "EngineInterface/SpaceCalculator.h"
#include "EngineInterface/Ids.h"

#include "DataAccessKernels.cuh"
#include "TO.cuh"
#include "Base.cuh"
#include "GarbageCollectorKernels.cuh"
#include "ConstantMemory.cuh"
#include "CudaMemoryManager.cuh"
#include "SimulationStatistics.cuh"
#include "Objects.cuh"
#include "Map.cuh"
#include "StatisticsKernels.cuh"
#include "EditKernels.cuh"
#include "RenderingKernels.cuh"
#include "SimulationData.cuh"
#include "SimulationKernelsService.cuh"
#include "DataAccessKernelsService.cuh"
#include "RenderingKernelsService.cuh"
#include "EditKernelsService.cuh"
#include "StatisticsKernelsService.cuh"
#include "SelectionResult.cuh"
#include "BufferData.cuh"
#include "SimulationParametersUpdateService.cuh"
#include "TestKernelsService.cuh"
#include "StatisticsService.cuh"
#include "MaxAgeBalancer.cuh"
#include "CudaTOProvider.cuh"
#include "TOProvider.cuh"

namespace
{
    std::chrono::milliseconds const StatisticsUpdate(30);
    ArraySizesForGpu const PreviewCapacityGpu{10000, 10000, 10000000};
    ArraySizesForTO const PreviewCapacityTO{1000, 1000, 10000, 10000, 10000, 10000000};
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
    _cudaRenderingData = std::make_shared<BufferData>();
    _cudaSelectionResult = std::make_shared<SelectionResult>();
    _collectionTOProvider = std::make_shared<_TOProvider>();
    _cudaTOProvider = std::make_shared<_CudaTOProvider>();
    _cudaSimulationStatistics = std::make_shared<SimulationStatistics>();
    _cudaPreviewStatistics = std::make_shared<SimulationStatistics>();
    _maxAgeBalancer = std::make_shared<_MaxAgeBalancer>();

    _cudaSimulationData->init({_settings.worldSizeX, _settings.worldSizeY}, timestep);
    _cudaPreviewData->init({_settingsForPreview.worldSizeX, _settingsForPreview.worldSizeY}, 0);
    _cudaRenderingData->init();
    _cudaSimulationStatistics->init();
    _cudaPreviewStatistics->init();
    _cudaSelectionResult->init();

    _simulationKernels = std::make_shared<_SimulationKernelsService>();
    _dataAccessKernels = std::make_shared<_DataAccessKernelsService>();
    _garbageCollectorKernels = std::make_shared<_GarbageCollectorKernelsService>();
    _renderingKernels = std::make_shared<_RenderingKernelsService>();
    _editKernels = std::make_shared<_EditKernelsService>();
    _statisticsKernels = std::make_shared<_StatisticsKernelsService>();
    _testKernels = std::make_shared<_TestKernelsService>();

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
    _cudaRenderingData->free();
    _cudaSimulationStatistics->free();
    _cudaSelectionResult->free();

    _simulationKernels.reset();
    _dataAccessKernels.reset();
    _garbageCollectorKernels.reset();
    _renderingKernels.reset();
    _editKernels.reset();
    _statisticsKernels.reset();
    _testKernels.reset();

    _cudaTOProvider.reset();
    _collectionTOProvider.reset();

    CHECK_FOR_CUDA_ERROR(cudaDeviceReset());
    log(Priority::Important, "simulation closed");
}

NumRenderObjects _SimulationCudaFacade::copyBuffersFromCudaToOpenGL(GeometryBuffers const& geometryBuffers)
{
    checkAndProcessSimulationParameterChanges();
    auto simulationData = getSimulationDataPtrCopy();

    auto numRenderObjects = _renderingKernels->getNumRenderObjects(_settings, simulationData);
    _cudaRenderingData->resizeObjectBufferIfNecessary(numRenderObjects, geometryBuffers);
    _cudaRenderingData->registerBuffers(geometryBuffers);

    _renderingKernels->extractObjectData(_settings, simulationData, *_cudaRenderingData);
    syncAndCheck();

    return numRenderObjects;
}

void _SimulationCudaFacade::calcTimestep(uint64_t timesteps, bool forceUpdateStatistics)
{
    for (uint64_t i = 0; i < timesteps; ++i) {
        checkAndProcessSimulationParameterChanges();

        auto simulationData = getSimulationDataPtrCopy();
        _simulationKernels->calcTimestep(_settings, simulationData, *_cudaSimulationStatistics);
        syncAndCheck();

        automaticResizeArrays();

        {
            std::lock_guard lock(_mutexForSimulationData);
            ++_cudaSimulationData->timestep;
        }
        auto statistics = getStatisticsRawData();
        {
            std::lock_guard lock(_mutexForSimulationParameters);
            if (SimulationParametersUpdateService::get().updateSimulationParametersAfterTimestep(_settings, _maxAgeBalancer, simulationData, statistics)) {
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
        _editKernels->applyCataclysm(_settings.cudaSettings, getSimulationDataPtrCopy());
        syncAndCheck();
        resizeArraysIfNecessary();
    }
}

TO _SimulationCudaFacade::getSimulationData(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight)
{
    auto cudaDataTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    _dataAccessKernels->getData(_settings.cudaSettings, getSimulationDataPtrCopy(), rectUpperLeft, rectLowerRight, cudaDataTO);
    syncAndCheck();

    auto dataTO = _collectionTOProvider->provideNewUnmanagedDataTO(cudaDataTO.capacities);
    copyDataTOtoHost(dataTO, cudaDataTO);

    return dataTO;
}

TO _SimulationCudaFacade::getSelectedSimulationData(bool includeClusters)
{
    auto cudaDataTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    _dataAccessKernels->getSelectedData(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters, cudaDataTO);
    syncAndCheck();

    auto dataTO = _collectionTOProvider->provideDataTO(cudaDataTO.capacities);
    copyDataTOtoHost(dataTO, cudaDataTO);

    return dataTO;
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

    auto cudaDataTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    _dataAccessKernels->getInspectedData(_settings.cudaSettings, getSimulationDataPtrCopy(), ids, cudaDataTO);
    syncAndCheck();

    auto dataTO = _collectionTOProvider->provideDataTO(cudaDataTO.capacities);
    copyDataTOtoHost(dataTO, cudaDataTO);

    return dataTO;
}

TO _SimulationCudaFacade::getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight)
{
    auto cudaDataTO = _cudaTOProvider->provideDataTO(estimateCapacityNeededForTO());
    _dataAccessKernels->getOverlayData(_settings.cudaSettings, getSimulationDataPtrCopy(), rectUpperLeft, rectLowerRight, cudaDataTO);
    syncAndCheck();

    auto dataTO = _collectionTOProvider->provideDataTO(cudaDataTO.capacities);
    copyDataTOtoHost(dataTO, cudaDataTO);

    return dataTO;
}

void _SimulationCudaFacade::addAndSelectSimulationData(TO const& dataTO)
{
    auto cudaDataTO = _cudaTOProvider->provideDataTO(dataTO.capacities);
    copyDataTOtoGpu(cudaDataTO, dataTO);

    auto sizeDelta = _dataAccessKernels->estimateCapacityNeededForGpu(_settings.cudaSettings, cudaDataTO);
    resizeArraysIfNecessary(sizeDelta);

    _editKernels->removeSelection(_settings.cudaSettings, getSimulationDataPtrCopy());
    _dataAccessKernels->addData(_settings.cudaSettings, getSimulationDataPtrCopy(), cudaDataTO, true);
    syncAndCheck();
    updateStatistics();
}

void _SimulationCudaFacade::setSimulationData(TO const& dataTO)
{
    auto cudaDataTO = _cudaTOProvider->provideDataTO(dataTO.capacities);
    copyDataTOtoGpu(cudaDataTO, dataTO);

    auto sizeDelta = _dataAccessKernels->estimateCapacityNeededForGpu(_settings.cudaSettings, cudaDataTO);
    resizeArraysIfNecessary(sizeDelta);

    _dataAccessKernels->clearData(_settings.cudaSettings, getSimulationDataPtrCopy());
    _dataAccessKernels->addData(_settings.cudaSettings, getSimulationDataPtrCopy(), cudaDataTO, false);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::removeSelectedObjects(bool includeClusters)
{
    _editKernels->removeSelectedObjects(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::relaxSelectedObjects(bool includeClusters)
{
    _editKernels->relaxSelectedObjects(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

Ids _SimulationCudaFacade::getMaxIds() const
{
    return _cudaSimulationData->primaryNumberGen.getIds_host();
}

void _SimulationCudaFacade::uniformVelocitiesForSelectedObjects(bool includeClusters)
{
    _editKernels->uniformVelocities(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::makeSticky(bool includeClusters)
{
    _editKernels->makeSticky(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::removeStickiness(bool includeClusters)
{
    _editKernels->removeStickiness(_settings.cudaSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::setBarrier(bool value, bool includeClusters)
{
    _editKernels->setBarrier(_settings.cudaSettings, getSimulationDataPtrCopy(), value, includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::changeInspectedSimulationData(TO const& changeDataTO)
{
    auto cudaDataTO = _cudaTOProvider->provideDataTO(changeDataTO.capacities);
    copyDataTOtoGpu(cudaDataTO, changeDataTO);

    _editKernels->changeSimulationData(_settings.cudaSettings, getSimulationDataPtrCopy(), cudaDataTO);
    syncAndCheck();

    updateStatistics();

    resizeArraysIfNecessary();
}

bool _SimulationCudaFacade::changeCreature(TO const& dataTO)
{
    auto cudaDataTO = _cudaTOProvider->provideDataTO(dataTO.capacities);
    copyDataTOtoGpu(cudaDataTO, dataTO);

    auto result = _editKernels->changeCreature(_settings.cudaSettings, getSimulationDataPtrCopy(), cudaDataTO);
    syncAndCheck();

    updateStatistics();

    resizeArraysIfNecessary();

    return result;
}

void _SimulationCudaFacade::applyForce(ApplyForceData const& applyData)
{
    _editKernels->applyForce(_settings.cudaSettings, getSimulationDataPtrCopy(), applyData);
    syncAndCheck();
}

void _SimulationCudaFacade::switchSelection(PointSelectionData const& pointData)
{
    _editKernels->switchSelection(_settings.cudaSettings, getSimulationDataPtrCopy(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::swapSelection(PointSelectionData const& pointData)
{
    _editKernels->swapSelection(_settings.cudaSettings, getSimulationDataPtrCopy(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::setSelection(AreaSelectionData const& selectionData)
{
    _editKernels->setSelection(_settings.cudaSettings, getSimulationDataPtrCopy(), selectionData);
}

 SelectionShallowData _SimulationCudaFacade::getSelectionShallowData()
{
    _editKernels->getSelectionShallowData(_settings.cudaSettings, getSimulationDataPtrCopy(), *_cudaSelectionResult);
    syncAndCheck();
    return _cudaSelectionResult->getSelectionShallowData();
}

void _SimulationCudaFacade::shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& shallowUpdateData)
{
    _editKernels->shallowUpdateSelectedObjects(_settings.cudaSettings, getSimulationDataPtrCopy(), shallowUpdateData);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::removeSelection()
{
    _editKernels->removeSelection(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::updateSelection()
{
    _editKernels->updateSelection(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::colorSelectedObjects(unsigned char color, bool includeClusters)
{
    _editKernels->colorSelectedCells(_settings.cudaSettings, getSimulationDataPtrCopy(), color, includeClusters);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::reconnectSelectedObjects()
{
    _editKernels->reconnect(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::setDetached(bool value)
{
    _editKernels->setDetached(_settings.cudaSettings, getSimulationDataPtrCopy(), value);
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
    return _dataAccessKernels->estimateCapacityNeededForTO(_settings.cudaSettings, getSimulationDataPtrCopy());
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
    _statisticsKernels->updateStatistics(_settings.cudaSettings, getSimulationDataPtrCopy(), *_cudaSimulationStatistics);
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
    return _cudaSimulationData->timestep;
}

void _SimulationCudaFacade::setCurrentTimestep(uint64_t timestep)
{
    {
        std::lock_guard lock(_mutexForSimulationData);
        _cudaSimulationData->timestep = timestep;
    }
    StatisticsService::get().resetTime(_statisticsHistory, timestep);
}

void _SimulationCudaFacade::clear()
{
    _dataAccessKernels->clearData(_settings.cudaSettings, getSimulationDataPtrCopy());
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

void _SimulationCudaFacade::newPreview(TO const& dataTO)
{
    auto cudaDataTO = _cudaTOProvider->provideDataTO(dataTO.capacities);
    copyDataTOtoGpu(cudaDataTO, dataTO);

    _dataAccessKernels->clearData(_settings.cudaSettings, *_cudaPreviewData);
    _dataAccessKernels->addData(_settings.cudaSettings, *_cudaPreviewData, cudaDataTO, false);
    syncAndCheck();
}

void _SimulationCudaFacade::calcTimestepsForPreview(std::chrono::milliseconds const& duration, bool detailSimulation)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaSimulationParameters, &_settingsForPreview.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));

    auto startTimepoint = std::chrono::steady_clock::now();
    do {

        _simulationKernels->calcTimestepForPreview(_settingsForPreview, *_cudaPreviewData, *_cudaPreviewStatistics, detailSimulation);
        syncAndCheck();

        ++_cudaPreviewData->timestep;
    } while (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTimepoint) < duration);

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void _SimulationCudaFacade::calcTimestepsForPreview(int numSteps, bool detailSimulation)
{
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settingsForPreview.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));

    auto startTimepoint = std::chrono::steady_clock::now();
    for (int i = 0; i < numSteps; ++i) {
        _simulationKernels->calcTimestepForPreview(_settingsForPreview, *_cudaPreviewData, *_cudaPreviewStatistics, detailSimulation);
        syncAndCheck();

        ++_cudaPreviewData->timestep;
    }

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

uint64_t _SimulationCudaFacade::getCurrentTimestepForPreview()
{
    return _cudaPreviewData->timestep;
}

void _SimulationCudaFacade::setCurrentTimestepForPreview(uint64_t timestep)
{
    _cudaPreviewData->timestep = timestep;
}

TO _SimulationCudaFacade::getPreviewData()
{
    auto cudaDataTO = _cudaTOProvider->provideDataTO(PreviewCapacityTO);
    _dataAccessKernels->getData(
        _settings.cudaSettings, *_cudaPreviewData, {-10, -10}, {_settingsForPreview.worldSizeX + 10, _settingsForPreview.worldSizeY + 10}, cudaDataTO);
    syncAndCheck();

    auto dataTO = _collectionTOProvider->provideNewUnmanagedDataTO(cudaDataTO.capacities);
    copyDataTOtoHost(dataTO, cudaDataTO);

    return dataTO;
}

void _SimulationCudaFacade::testOnly_mutate(uint64_t cellId, MutationType mutationType)
{
    checkAndProcessSimulationParameterChanges();
    _testKernels->testOnly_mutate(_settings.cudaSettings, getSimulationDataPtrCopy(), cellId, mutationType);
    syncAndCheck();

    resizeArraysIfNecessary();
}

void _SimulationCudaFacade::testOnly_mutationCheck(uint64_t cellId)
{
    checkAndProcessSimulationParameterChanges();
    _testKernels->testOnly_mutationCheck(_settings.cudaSettings, getSimulationDataPtrCopy(), cellId);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_createConnection(uint64_t cellId1, uint64_t cellId2)
{
    checkAndProcessSimulationParameterChanges();
    _testKernels->testOnly_createConnection(_settings.cudaSettings, getSimulationDataPtrCopy(), cellId1, cellId2);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_cleanupAfterTimestep()
{
    checkAndProcessSimulationParameterChanges();
    _garbageCollectorKernels->cleanupAfterTimestep(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_cleanupAfterDataManipulation()
{
    checkAndProcessSimulationParameterChanges();
    _garbageCollectorKernels->cleanupAfterDataManipulation(_settings.cudaSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_resizeArrays(ArraySizesForGpu const& sizeDelta)
{
    checkAndProcessSimulationParameterChanges();
    resizeArrays(sizeDelta);
    syncAndCheck();
}

bool _SimulationCudaFacade::testOnly_areArraysValid()
{
    checkAndProcessSimulationParameterChanges();
    auto result = _testKernels->testOnly_areArraysValid(_settings.cudaSettings, getSimulationDataPtrCopy());
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

    cudaGetLastError(); //reset error code

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

void _SimulationCudaFacade::copyDataTOtoGpu(TO const& cudaDataTO, TO const& dataTO)
{
    copyToDevice(cudaDataTO.numCells, dataTO.numCells);
    copyToDevice(cudaDataTO.numParticles, dataTO.numParticles);
    copyToDevice(cudaDataTO.numCreatures, dataTO.numCreatures);
    copyToDevice(cudaDataTO.numGenes, dataTO.numGenes);
    copyToDevice(cudaDataTO.numNodes, dataTO.numNodes);
    copyToDevice(cudaDataTO.heapSize, dataTO.heapSize);

    copyToDevice(cudaDataTO.cells, dataTO.cells, *dataTO.numCells);
    copyToDevice(cudaDataTO.particles, dataTO.particles, *dataTO.numParticles);
    copyToDevice(cudaDataTO.creatures, dataTO.creatures, *dataTO.numCreatures);
    copyToDevice(cudaDataTO.genes, dataTO.genes, *dataTO.numGenes);
    copyToDevice(cudaDataTO.nodes, dataTO.nodes, *dataTO.numNodes);
    copyToDevice(cudaDataTO.heap, dataTO.heap, *dataTO.heapSize);
}

void _SimulationCudaFacade::copyDataTOtoHost(TO const& dataTO, TO const& cudaDataTO)
{
    copyToHost(dataTO.numCells, cudaDataTO.numCells);
    copyToHost(dataTO.numParticles, cudaDataTO.numParticles);
    copyToHost(dataTO.numCreatures, cudaDataTO.numCreatures);
    copyToHost(dataTO.numGenes, cudaDataTO.numGenes);
    copyToHost(dataTO.numNodes, cudaDataTO.numNodes);
    copyToHost(dataTO.heapSize, cudaDataTO.heapSize);

    copyToHost(dataTO.cells, cudaDataTO.cells, *dataTO.numCells);
    copyToHost(dataTO.particles, cudaDataTO.particles, *dataTO.numParticles);
    copyToHost(dataTO.creatures, cudaDataTO.creatures, *dataTO.numCreatures);
    copyToHost(dataTO.genes, cudaDataTO.genes, *dataTO.numGenes);
    copyToHost(dataTO.nodes, cudaDataTO.nodes, *dataTO.numNodes);
    copyToHost(dataTO.heap, cudaDataTO.heap, *dataTO.heapSize);
}

void _SimulationCudaFacade::automaticResizeArrays()
{
    uint64_t timestep;
    {
        std::lock_guard lock(_mutexForSimulationData);
        timestep = _cudaSimulationData->timestep;
    }
    //make check after every 10th time step
    if (timestep % 10 == 0) {
        resizeArraysIfNecessary();
    }
}

void _SimulationCudaFacade::resizeArrays(ArraySizesForGpu const& sizeDelta)
{
    log(Priority::Important, "resize arrays");

    _cudaSimulationData->resizeTempObjects(sizeDelta);

    if (!_cudaSimulationData->isEmpty()) {
        _garbageCollectorKernels->copyArrays(_settings.cudaSettings, getSimulationDataPtrCopy());
        syncAndCheck();

        _cudaSimulationData->resizeObjectsByMatchingTempObjects();

        _garbageCollectorKernels->swapArrays(_settings.cudaSettings, getSimulationDataPtrCopy());
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
        _settings.simulationParameters =
            SimulationParametersUpdateService::get().integrateChanges(_settings.simulationParameters, *_newSimulationParameters, _simulationParametersUpdateConfig);
        CHECK_FOR_CUDA_ERROR(
            cudaMemcpyToSymbol(cudaSimulationParameters, &_settings.simulationParameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
        _newSimulationParameters.reset();

        if (_cudaSimulationData) {
            _simulationKernels->prepareForSimulationParametersChanges(_settings, getSimulationDataPtrCopy());
        }
    }
}

SimulationData _SimulationCudaFacade::getSimulationDataPtrCopy() const
{
    std::lock_guard lock(_mutexForSimulationData);
    return *_cudaSimulationData;
}
