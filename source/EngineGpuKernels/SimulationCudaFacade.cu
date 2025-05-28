#include "SimulationCudaFacade.cuh"

#include <functional>
#include <iostream>
#include <list>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>

#include "Base/Exceptions.h"
#include "Base/LoggingService.h"

#include "EngineInterface/InspectedEntityIds.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuSettings.h"
#include "EngineInterface/SpaceCalculator.h"
#include "EngineInterface/Ids.h"

#include "DataAccessKernels.cuh"
#include "ObjectTO.cuh"
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
#include "RenderingData.cuh"
#include "SimulationParametersUpdateService.cuh"
#include "TestKernelsService.cuh"
#include "StatisticsService.cuh"
#include "MaxAgeBalancer.cuh"
#include "CudaCollectionTOProvider.cuh"
#include "CollectionTOProvider.cuh"

namespace
{
    std::chrono::milliseconds const StatisticsUpdate(30);
}

_SimulationCudaFacade::_SimulationCudaFacade(uint64_t timestep, SettingsForSimulation const& settings)
{
    initCuda();
    CudaMemoryManager::getInstance().reset();

    _settings = settings;
    setSimulationParameters(settings.simulationParameters);
    setGpuConstants(settings.gpuSettings);

    log(Priority::Important, "initialize simulation");

    _cudaSimulationData = std::make_shared<SimulationData>();
    _cudaRenderingData = std::make_shared<RenderingData>();
    _cudaSelectionResult = std::make_shared<SelectionResult>();
    _collectionTOProvider = std::make_shared<_CollectionTOProvider>();
    _cudaCollectionTOProvider = std::make_shared<_CudaCollectionTOProvider>();
    _cudaSimulationStatistics = std::make_shared<SimulationStatistics>();
    _maxAgeBalancer = std::make_shared<_MaxAgeBalancer>();

    _cudaSimulationData->init({settings.worldSizeX, settings.worldSizeY}, timestep);
    _cudaRenderingData->init();
    _cudaSimulationStatistics->init();
    _cudaSelectionResult->init();

    _simulationKernels = std::make_shared<_SimulationKernelsService>();
    _dataAccessKernels = std::make_shared<_DataAccessKernelsService>();
    _garbageCollectorKernels = std::make_shared<_GarbageCollectorKernelsService>();
    _renderingKernels = std::make_shared<_RenderingKernelsService>();
    _editKernels = std::make_shared<_EditKernelsService>();
    _statisticsKernels = std::make_shared<_StatisticsKernelsService>();
    _testKernels = std::make_shared<_TestKernelsService>();

    //default array sizes for empty simulation (will be resized later if not sufficient)
    resizeArrays({100000, 100000, 100000});
}

_SimulationCudaFacade::~_SimulationCudaFacade()
{
    _cudaSimulationData->free();
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

    _cudaCollectionTOProvider.reset();
    _collectionTOProvider.reset();

    CHECK_FOR_CUDA_ERROR(cudaDeviceReset());
    log(Priority::Important, "simulation closed");
}

void* _SimulationCudaFacade::registerImageResource(GLuint image)
{
    //unregister old resource
    if (_cudaResource) {
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnregisterResource(_cudaResource));
    }

    //register new resource
    CHECK_FOR_CUDA_ERROR(
        cudaGraphicsGLRegisterImage(&_cudaResource, image, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

    return reinterpret_cast<void*>(_cudaResource);
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
        _editKernels->applyCataclysm(_settings.gpuSettings, getSimulationDataPtrCopy());
        syncAndCheck();
        resizeArraysIfNecessary();
    }
}

void _SimulationCudaFacade::drawVectorGraphics(
    float2 const& rectUpperLeft,
    float2 const& rectLowerRight,
    void* cudaResource,
    int2 const& imageSize,
    double zoom)
{
    checkAndProcessSimulationParameterChanges();

    auto cudaResourceImpl = reinterpret_cast<cudaGraphicsResource*>(cudaResource);
    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResourceImpl));

    cudaArray* mappedArray;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&mappedArray, cudaResourceImpl, 0, 0));

    _cudaRenderingData->resizeImageIfNecessary(imageSize);

    _renderingKernels->drawImage(_settings, rectUpperLeft, rectLowerRight, imageSize, static_cast<float>(zoom), getSimulationDataPtrCopy(), *_cudaRenderingData);
    syncAndCheck();

    const size_t widthBytes = sizeof(uint64_t) * imageSize.x;
    CHECK_FOR_CUDA_ERROR(cudaMemcpy2DToArray(
        mappedArray,
        0,
        0,
        _cudaRenderingData->imageData,
        widthBytes,
        widthBytes,
        imageSize.y,
        cudaMemcpyDeviceToDevice));

    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResourceImpl));
}

CollectionTO _SimulationCudaFacade::getSimulationData(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight)
{
    auto cudaDataTO = _cudaCollectionTOProvider->provideDataTO(estimateCapacityNeededForTO());
    _dataAccessKernels->getData(_settings.gpuSettings, getSimulationDataPtrCopy(), rectUpperLeft, rectLowerRight, cudaDataTO);
    syncAndCheck();

    auto dataTO = _collectionTOProvider->provideNewUnmanagedDataTO(cudaDataTO.capacities);
    copyDataTOtoHost(dataTO, cudaDataTO);

    return dataTO;
}

CollectionTO _SimulationCudaFacade::getSelectedSimulationData(bool includeClusters)
{
    auto cudaDataTO = _cudaCollectionTOProvider->provideDataTO(estimateCapacityNeededForTO());
    _dataAccessKernels->getSelectedData(_settings.gpuSettings, getSimulationDataPtrCopy(), includeClusters, cudaDataTO);
    syncAndCheck();

    auto dataTO = _collectionTOProvider->provideDataTO(cudaDataTO.capacities);
    copyDataTOtoHost(dataTO, cudaDataTO);

    return dataTO;
}

CollectionTO _SimulationCudaFacade::getInspectedSimulationData(std::vector<uint64_t> entityIds)
{
    InspectedEntityIds ids;
    if (entityIds.size() > Const::MaxInspectedObjects) {
        return CollectionTO{};
    }
    for (int i = 0; i < entityIds.size(); ++i) {
        ids.values[i] = entityIds.at(i);
    }
    if (entityIds.size() < Const::MaxInspectedObjects) {
        ids.values[entityIds.size()] = 0;
    }

    auto cudaDataTO = _cudaCollectionTOProvider->provideDataTO(estimateCapacityNeededForTO());
    _dataAccessKernels->getInspectedData(_settings.gpuSettings, getSimulationDataPtrCopy(), ids, cudaDataTO);
    syncAndCheck();

    auto dataTO = _collectionTOProvider->provideDataTO(cudaDataTO.capacities);
    copyDataTOtoHost(dataTO, cudaDataTO);

    return dataTO;
}

CollectionTO _SimulationCudaFacade::getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight)
{
    auto cudaDataTO = _cudaCollectionTOProvider->provideDataTO(estimateCapacityNeededForTO());
    _dataAccessKernels->getOverlayData(_settings.gpuSettings, getSimulationDataPtrCopy(), rectUpperLeft, rectLowerRight, cudaDataTO);
    syncAndCheck();

    auto dataTO = _collectionTOProvider->provideDataTO(cudaDataTO.capacities);
    copyDataTOtoHost(dataTO, cudaDataTO);

    return dataTO;
}

void _SimulationCudaFacade::addAndSelectSimulationData(CollectionTO const& dataTO)
{
    auto cudaDataTO = _cudaCollectionTOProvider->provideDataTO(dataTO.capacities);
    copyDataTOtoGpu(cudaDataTO, dataTO);

    auto sizeDelta = _dataAccessKernels->estimateCapacityNeededForGpu(_settings.gpuSettings, cudaDataTO);
    resizeArraysIfNecessary(sizeDelta);

    _editKernels->removeSelection(_settings.gpuSettings, getSimulationDataPtrCopy());
    _dataAccessKernels->addData(_settings.gpuSettings, getSimulationDataPtrCopy(), cudaDataTO, true);
    syncAndCheck();
    updateStatistics();
}

void _SimulationCudaFacade::setSimulationData(CollectionTO const& dataTO)
{
    auto cudaDataTO = _cudaCollectionTOProvider->provideDataTO(dataTO.capacities);
    copyDataTOtoGpu(cudaDataTO, dataTO);

    auto sizeDelta = _dataAccessKernels->estimateCapacityNeededForGpu(_settings.gpuSettings, cudaDataTO);
    resizeArraysIfNecessary(sizeDelta);

    _dataAccessKernels->clearData(_settings.gpuSettings, getSimulationDataPtrCopy());
    _dataAccessKernels->addData(_settings.gpuSettings, getSimulationDataPtrCopy(), cudaDataTO, false);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::removeSelectedObjects(bool includeClusters)
{
    _editKernels->removeSelectedObjects(_settings.gpuSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::relaxSelectedObjects(bool includeClusters)
{
    _editKernels->relaxSelectedObjects(_settings.gpuSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

Ids _SimulationCudaFacade::getMaxIds() const
{
    return _cudaSimulationData->primaryNumberGen.getIds_host();
}

void _SimulationCudaFacade::uniformVelocitiesForSelectedObjects(bool includeClusters)
{
    _editKernels->uniformVelocities(_settings.gpuSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::makeSticky(bool includeClusters)
{
    _editKernels->makeSticky(_settings.gpuSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::removeStickiness(bool includeClusters)
{
    _editKernels->removeStickiness(_settings.gpuSettings, getSimulationDataPtrCopy(), includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::setBarrier(bool value, bool includeClusters)
{
    _editKernels->setBarrier(_settings.gpuSettings, getSimulationDataPtrCopy(), value, includeClusters);
    syncAndCheck();
}

void _SimulationCudaFacade::changeInspectedSimulationData(CollectionTO const& changeDataTO)
{
    auto cudaDataTO = _cudaCollectionTOProvider->provideDataTO(changeDataTO.capacities);
    copyDataTOtoGpu(cudaDataTO, changeDataTO);

    _editKernels->changeSimulationData(_settings.gpuSettings, getSimulationDataPtrCopy(), cudaDataTO);
    syncAndCheck();

    updateStatistics();

    resizeArraysIfNecessary();
}

void _SimulationCudaFacade::applyForce(ApplyForceData const& applyData)
{
    _editKernels->applyForce(_settings.gpuSettings, getSimulationDataPtrCopy(), applyData);
    syncAndCheck();
}

void _SimulationCudaFacade::switchSelection(PointSelectionData const& pointData)
{
    _editKernels->switchSelection(_settings.gpuSettings, getSimulationDataPtrCopy(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::swapSelection(PointSelectionData const& pointData)
{
    _editKernels->swapSelection(_settings.gpuSettings, getSimulationDataPtrCopy(), pointData);
    syncAndCheck();
}

void _SimulationCudaFacade::setSelection(AreaSelectionData const& selectionData)
{
    _editKernels->setSelection(_settings.gpuSettings, getSimulationDataPtrCopy(), selectionData);
}

 SelectionShallowData _SimulationCudaFacade::getSelectionShallowData()
{
    _editKernels->getSelectionShallowData(_settings.gpuSettings, getSimulationDataPtrCopy(), *_cudaSelectionResult);
    syncAndCheck();
    return _cudaSelectionResult->getSelectionShallowData();
}

void _SimulationCudaFacade::shallowUpdateSelectedObjects(ShallowUpdateSelectionData const& shallowUpdateData)
{
    _editKernels->shallowUpdateSelectedObjects(_settings.gpuSettings, getSimulationDataPtrCopy(), shallowUpdateData);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::removeSelection()
{
    _editKernels->removeSelection(_settings.gpuSettings, getSimulationDataPtrCopy());
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::updateSelection()
{
    _editKernels->updateSelection(_settings.gpuSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::colorSelectedObjects(unsigned char color, bool includeClusters)
{
    _editKernels->colorSelectedCells(_settings.gpuSettings, getSimulationDataPtrCopy(), color, includeClusters);
    syncAndCheck();

    updateStatistics();
}

void _SimulationCudaFacade::reconnectSelectedObjects()
{
    _editKernels->reconnect(_settings.gpuSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::setDetached(bool value)
{
    _editKernels->setDetached(_settings.gpuSettings, getSimulationDataPtrCopy(), value);
    syncAndCheck();
}

void _SimulationCudaFacade::setGpuConstants(GpuSettings const& gpuConstants)
{
    _settings.gpuSettings = gpuConstants;

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaThreadSettings, &gpuConstants, sizeof(GpuSettings), 0, cudaMemcpyHostToDevice));
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
    return _dataAccessKernels->estimateCapacityNeededForTO(_settings.gpuSettings, getSimulationDataPtrCopy());
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
    _statisticsKernels->updateStatistics(_settings.gpuSettings, getSimulationDataPtrCopy(), *_cudaSimulationStatistics);
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
    _dataAccessKernels->clearData(_settings.gpuSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::resizeArraysIfNecessary(ArraySizesForGpu const& sizeDelta)
{
    if (_cudaSimulationData->shouldResize(sizeDelta)) {
        resizeArrays(sizeDelta);
    }
}

void _SimulationCudaFacade::testOnly_mutate(uint64_t cellId, MutationType mutationType)
{
    checkAndProcessSimulationParameterChanges();
    _testKernels->testOnly_mutate(_settings.gpuSettings, getSimulationDataPtrCopy(), cellId, mutationType);
    syncAndCheck();

    resizeArraysIfNecessary();
}

void _SimulationCudaFacade::testOnly_mutationCheck(uint64_t cellId)
{
    checkAndProcessSimulationParameterChanges();
    _testKernels->testOnly_mutationCheck(_settings.gpuSettings, getSimulationDataPtrCopy(), cellId);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_createConnection(uint64_t cellId1, uint64_t cellId2)
{
    checkAndProcessSimulationParameterChanges();
    _testKernels->testOnly_createConnection(_settings.gpuSettings, getSimulationDataPtrCopy(), cellId1, cellId2);
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_cleanupAfterTimestep()
{
    checkAndProcessSimulationParameterChanges();
    _garbageCollectorKernels->cleanupAfterTimestep(_settings.gpuSettings, getSimulationDataPtrCopy());
    syncAndCheck();
}

void _SimulationCudaFacade::testOnly_cleanupAfterDataManipulation()
{
    checkAndProcessSimulationParameterChanges();
    _garbageCollectorKernels->cleanupAfterDataManipulation(_settings.gpuSettings, getSimulationDataPtrCopy());
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
    auto result = _testKernels->testOnly_areArraysValid(_settings.gpuSettings, getSimulationDataPtrCopy());
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

void _SimulationCudaFacade::copyDataTOtoGpu(CollectionTO const& cudaDataTO, CollectionTO const& dataTO)
{
    copyToDevice(cudaDataTO.numCells, dataTO.numCells);
    copyToDevice(cudaDataTO.numParticles, dataTO.numParticles);
    copyToDevice(cudaDataTO.numGenomes, dataTO.numGenomes);
    copyToDevice(cudaDataTO.numGenes, dataTO.numGenes);
    copyToDevice(cudaDataTO.numNodes, dataTO.numNodes);
    copyToDevice(cudaDataTO.heapSize, dataTO.heapSize);

    copyToDevice(cudaDataTO.cells, dataTO.cells, *dataTO.numCells);
    copyToDevice(cudaDataTO.particles, dataTO.particles, *dataTO.numParticles);
    copyToDevice(cudaDataTO.genomes, dataTO.genomes, *dataTO.numGenomes);
    copyToDevice(cudaDataTO.genes, dataTO.genes, *dataTO.numGenes);
    copyToDevice(cudaDataTO.nodes, dataTO.nodes, *dataTO.numNodes);
    copyToDevice(cudaDataTO.heap, dataTO.heap, *dataTO.heapSize);
}

void _SimulationCudaFacade::copyDataTOtoHost(CollectionTO const& dataTO, CollectionTO const& cudaDataTO)
{
    copyToHost(dataTO.numCells, cudaDataTO.numCells);
    copyToHost(dataTO.numParticles, cudaDataTO.numParticles);
    copyToHost(dataTO.numGenomes, cudaDataTO.numGenomes);
    copyToHost(dataTO.numGenes, cudaDataTO.numGenes);
    copyToHost(dataTO.numNodes, cudaDataTO.numNodes);
    copyToHost(dataTO.heapSize, cudaDataTO.heapSize);

    copyToHost(dataTO.cells, cudaDataTO.cells, *dataTO.numCells);
    copyToHost(dataTO.particles, cudaDataTO.particles, *dataTO.numParticles);
    copyToHost(dataTO.genomes, cudaDataTO.genomes, *dataTO.numGenomes);
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

    _cudaSimulationData->resizeTargetObjects(sizeDelta);

    if (!_cudaSimulationData->isEmpty()) {
        _garbageCollectorKernels->copyArrays(_settings.gpuSettings, getSimulationDataPtrCopy());
        syncAndCheck();

        _cudaSimulationData->resizeObjects();

        _garbageCollectorKernels->swapArrays(_settings.gpuSettings, getSimulationDataPtrCopy());
        syncAndCheck();
    } else {
        _cudaSimulationData->resizeObjects();
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
