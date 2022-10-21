#include "CudaSimulationFacade.cuh"

#include <functional>
#include <iostream>
#include <list>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include <cuda/helper_cuda.h>

#include "Base/Exceptions.h"
#include "EngineInterface/InspectedEntityIds.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuSettings.h"

#include "Base/LoggingService.h"
#include "DataAccessKernels.cuh"
#include "TOs.cuh"
#include "Base.cuh"
#include "GarbageCollectorKernels.cuh"
#include "ConstantMemory.cuh"
#include "CudaMemoryManager.cuh"
#include "CudaMonitorData.cuh"
#include "Objects.cuh"
#include "Map.cuh"
#include "MonitorKernels.cuh"
#include "EditKernels.cuh"
#include "RenderingKernels.cuh"
#include "SimulationData.cuh"
#include "SimulationKernelsLauncher.cuh"
#include "DataAccessKernelsLauncher.cuh"
#include "RenderingKernelsLauncher.cuh"
#include "EditKernelsLauncher.cuh"
#include "MonitorKernelsLauncher.cuh"
#include "SimulationResult.cuh"
#include "SelectionResult.cuh"
#include "RenderingData.cuh"

namespace
{
    class CudaInitializer
    {
    public:
        static void init() { [[maybe_unused]] static CudaInitializer instance; }

        CudaInitializer()
        {
            int deviceNumber = getDeviceNumberOfHighestComputeCapability();

            auto result = cudaSetDevice(deviceNumber);
            if (result != cudaSuccess) {
                throw SystemRequirementNotMetException("CUDA device could not be initialized.");
            }

            std::stringstream stream;
            stream << "device " << deviceNumber << " is set";
            log(Priority::Important, stream.str());
        }

        ~CudaInitializer() { cudaDeviceReset(); }

    private:
        int getDeviceNumberOfHighestComputeCapability()
        {
            int result = 0;
            int numberOfDevices;
            CHECK_FOR_CUDA_ERROR(cudaGetDeviceCount(&numberOfDevices));
            if (numberOfDevices < 1) {
                throw SystemRequirementNotMetException("No CUDA device found.");
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
                stream << "device " << deviceNumber << ": " << prop.name << " with compute capability " << prop.major
                       << "." << prop.minor;
                log(Priority::Important, stream.str());

                int computeCapability = prop.major * 100 + prop.minor;
                if (computeCapability > highestComputeCapability) {
                    result = deviceNumber;
                    highestComputeCapability = computeCapability;
                }
            }
            if (highestComputeCapability < 502) {
                throw SystemRequirementNotMetException(
                    "No CUDA device with compute capability of 5.2 or higher found.");
            }

            return result;
        }
    };
}

void _CudaSimulationFacade::initCuda()
{
    CudaInitializer::init();
}

_CudaSimulationFacade::_CudaSimulationFacade(uint64_t timestep, Settings const& settings)
{
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    setSimulationParameters(settings.simulationParameters);
    setSimulationParametersSpots(settings.simulationParametersSpots);
    setGpuConstants(settings.gpuSettings);
    setFlowFieldSettings(settings.flowFieldSettings);

    log(Priority::Important, "initialize simulation");

    _timestepOfLastMonitorData = timestep;
    _cudaSimulationData = std::make_shared<SimulationData>();
    _cudaRenderingData = std::make_shared<RenderingData>();
    _cudaSimulationResult = std::make_shared<SimulationResult>();
    _cudaSelectionResult = std::make_shared<SelectionResult>();
    _cudaAccessTO = std::make_shared<DataTO>();
    _cudaMonitorData = std::make_shared<CudaMonitorData>();

    _cudaSimulationData->init({settings.generalSettings.worldSizeX, settings.generalSettings.worldSizeY}, timestep);
    _cudaRenderingData->init();
    _cudaMonitorData->init();
    _cudaSimulationResult->init();
    _cudaSelectionResult->init();

    _simulationKernels = std::make_shared<_SimulationKernelsLauncher>();
    _dataAccessKernels = std::make_shared<_DataAccessKernelsLauncher>();
    _garbageCollectorKernels = std::make_shared<_GarbageCollectorKernelsLauncher>();
    _renderingKernels = std::make_shared<_RenderingKernelsLauncher>();
    _editKernels = std::make_shared<_EditKernelsLauncher>();
    _monitorKernels = std::make_shared<_MonitorKernelsLauncher>();

    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().acquireMemory<uint64_t>(1, _cudaAccessTO->numAuxiliaryData);

    //default array sizes for empty simulation (will be resized later if not sufficient)
    resizeArrays({100000, 100000});
}

_CudaSimulationFacade::~_CudaSimulationFacade()
{
    _cudaSimulationData->free();
    _cudaRenderingData->free();
    _cudaMonitorData->free();
    _cudaSimulationResult->free();
    _cudaSelectionResult->free();

    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->cells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->particles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->auxiliaryData);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numAuxiliaryData);

    log(Priority::Important, "close simulation");
}

void* _CudaSimulationFacade::registerImageResource(GLuint image)
{
    cudaGraphicsResource* cudaResource;

    CHECK_FOR_CUDA_ERROR(
        cudaGraphicsGLRegisterImage(&cudaResource, image, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

    return reinterpret_cast<void*>(cudaResource);
}

void _CudaSimulationFacade::calcTimestep()
{
    _simulationKernels->calcTimestep(_settings, getSimulationDataIntern(), *_cudaSimulationResult);
    syncAndCheck();

    automaticResizeArrays();

    std::lock_guard lock(_mutex);
    ++_cudaSimulationData->timestep;
}

void _CudaSimulationFacade::drawVectorGraphics(
    float2 const& rectUpperLeft,
    float2 const& rectLowerRight,
    void* cudaResource,
    int2 const& imageSize,
    double zoom)
{
    auto cudaResourceImpl = reinterpret_cast<cudaGraphicsResource*>(cudaResource);
    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResourceImpl));

    cudaArray* mappedArray;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&mappedArray, cudaResourceImpl, 0, 0));

    _cudaRenderingData->resizeImageIfNecessary(imageSize);

    _renderingKernels->drawImage(
        _settings.gpuSettings, rectUpperLeft, rectLowerRight, imageSize, static_cast<float>(zoom), getSimulationDataIntern(), *_cudaRenderingData);
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

void _CudaSimulationFacade::getSimulationData(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataTO const& dataTO)
{
    _dataAccessKernels->getData(_settings.gpuSettings, getSimulationDataIntern(), rectUpperLeft, rectLowerRight, *_cudaAccessTO);
    syncAndCheck();

    copyDataTOtoHost(dataTO);
}

void _CudaSimulationFacade::getSelectedSimulationData(bool includeClusters, DataTO const& dataTO)
{
    _dataAccessKernels->getSelectedData(_settings.gpuSettings, getSimulationDataIntern(), includeClusters, *_cudaAccessTO);
    syncAndCheck();

    copyDataTOtoHost(dataTO);
}

void _CudaSimulationFacade::getInspectedSimulationData(std::vector<uint64_t> entityIds, DataTO const& dataTO)
{
    InspectedEntityIds ids;
    if (entityIds.size() > Const::MaxInspectedEntities) {
        return;
    }
    for (int i = 0; i < entityIds.size(); ++i) {
        ids.values[i] = entityIds.at(i);
    }
    if (entityIds.size() < Const::MaxInspectedEntities) {
        ids.values[entityIds.size()] = 0;
    }
    _dataAccessKernels->getInspectedData(_settings.gpuSettings, getSimulationDataIntern(), ids, *_cudaAccessTO);
    syncAndCheck();
    copyDataTOtoHost(dataTO);
}

void _CudaSimulationFacade::getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataTO const& dataTO)
{
    _dataAccessKernels->getOverlayData(_settings.gpuSettings, getSimulationDataIntern(), rectUpperLeft, rectLowerRight, *_cudaAccessTO);
    syncAndCheck();

    copyToHost(dataTO.numCells, _cudaAccessTO->numCells);
    copyToHost(dataTO.numParticles, _cudaAccessTO->numParticles);
    copyToHost(dataTO.cells, _cudaAccessTO->cells, *dataTO.numCells);
    copyToHost(dataTO.particles, _cudaAccessTO->particles, *dataTO.numParticles);
}

void _CudaSimulationFacade::addAndSelectSimulationData(DataTO const& dataTO)
{
    copyDataTOtoDevice(dataTO);
    _editKernels->removeSelection(_settings.gpuSettings, getSimulationDataIntern());
    _dataAccessKernels->addData(_settings.gpuSettings, getSimulationDataIntern(), *_cudaAccessTO, true, true);
    syncAndCheck();
}

void _CudaSimulationFacade::setSimulationData(DataTO const& dataTO)
{
    copyDataTOtoDevice(dataTO);
    _dataAccessKernels->clearData(_settings.gpuSettings, getSimulationDataIntern());
    _dataAccessKernels->addData(_settings.gpuSettings, getSimulationDataIntern(), *_cudaAccessTO, false, false);
    syncAndCheck();
}

void _CudaSimulationFacade::removeSelectedEntities(bool includeClusters)
{
    _editKernels->removeSelectedEntities(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
}

void _CudaSimulationFacade::relaxSelectedEntities(bool includeClusters)
{
    _editKernels->relaxSelectedEntities(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
}

void _CudaSimulationFacade::uniformVelocitiesForSelectedEntities(bool includeClusters)
{
    _editKernels->uniformVelocitiesForSelectedEntities(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
}

void _CudaSimulationFacade::makeSticky(bool includeClusters)
{
    _editKernels->makeSticky(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
}

void _CudaSimulationFacade::removeStickiness(bool includeClusters)
{
    _editKernels->removeStickiness(_settings.gpuSettings, getSimulationDataIntern(), includeClusters);
    syncAndCheck();
}

void _CudaSimulationFacade::setBarrier(bool value, bool includeClusters)
{
    _editKernels->setBarrier(_settings.gpuSettings, getSimulationDataIntern(), value, includeClusters);
    syncAndCheck();
}

void _CudaSimulationFacade::changeInspectedSimulationData(DataTO const& changeDataTO)
{
    copyDataTOtoDevice(changeDataTO);
    _editKernels->changeSimulationData(_settings.gpuSettings, getSimulationDataIntern(), *_cudaAccessTO);
    syncAndCheck();
}

void _CudaSimulationFacade::applyForce(ApplyForceData const& applyData)
{
    _editKernels->applyForce(_settings.gpuSettings, getSimulationDataIntern(), applyData);
    syncAndCheck();
}

void _CudaSimulationFacade::switchSelection(PointSelectionData const& pointData)
{
    _editKernels->switchSelection(_settings.gpuSettings, getSimulationDataIntern(), pointData);
    syncAndCheck();
}

void _CudaSimulationFacade::swapSelection(PointSelectionData const& pointData)
{
    _editKernels->swapSelection(_settings.gpuSettings, getSimulationDataIntern(), pointData);
    syncAndCheck();
}

void _CudaSimulationFacade::setSelection(AreaSelectionData const& selectionData)
{
    _editKernels->setSelection(_settings.gpuSettings, getSimulationDataIntern(), selectionData);
}

 SelectionShallowData _CudaSimulationFacade::getSelectionShallowData()
{
    _editKernels->getSelectionShallowData(_settings.gpuSettings, getSimulationDataIntern(), *_cudaSelectionResult);
    syncAndCheck();
    return _cudaSelectionResult->getSelectionShallowData();
}

void _CudaSimulationFacade::shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& shallowUpdateData)
{
    _editKernels->shallowUpdateSelectedEntities(_settings.gpuSettings, getSimulationDataIntern(), shallowUpdateData);
    syncAndCheck();
}

void _CudaSimulationFacade::removeSelection()
{
    _editKernels->removeSelection(_settings.gpuSettings, getSimulationDataIntern());
    syncAndCheck();
}

void _CudaSimulationFacade::updateSelection()
{
    _editKernels->updateSelection(_settings.gpuSettings, getSimulationDataIntern());
    syncAndCheck();
}

void _CudaSimulationFacade::colorSelectedEntities(unsigned char color, bool includeClusters)
{
    _editKernels->colorSelectedCells(_settings.gpuSettings, getSimulationDataIntern(), color, includeClusters);
    syncAndCheck();
}

void _CudaSimulationFacade::reconnectSelectedEntities()
{
    _editKernels->reconnectSelectedEntities(_settings.gpuSettings, getSimulationDataIntern());
    syncAndCheck();
}

void _CudaSimulationFacade::setGpuConstants(GpuSettings const& gpuConstants)
{
    _settings.gpuSettings = gpuConstants;

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaThreadSettings, &gpuConstants, sizeof(GpuSettings), 0, cudaMemcpyHostToDevice));
}

auto _CudaSimulationFacade::getArraySizes() const -> ArraySizes
{
    return {
        _cudaSimulationData->objects.cells.getSize_host(),
        _cudaSimulationData->objects.particles.getSize_host(),
        _cudaSimulationData->objects.auxiliaryData.getSize_host()
    };
}

MonitorData _CudaSimulationFacade::getMonitorData()
{
    _monitorKernels->getMonitorData(_settings.gpuSettings, getSimulationDataIntern(), *_cudaMonitorData);
    syncAndCheck();
    
    MonitorData result;
    auto monitorData = _cudaMonitorData->getMonitorData(getCurrentTimestep());
    result.timestep = monitorData.timeStep;
    for (int i = 0; i < 7; ++i) {
        result.numCellsByColor[i] = monitorData.numCellsByColor[i];
    }
    result.numConnections = monitorData.numConnections;
    result.numParticles = monitorData.numParticles;
    result.totalInternalEnergy = monitorData.totalInternalEnergy;

    auto processStatistics = _cudaSimulationResult->getAndResetProcessMonitorData();
    result.numCreatedCells = processStatistics.createdCells;
    result.numSuccessfulAttacks = processStatistics.sucessfulAttacks;
    result.numFailedAttacks = processStatistics.failedAttacks;
    result.numMuscleActivities = processStatistics.muscleActivities;

    auto deltaTime = static_cast<int64_t>(result.timestep) - static_cast<int64_t>(_timestepOfLastMonitorData);
    auto divisor = deltaTime > 0 ? deltaTime : 1;
    result.numCreatedCells /= divisor;
    result.numSuccessfulAttacks /= divisor;
    result.numFailedAttacks /= divisor;
    result.numMuscleActivities /= divisor;
    if (deltaTime != 0) {
        _timestepOfLastMonitorData = result.timestep;
    }
    return result;
}

void _CudaSimulationFacade::resetProcessMonitorData()
{
    _cudaSimulationResult->getAndResetProcessMonitorData();
}

uint64_t _CudaSimulationFacade::getCurrentTimestep() const
{
    std::lock_guard lock(_mutex);
    return _cudaSimulationData->timestep;
}

void _CudaSimulationFacade::setCurrentTimestep(uint64_t timestep)
{
    std::lock_guard lock(_mutex);
    _cudaSimulationData->timestep = timestep;
}

void _CudaSimulationFacade::setSimulationParameters(SimulationParameters const& parameters)
{
    _settings.simulationParameters = parameters;
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(cudaSimulationParameters, &parameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void _CudaSimulationFacade::setSimulationParametersSpots(SimulationParametersSpots const& spots)
{
    _settings.simulationParametersSpots = spots;
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaSimulationParametersSpots, &spots, sizeof(SimulationParametersSpots), 0, cudaMemcpyHostToDevice));
}

void _CudaSimulationFacade::setFlowFieldSettings(FlowFieldSettings const& settings)
{
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaFlowFieldSettings, &settings, sizeof(FlowFieldSettings), 0, cudaMemcpyHostToDevice));

    _settings.flowFieldSettings = settings;
}


void _CudaSimulationFacade::clear()
{
    _dataAccessKernels->clearData(_settings.gpuSettings, getSimulationDataIntern());
    syncAndCheck();
}

void _CudaSimulationFacade::resizeArraysIfNecessary(ArraySizes const& additionals)
{
    if (_cudaSimulationData->shouldResize(additionals)) {
        resizeArrays(additionals);
    }
}

void _CudaSimulationFacade::syncAndCheck()
{
    cudaDeviceSynchronize();
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());
}

void _CudaSimulationFacade::copyDataTOtoDevice(DataTO const& dataTO)
{
    copyToDevice(_cudaAccessTO->numCells, dataTO.numCells);
    copyToDevice(_cudaAccessTO->numParticles, dataTO.numParticles);
    copyToDevice(_cudaAccessTO->numAuxiliaryData, dataTO.numAuxiliaryData);

    copyToDevice(_cudaAccessTO->cells, dataTO.cells, *dataTO.numCells);
    copyToDevice(_cudaAccessTO->particles, dataTO.particles, *dataTO.numParticles);
    copyToDevice(_cudaAccessTO->auxiliaryData, dataTO.auxiliaryData, *dataTO.numAuxiliaryData);
}

void _CudaSimulationFacade::copyDataTOtoHost(DataTO const& dataTO)
{
    copyToHost(dataTO.numCells, _cudaAccessTO->numCells);
    copyToHost(dataTO.numParticles, _cudaAccessTO->numParticles);
    copyToHost(dataTO.numAuxiliaryData, _cudaAccessTO->numAuxiliaryData);

    copyToHost(dataTO.cells, _cudaAccessTO->cells, *dataTO.numCells);
    copyToHost(dataTO.particles, _cudaAccessTO->particles, *dataTO.numParticles);
    copyToHost(dataTO.auxiliaryData, _cudaAccessTO->auxiliaryData, *dataTO.numAuxiliaryData);
}

void _CudaSimulationFacade::automaticResizeArrays()
{
    //make check after every 10th time step
    std::lock_guard lock(_mutex);
    if (_cudaSimulationData->timestep % 10 == 0) {
        if (_cudaSimulationResult->isArrayResizeNeeded()) {
            resizeArrays({0, 0});
        }
    }
}

void _CudaSimulationFacade::resizeArrays(ArraySizes const& additionals)
{
    log(Priority::Important, "resize arrays");

    _cudaSimulationData->resizeTargetObjects(additionals);
    if (!_cudaSimulationData->isEmpty()) {
        _garbageCollectorKernels->copyArrays(_settings.gpuSettings, getSimulationDataIntern());
        syncAndCheck();

        _cudaSimulationData->resizeObjects();

        _garbageCollectorKernels->swapArrays(_settings.gpuSettings, getSimulationDataIntern());
        syncAndCheck();
    } else {
        _cudaSimulationData->resizeObjects();
    }

    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->cells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->particles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->auxiliaryData);

    auto cellArraySize = _cudaSimulationData->objects.cells.getSize_host();
    CudaMemoryManager::getInstance().acquireMemory<CellTO>(cellArraySize, _cudaAccessTO->cells);
    auto particleArraySize = _cudaSimulationData->objects.particlePointers.getSize_host();
    CudaMemoryManager::getInstance().acquireMemory<ParticleTO>(particleArraySize, _cudaAccessTO->particles);
    auto auxiliaryDataSize = _cudaSimulationData->objects.auxiliaryData.getSize_host();
    CudaMemoryManager::getInstance().acquireMemory<uint8_t>(auxiliaryDataSize, _cudaAccessTO->auxiliaryData);

    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    log(Priority::Unimportant, "cell array size: " + std::to_string(cellArraySize));
    log(Priority::Unimportant, "particle array size: " + std::to_string(particleArraySize));
    log(Priority::Unimportant, "auxiliary data size: " + std::to_string(auxiliaryDataSize));

    auto const memorySizeAfter = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();
    log(Priority::Important, std::to_string(memorySizeAfter / (1024 * 1024)) + " MB GPU memory acquired");
}

SimulationData _CudaSimulationFacade::getSimulationDataIntern() const
{
    std::lock_guard lock(_mutex);
    return *_cudaSimulationData;
}
