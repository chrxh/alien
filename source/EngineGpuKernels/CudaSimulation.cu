#include "CudaSimulation.cuh"

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
#include "Base/ServiceLocator.h"
#include "AccessKernels.cuh"
#include "AccessTOs.cuh"
#include "Base.cuh"
#include "GarbageCollectorKernels.cuh"
#include "ConstantMemory.cuh"
#include "CudaMemoryManager.cuh"
#include "CudaMonitorData.cuh"
#include "Entities.cuh"
#include "Map.cuh"
#include "MonitorKernels.cuh"
#include "EditKernels.cuh"
#include "RenderingKernels.cuh"
#include "SimulationData.cuh"
#include "SimulationKernelLauncher.cuh"
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

            auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
            auto result = cudaSetDevice(deviceNumber);
            if (result != cudaSuccess) {
                throw SystemRequirementNotMetException("CUDA device could not be initialized.");
            }

            std::stringstream stream;
            stream << "device " << deviceNumber << " is set";
            loggingService->logMessage(Priority::Important, stream.str());
        }

        ~CudaInitializer() { cudaDeviceReset(); }

    private:
        int getDeviceNumberOfHighestComputeCapability()
        {
            auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
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
                loggingService->logMessage(Priority::Important, stream.str());
            }

            int highestComputeCapability = 0;
            for (int deviceNumber = 0; deviceNumber < numberOfDevices; ++deviceNumber) {
                cudaDeviceProp prop;
                CHECK_FOR_CUDA_ERROR(cudaGetDeviceProperties(&prop, deviceNumber));

                std::stringstream stream;
                stream << "device " << deviceNumber << ": " << prop.name << " with compute capability " << prop.major
                       << "." << prop.minor;
                loggingService->logMessage(Priority::Important, stream.str());

                int computeCapability = prop.major * 100 + prop.minor;
                if (computeCapability > highestComputeCapability) {
                    result = deviceNumber;
                    highestComputeCapability = computeCapability;
                }
            }
            if (highestComputeCapability < 600) {
                throw SystemRequirementNotMetException(
                    "No CUDA device with compute capability of 6.0 or higher found.");
            }

            return result;
        }
    };
}

void _CudaSimulation::initCuda()
{
    CudaInitializer::init();
}

_CudaSimulation::_CudaSimulation(uint64_t timestep, Settings const& settings, GpuSettings const& gpuSettings)
{
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    setSimulationParameters(settings.simulationParameters);
    setSimulationParametersSpots(settings.simulationParametersSpots);
    setGpuConstants(gpuSettings);
    setFlowFieldSettings(settings.flowFieldSettings);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "initialize simulation");

    _currentTimestep.store(timestep);
    _cudaSimulationData = new SimulationData();
    _cudaRenderingData = new RenderingData();
    _cudaSimulationResult = new SimulationResult();
    _cudaSelectionResult = new SelectionResult();
    _cudaAccessTO = new DataAccessTO();
    _cudaMonitorData = new CudaMonitorData();

    _simulationKernels = new SimulationKernelLauncher();

    int2 worldSize{settings.generalSettings.worldSizeX, settings.generalSettings.worldSizeY};
    _cudaSimulationData->init(worldSize);
    _cudaRenderingData->init();
    _cudaMonitorData->init();
    _cudaSimulationResult->init();
    _cudaSelectionResult->init();

    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numTokens);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numStringBytes);
    CudaMemoryManager::getInstance().acquireMemory<char>(Const::MetadataMemorySize, _cudaAccessTO->stringBytes);

    //default array sizes for empty simulation (will be resized later if not sufficient)
    resizeArrays({100000, 100000, 10000});
}

_CudaSimulation::~_CudaSimulation()
{
    _cudaSimulationData->free();
    _cudaRenderingData->free();
    _cudaMonitorData->free();
    _cudaSimulationResult->free();
    _cudaSelectionResult->free();

    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->cells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->particles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->tokens);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->stringBytes);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numTokens);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numStringBytes);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "close simulation");

    delete _cudaAccessTO;
    delete _cudaSimulationData;
    delete _cudaRenderingData;
    delete _cudaMonitorData;
    delete _simulationKernels;
}

void* _CudaSimulation::registerImageResource(GLuint image)
{
    cudaGraphicsResource* cudaResource;

    CHECK_FOR_CUDA_ERROR(
        cudaGraphicsGLRegisterImage(&cudaResource, image, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

    return reinterpret_cast<void*>(cudaResource);
}

void _CudaSimulation::calcTimestep()
{
    _simulationKernels->calcTimestep(_gpuSettings, *_cudaSimulationData, *_cudaSimulationResult);
    automaticResizeArrays();
    ++_currentTimestep;
}

void _CudaSimulation::drawVectorGraphics(
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

    DEPRECATED_KERNEL_CALL_HOST_SYNC(
        drawImageKernel,
        rectUpperLeft,
        rectLowerRight,
        imageSize,
        static_cast<float>(zoom),
        *_cudaSimulationData,
        *_cudaRenderingData);

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

void _CudaSimulation::getSimulationData(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataAccessTO const& dataTO)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaGetSimulationData, rectUpperLeft, rectLowerRight, *_cudaSimulationData, *_cudaAccessTO);
    copyFromGpu(dataTO);
}

void _CudaSimulation::getSelectedSimulationData(bool includeClusters, DataAccessTO const& dataTO)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaGetSelectedSimulationData, *_cudaSimulationData, includeClusters, * _cudaAccessTO);
    copyFromGpu(dataTO);
}

void _CudaSimulation::getInspectedSimulationData(std::vector<uint64_t> entityIds, DataAccessTO const& dataTO)
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
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaGetInspectedSimulationData, *_cudaSimulationData, ids, *_cudaAccessTO);
    copyFromGpu(dataTO);
}

void _CudaSimulation::getOverlayData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(
        cudaGetSimulationOverlayData, rectUpperLeft, rectLowerRight, *_cudaSimulationData, *_cudaAccessTO);
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        dataTO.cells, _cudaAccessTO->cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(dataTO.numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        dataTO.particles,
        _cudaAccessTO->particles,
        sizeof(CellAccessTO) * (*dataTO.numParticles),
        cudaMemcpyDeviceToHost));
}

void _CudaSimulation::addAndSelectSimulationData(DataAccessTO const& dataTO)
{
    copyToGpu(dataTO);
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaRemoveSelection, *_cudaSimulationData);
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaSetSimulationAccessData, *_cudaSimulationData, *_cudaAccessTO, true);
}

void _CudaSimulation::setSimulationData(DataAccessTO const& dataTO)
{
    copyToGpu(dataTO);
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaClearData, *_cudaSimulationData);
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaSetSimulationAccessData, *_cudaSimulationData, *_cudaAccessTO, false);
}

void _CudaSimulation::removeSelectedEntities(bool includeClusters)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaRemoveSelectedEntities, *_cudaSimulationData, includeClusters);
}

void _CudaSimulation::changeInspectedSimulationData(DataAccessTO const& changeDataTO)
{
    copyToGpu(changeDataTO);
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaChangeSimulationData, *_cudaSimulationData, *_cudaAccessTO);
}

void _CudaSimulation::applyForce(ApplyForceData const& applyData)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaApplyForce, applyData, *_cudaSimulationData);
}

void _CudaSimulation::switchSelection(PointSelectionData const& pointData)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaSwitchSelection, pointData, *_cudaSimulationData);
}

void _CudaSimulation::swapSelection(PointSelectionData const& pointData)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaSwapSelection, pointData, *_cudaSimulationData);
}

void _CudaSimulation::setSelection(AreaSelectionData const& selectionData)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaSetSelection, selectionData, *_cudaSimulationData);
}

 SelectionShallowData _CudaSimulation::getSelectionShallowData()
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaGetSelectionShallowData, *_cudaSimulationData, *_cudaSelectionResult);
    return _cudaSelectionResult->getSelectionShallowData();
}

void _CudaSimulation::shallowUpdateSelectedEntities(ShallowUpdateSelectionData const& shallowUpdateData)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaShallowUpdateSelectedEntities, shallowUpdateData, *_cudaSimulationData);
}

void _CudaSimulation::removeSelection()
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaRemoveSelection, *_cudaSimulationData);
}

void _CudaSimulation::updateSelection()
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaUpdateSelection, *_cudaSimulationData);
}

void _CudaSimulation::colorSelectedEntities(unsigned char color, bool includeClusters)
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaColorSelectedEntities, *_cudaSimulationData, color, includeClusters);
}

void _CudaSimulation::setGpuConstants(GpuSettings const& gpuConstants)
{
    _gpuSettings = gpuConstants;

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaThreadSettings, &gpuConstants, sizeof(GpuSettings), 0, cudaMemcpyHostToDevice));
}

auto _CudaSimulation::getArraySizes() const -> ArraySizes
{
    return {
        _cudaSimulationData->entities.cells.getSize_host(),
        _cudaSimulationData->entities.particles.getSize_host(),
        _cudaSimulationData->entities.tokens.getSize_host()};
}

OverallStatistics _CudaSimulation::getMonitorData()
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaGetCudaMonitorData, *_cudaSimulationData, *_cudaMonitorData);
    
    OverallStatistics result;

    auto monitorData = _cudaMonitorData->getMonitorData(getCurrentTimestep());
    result.timeStep = monitorData.timeStep;
    result.numCells = monitorData.numCells;
    result.numParticles = monitorData.numParticles;
    result.numTokens = monitorData.numTokens;
    result.totalInternalEnergy = monitorData.totalInternalEnergy;

    auto processStatistics = _cudaSimulationResult->getStatistics();
    result.numCreatedCells = processStatistics.createdCells;
    result.numSuccessfulAttacks = processStatistics.sucessfulAttacks;
    result.numFailedAttacks = processStatistics.failedAttacks;
    result.numMuscleActivities = processStatistics.muscleActivities;
    return result;
}

uint64_t _CudaSimulation::getCurrentTimestep() const
{
    return _currentTimestep.load();
}

void _CudaSimulation::setCurrentTimestep(uint64_t timestep)
{
    _currentTimestep.store(timestep);
}

void _CudaSimulation::setSimulationParameters(SimulationParameters const& parameters)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaSimulationParameters, &parameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void _CudaSimulation::setSimulationParametersSpots(SimulationParametersSpots const& spots)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaSimulationParametersSpots, &spots, sizeof(SimulationParametersSpots), 0, cudaMemcpyHostToDevice));
}

void _CudaSimulation::setFlowFieldSettings(FlowFieldSettings const& settings)
{
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaFlowFieldSettings, &settings, sizeof(FlowFieldSettings), 0, cudaMemcpyHostToDevice));
}


void _CudaSimulation::clear()
{
    DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaClearData, *_cudaSimulationData);
}

void _CudaSimulation::resizeArraysIfNecessary(ArraySizes const& additionals)
{
    if (_cudaSimulationData->shouldResize(
            additionals.cellArraySize, additionals.particleArraySize, additionals.tokenArraySize)) {
        resizeArrays(additionals);
    }
}

void _CudaSimulation::copyToGpu(DataAccessTO const& dataTO)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->numCells, dataTO.numCells, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(_cudaAccessTO->numParticles, dataTO.numParticles, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->numTokens, dataTO.numTokens, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(_cudaAccessTO->numStringBytes, dataTO.numStringBytes, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        _cudaAccessTO->cells, dataTO.cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        _cudaAccessTO->particles,
        dataTO.particles,
        sizeof(ParticleAccessTO) * (*dataTO.numParticles),
        cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        _cudaAccessTO->tokens, dataTO.tokens, sizeof(TokenAccessTO) * (*dataTO.numTokens), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        _cudaAccessTO->stringBytes,
        dataTO.stringBytes,
        sizeof(char) * (*dataTO.numStringBytes),
        cudaMemcpyHostToDevice));
}

void _CudaSimulation::copyFromGpu(DataAccessTO const& dataTO)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(dataTO.numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numTokens, _cudaAccessTO->numTokens, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(dataTO.numStringBytes, _cudaAccessTO->numStringBytes, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        dataTO.cells, _cudaAccessTO->cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        dataTO.particles,
        _cudaAccessTO->particles,
        sizeof(ParticleAccessTO) * (*dataTO.numParticles),
        cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        dataTO.tokens, _cudaAccessTO->tokens, sizeof(TokenAccessTO) * (*dataTO.numTokens), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        dataTO.stringBytes,
        _cudaAccessTO->stringBytes,
        sizeof(char) * (*dataTO.numStringBytes),
        cudaMemcpyDeviceToHost));
}

void _CudaSimulation::automaticResizeArrays()
{
    //make check after every 10th time step
    if (_currentTimestep.load() % 10 == 0) {
        if (_cudaSimulationResult->isArrayResizeNeeded()) {
            resizeArrays({0, 0, 0});
        }
    }
}

void _CudaSimulation::resizeArrays(ArraySizes const& additionals)
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "resize arrays");

    _cudaSimulationData->resizeEntitiesForCleanup(
        additionals.cellArraySize, additionals.particleArraySize, additionals.tokenArraySize);
    if (!_cudaSimulationData->isEmpty()) {
        DEPRECATED_KERNEL_CALL_HOST_SYNC(cudaCopyEntities, *_cudaSimulationData);
        _cudaSimulationData->resizeRemainings();
        _cudaSimulationData->swap();
    } else {
        _cudaSimulationData->resizeRemainings();
    }

    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->cells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->particles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->tokens);

    auto cellArraySize = _cudaSimulationData->entities.cells.getSize_host();
    auto tokenArraySize = _cudaSimulationData->entities.tokens.getSize_host();
    CudaMemoryManager::getInstance().acquireMemory<CellAccessTO>(cellArraySize, _cudaAccessTO->cells);
    CudaMemoryManager::getInstance().acquireMemory<ParticleAccessTO>(cellArraySize, _cudaAccessTO->particles);
    CudaMemoryManager::getInstance().acquireMemory<TokenAccessTO>(tokenArraySize, _cudaAccessTO->tokens);

    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

    loggingService->logMessage(Priority::Unimportant, "cell array size: " + std::to_string(cellArraySize));
    loggingService->logMessage(Priority::Unimportant, "particle array size: " + std::to_string(cellArraySize));
    loggingService->logMessage(Priority::Unimportant, "token array size: " + std::to_string(tokenArraySize));

        auto const memorySizeAfter = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();
    loggingService->logMessage(Priority::Important, std::to_string(memorySizeAfter / (1024 * 1024)) + " MB GPU memory acquired");
}
