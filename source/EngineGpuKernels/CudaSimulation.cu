#include "CudaSimulation.cuh"

#include <functional>
#include <iostream>
#include <list>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <device_launch_parameters.h>
#include <helper_cuda.h>

#include "Base/Exceptions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/GpuSettings.h"

#include "Base/LoggingService.h"
#include "Base/ServiceLocator.h"
#include "AccessKernels.cuh"
#include "AccessTOs.cuh"
#include "Base.cuh"
#include "CleanupKernels.cuh"
#include "ConstantMemory.cuh"
#include "CudaMemoryManager.cuh"
#include "CudaMonitorData.cuh"
#include "Entities.cuh"
#include "Map.cuh"
#include "MonitorKernels.cuh"
#include "PhysicalActionKernels.cuh"
#include "RenderingKernels.cuh"
#include "SimulationData.cuh"
#include "SimulationKernels.cuh"
#include "SimulationResult.cuh"

#define GPU_FUNCTION(func, ...) \
    func<<<1, 1>>>(__VA_ARGS__); \
    cudaDeviceSynchronize(); \
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

namespace
{
    class CudaInitializer
    {
    public:
        static void init() { static CudaInitializer instance; }

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

//    CudaMemoryManager::getInstance().reset();

    setSimulationParameters(settings.simulationParameters);
    setGpuConstants(gpuSettings);
    setFlowFieldSettings(settings.flowFieldSettings);
    _currentTimestep.store(timestep);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "initialize simulation");

    _cudaSimulationData = new SimulationData();
    _cudaSimulationResult = new SimulationResult();
    _cudaAccessTO = new DataAccessTO();
    _cudaMonitorData = new CudaMonitorData();

    _cudaSimulationData->init({settings.generalSettings.worldSizeX, settings.generalSettings.worldSizeY}, timestep);
    _cudaMonitorData->init();
    _cudaSimulationResult->init();

    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numTokens);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numStringBytes);
    CudaMemoryManager::getInstance().acquireMemory<char>(Const::MetadataMemorySize, _cudaAccessTO->stringBytes);

    resizeArrays({100000, 100000, 10000});
}

_CudaSimulation::~_CudaSimulation()
{
    _cudaSimulationData->free();
    _cudaMonitorData->free();
    _cudaSimulationResult->free();

    if (_cudaAccessTO->cells) {
        int numCells;
        checkCudaErrors(cudaMemcpy(&numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost));
        CudaMemoryManager::getInstance().freeMemory(numCells, _cudaAccessTO->cells);
    }
    if (_cudaAccessTO->particles) {
        int numParticles;
        checkCudaErrors(cudaMemcpy(&numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost));
        CudaMemoryManager::getInstance().freeMemory(numParticles, _cudaAccessTO->particles);
    }
    if (_cudaAccessTO->tokens) {
        int numTokens;
        checkCudaErrors(cudaMemcpy(&numTokens, _cudaAccessTO->numTokens, sizeof(int), cudaMemcpyDeviceToHost));
        CudaMemoryManager::getInstance().freeMemory(numTokens, _cudaAccessTO->tokens);
    }
    int numStringBytes;
    checkCudaErrors(cudaMemcpy(&numStringBytes, _cudaAccessTO->numStringBytes, sizeof(int), cudaMemcpyDeviceToHost));
    CudaMemoryManager::getInstance().freeMemory(numStringBytes, _cudaAccessTO->stringBytes);
    CudaMemoryManager::getInstance().freeMemory(1, _cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().freeMemory(1, _cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().freeMemory(1, _cudaAccessTO->numTokens);
    CudaMemoryManager::getInstance().freeMemory(1, _cudaAccessTO->numStringBytes);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "close simulation");

    delete _cudaAccessTO;
    delete _cudaSimulationData;
    delete _cudaMonitorData;
}

void* _CudaSimulation::registerImageResource(GLuint image)
{
    cudaGraphicsResource* cudaResource;

    CHECK_FOR_CUDA_ERROR(
        cudaGraphicsGLRegisterImage(&cudaResource, image, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

    return reinterpret_cast<void*>(cudaResource);
}

void _CudaSimulation::calcCudaTimestep()
{
    GPU_FUNCTION(calcSimulationTimestepKernel, *_cudaSimulationData, *_cudaSimulationResult);
    if (_currentTimestep % 10 == 0) {
        if (_cudaSimulationResult->isArrayResizeNeeded()) {
            resizeArrays({0, 0, 0});
        }
    }

    ++_cudaSimulationData->timestep;
    ++_currentTimestep;
}

void _CudaSimulation::getVectorImage(
    float2 const& rectUpperLeft,
    float2 const& rectLowerRight,
    void* cudaResource,
    int2 const& imageSize,
    double zoom)
{
    auto cudaResource_ = reinterpret_cast<cudaGraphicsResource*>(cudaResource);
    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource_));

    cudaArray* mappedArray;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&mappedArray, cudaResource_, 0, 0));

    if (imageSize.x * imageSize.y > _cudaSimulationData->numPixels) {
        _cudaSimulationData->resizeImage(imageSize);
    }
    GPU_FUNCTION(
        drawImageKernel,
        rectUpperLeft,
        rectLowerRight,
        imageSize,
        static_cast<float>(zoom),
        *_cudaSimulationData);

    cudaMemcpyToArray(
        mappedArray,
        0,
        0,
        _cudaSimulationData->imageData,
        sizeof(unsigned int) * imageSize.x * imageSize.y * 2,
        cudaMemcpyDeviceToDevice);

    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource_));
}

void _CudaSimulation::getSimulationData(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataAccessTO const& dataTO)
{
    GPU_FUNCTION(cudaGetSimulationAccessData, rectUpperLeft, rectLowerRight, *_cudaSimulationData, *_cudaAccessTO);

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

void _CudaSimulation::setSimulationData(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataAccessTO const& dataTO)
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

    GPU_FUNCTION(cudaSetSimulationAccessData, rectUpperLeft, rectLowerRight, *_cudaSimulationData, *_cudaAccessTO);
}

void _CudaSimulation::selectData(int2 const& pos)
{
    GPU_FUNCTION(cudaSelectData, pos, *_cudaSimulationData);
}

void _CudaSimulation::deselectData()
{
    GPU_FUNCTION(cudaDeselectData, *_cudaSimulationData);
}

void _CudaSimulation::applyForce(ApplyForceData const& applyData)
{
    CudaApplyForceData cudaApplyData{
        applyData.startPos, applyData.endPos, applyData.force, applyData.radius, applyData.onlyRotation};
    GPU_FUNCTION(cudaApplyForce, cudaApplyData, *_cudaSimulationData);
}

void _CudaSimulation::moveSelection(float2 const& displacement)
{
    GPU_FUNCTION(cudaMoveSelection, displacement, *_cudaSimulationData);
}

void _CudaSimulation::setGpuConstants(GpuSettings const& gpuConstants_)
{
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(gpuConstants, &gpuConstants_, sizeof(GpuSettings), 0, cudaMemcpyHostToDevice));
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
    GPU_FUNCTION(cudaGetCudaMonitorData, *_cudaSimulationData, *_cudaMonitorData);
    
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
    _cudaSimulationData->timestep = static_cast<int>(timestep);
    _currentTimestep.store(timestep);
}

void _CudaSimulation::setSimulationParameters(SimulationParameters const& parameters)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaSimulationParameters, &parameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void _CudaSimulation::setFlowFieldSettings(FlowFieldSettings const& settings)
{
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaFlowFieldSettings, &settings, sizeof(FlowFieldSettings), 0, cudaMemcpyHostToDevice));
}


void _CudaSimulation::clear()
{
    GPU_FUNCTION(cudaClearData, *_cudaSimulationData);
}

void _CudaSimulation::resizeArraysIfNecessary(ArraySizes const& additionals)
{
    if (_cudaSimulationData->shouldResize(
            additionals.cellArraySize, additionals.particleArraySize, additionals.tokenArraySize)) {
        resizeArrays(additionals);
    }
}

void _CudaSimulation::resizeArrays(ArraySizes const& additionals)
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "resize arrays");

    _cudaSimulationData->resizeTarget(
        additionals.cellArraySize, additionals.particleArraySize, additionals.tokenArraySize);
    if (!_cudaSimulationData->isEmpty()) {
        GPU_FUNCTION(copyEntitiesKernel, *_cudaSimulationData);
    }
    _cudaSimulationData->resizeSource();
    _cudaSimulationData->swap();

    if (_cudaAccessTO->cells) {
        int numCells;
        checkCudaErrors(cudaMemcpy(&numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost));
        CudaMemoryManager::getInstance().freeMemory(numCells, _cudaAccessTO->cells);
    }
    if (_cudaAccessTO->particles) {
        int numParticles;
        checkCudaErrors(cudaMemcpy(&numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost));
        CudaMemoryManager::getInstance().freeMemory(numParticles, _cudaAccessTO->particles);
    }
    if (_cudaAccessTO->tokens) {
        int numTokens;
        checkCudaErrors(cudaMemcpy(&numTokens, _cudaAccessTO->numTokens, sizeof(int), cudaMemcpyDeviceToHost));
        CudaMemoryManager::getInstance().freeMemory(numTokens, _cudaAccessTO->tokens);
    }

    auto cellArraySize = _cudaSimulationData->entities.cells.getSize_host();
    auto tokenArraySize = _cudaSimulationData->entities.tokens.getSize_host();
    CudaMemoryManager::getInstance().acquireMemory<CellAccessTO>(cellArraySize, _cudaAccessTO->cells);
    CudaMemoryManager::getInstance().acquireMemory<ParticleAccessTO>(cellArraySize, _cudaAccessTO->particles);
    CudaMemoryManager::getInstance().acquireMemory<TokenAccessTO>(tokenArraySize, _cudaAccessTO->tokens);

    loggingService->logMessage(Priority::Unimportant, "cell array size: " + std::to_string(cellArraySize));
    loggingService->logMessage(Priority::Unimportant, "particle array size: " + std::to_string(cellArraySize));
    loggingService->logMessage(Priority::Unimportant, "token array size: " + std::to_string(tokenArraySize));

    auto const memorySizeAfter = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();
    loggingService->logMessage(Priority::Important, std::to_string(memorySizeAfter / (1024 * 1024)) + " MB GPU memory acquired");
}
