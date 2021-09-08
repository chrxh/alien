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

#include "AccessKernels.cuh"
#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Base/LoggingService.h"
#include "Base/ServiceLocator.h"
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
                stream << "device " << deviceNumber << ": " << prop.name << " has compute capability " << prop.major
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

_CudaSimulation::_CudaSimulation(
    int2 const& worldSize,
    int timestep,
    SimulationParameters const& parameters,
    GpuConstants const& gpuConstants)
{
    CudaMemoryManager::getInstance().reset();

    setSimulationParameters(parameters);
    setGpuConstants(gpuConstants);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "acquire GPU memory");

    _cudaSimulationData = new SimulationData();
    _cudaAccessTO = new DataAccessTO();
    _cudaMonitorData = new CudaMonitorData();

    auto const memorySizeBefore = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();

    _cudaSimulationData->init(worldSize, gpuConstants, timestep);
    _cudaMonitorData->init();

    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numTokens);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numStringBytes);
    CudaMemoryManager::getInstance().acquireMemory<CellAccessTO>(gpuConstants.MAX_CELLS, _cudaAccessTO->cells);
    CudaMemoryManager::getInstance().acquireMemory<ParticleAccessTO>(
        gpuConstants.MAX_PARTICLES, _cudaAccessTO->particles);
    CudaMemoryManager::getInstance().acquireMemory<TokenAccessTO>(gpuConstants.MAX_TOKENS, _cudaAccessTO->tokens);
    CudaMemoryManager::getInstance().acquireMemory<char>(
        gpuConstants.METADATA_DYNAMIC_MEMORY_SIZE, _cudaAccessTO->stringBytes);

    auto const memorySizeAfter = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();

    std::stringstream stream;
    stream << (memorySizeAfter - memorySizeBefore) / (1024 * 1024) << " MB GPU memory acquired";

    loggingService->logMessage(Priority::Important, stream.str());
}

_CudaSimulation::~_CudaSimulation()
{
    _cudaSimulationData->free();
    _cudaMonitorData->free();

    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numTokens);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numStringBytes);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->cells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->particles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->tokens);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->stringBytes);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "GPU memory released");

    delete _cudaAccessTO;
    delete _cudaSimulationData;
    delete _cudaMonitorData;
}

void* _CudaSimulation::registerImageResource(GLuint image)
{
    cudaGraphicsResource* resource;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsGLRegisterImage(&resource, image, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

    return reinterpret_cast<void*>(resource);
}

void _CudaSimulation::calcCudaTimestep()
{
    GPU_FUNCTION(cudaCalcSimulationTimestep, *_cudaSimulationData);
    ++_cudaSimulationData->timestep;
}

void _CudaSimulation::DEBUG_printNumEntries()
{
    std::stringstream stream;
    stream << "Particles: " << _cudaSimulationData->entities.particles.retrieveNumEntries() << "; "
           << "Cells: " << _cudaSimulationData->entities.cells.retrieveNumEntries() << "; "
           << "CellPointers: " << _cudaSimulationData->entities.cellPointers.retrieveNumEntries() << "; "
           << "Tokens: " << _cudaSimulationData->entities.tokens.retrieveNumEntries() << "; "
           << "TokenPointers: " << _cudaSimulationData->entities.tokenPointers.retrieveNumEntries() << "; ";

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, stream.str());
}

void _CudaSimulation::getVectorImage(
    float2 const& rectUpperLeft,
    float2 const& rectLowerRight,
    void* const& resource,
    int2 const& imageSize,
    double zoom)
{
    auto cudaResource = reinterpret_cast<cudaGraphicsResource*>(resource);
    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &cudaResource));

    cudaArray* mappedArray;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&mappedArray, cudaResource, 0, 0));

    if (imageSize.x * imageSize.y > _cudaSimulationData->numPixels) {
        _cudaSimulationData->resizeImage(imageSize);
    }
    GPU_FUNCTION(
        drawImage,
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

    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));
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
    CudaApplyForceData cudaApplyData{applyData.startPos, applyData.endPos, applyData.force, applyData.onlyRotation};
    GPU_FUNCTION(cudaApplyForce, cudaApplyData, *_cudaSimulationData);
}

void _CudaSimulation::moveSelection(float2 const& displacement)
{
    GPU_FUNCTION(cudaMoveSelection, displacement, *_cudaSimulationData);
}

GpuConstants _CudaSimulation::getGpuConstants() const
{
    return _gpuConstants;
}

MonitorData _CudaSimulation::getMonitorData()
{
    GPU_FUNCTION(cudaGetCudaMonitorData, *_cudaSimulationData, *_cudaMonitorData);
    return _cudaMonitorData->getMonitorData(getTimestep());
}

int _CudaSimulation::getTimestep() const
{
    return _cudaSimulationData->timestep;
}

void _CudaSimulation::setTimestep(int timestep)
{
    _cudaSimulationData->timestep = timestep;
}

void _CudaSimulation::setSimulationParameters(SimulationParameters const& parameters)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaSimulationParameters, &parameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void _CudaSimulation::clear()
{
    GPU_FUNCTION(cudaClearData, *_cudaSimulationData);
}

void _CudaSimulation::setGpuConstants(GpuConstants const& gpuConstants_)
{
    _gpuConstants = gpuConstants_;

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(gpuConstants, &gpuConstants_, sizeof(GpuConstants), 0, cudaMemcpyHostToDevice));
}
