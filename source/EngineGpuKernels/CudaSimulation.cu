#include <functional>
#include <iostream>
#include <list>

#if defined(_WIN32)
#include <windows.h>
#endif

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
#include "CudaConstants.h"
#include "CudaMemoryManager.cuh"
#include "CudaMonitorData.cuh"
#include "CudaSimulation.cuh"
#include "Entities.cuh"
#include "Map.cuh"
#include "MonitorKernels.cuh"
#include "PhysicalActionKernels.cuh"
#include "RenderingKernels.cuh"
#include "SimulationData.cuh"
#include "SimulationKernels.cuh"


#define GPU_FUNCTION(func, ...) \
    func<<<1, 1>>>(##__VA_ARGS__); \
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

            int result = 0;
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

CudaSimulation::CudaSimulation(
    int2 const& worldSize,
    int timestep,
    SimulationParameters const& parameters,
    CudaConstants const& cudaConstants)
{
    CudaInitializer::init();
    CudaMemoryManager::getInstance().reset();

    setSimulationParameters(parameters);
    setCudaConstants(cudaConstants);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "acquire GPU memory");

    _cudaSimulationData = new SimulationData();
    _cudaAccessTO = new DataAccessTO();
    _cudaMonitorData = new CudaMonitorData();

    auto const memorySizeBefore = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();

    _cudaSimulationData->init(worldSize, cudaConstants, timestep);
    _cudaMonitorData->init();

    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numClusters);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numTokens);
    CudaMemoryManager::getInstance().acquireMemory<int>(1, _cudaAccessTO->numStringBytes);
    CudaMemoryManager::getInstance().acquireMemory<ClusterAccessTO>(
        cudaConstants.MAX_CLUSTERS, _cudaAccessTO->clusters);
    CudaMemoryManager::getInstance().acquireMemory<CellAccessTO>(cudaConstants.MAX_CELLS, _cudaAccessTO->cells);
    CudaMemoryManager::getInstance().acquireMemory<ParticleAccessTO>(
        cudaConstants.MAX_PARTICLES, _cudaAccessTO->particles);
    CudaMemoryManager::getInstance().acquireMemory<TokenAccessTO>(cudaConstants.MAX_TOKENS, _cudaAccessTO->tokens);
    CudaMemoryManager::getInstance().acquireMemory<char>(
        cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE, _cudaAccessTO->stringBytes);

    auto const memorySizeAfter = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();

    std::stringstream stream;
    stream << (memorySizeAfter - memorySizeBefore) / (1024 * 1024) << " MB GPU memory acquired";

    loggingService->logMessage(Priority::Important, stream.str());
}

CudaSimulation::~CudaSimulation()
{
    _cudaSimulationData->free();
    _cudaMonitorData->free();

    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numClusters);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numCells);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numParticles);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numTokens);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->numStringBytes);
    CudaMemoryManager::getInstance().freeMemory(_cudaAccessTO->clusters);
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

void* CudaSimulation::registerImageResource(GLuint image)
{
    cudaGraphicsResource* resource;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsGLRegisterImage(&resource, image, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));

    return reinterpret_cast<void*>(resource);
}

void CudaSimulation::calcCudaTimestep()
{
    GPU_FUNCTION(cudaCalcSimulationTimestep, *_cudaSimulationData);
    ++_cudaSimulationData->timestep;
}

void CudaSimulation::DEBUG_printNumEntries()
{
    std::stringstream stream;
    stream << "Particles: " << _cudaSimulationData->entities.particles.retrieveNumEntries() << "; "
           << "Cells: " << _cudaSimulationData->entities.cells.retrieveNumEntries() << "; "
           << "Clusters: " << _cudaSimulationData->entities.clusters.retrieveNumEntries() << "; "
           << "CellPointers: " << _cudaSimulationData->entities.cellPointers.retrieveNumEntries() << "; "
           << "Tokens: " << _cudaSimulationData->entities.tokens.retrieveNumEntries() << "; "
           << "TokenPointers: " << _cudaSimulationData->entities.tokenPointers.retrieveNumEntries() << "; ";

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, stream.str());
}

void CudaSimulation::getPixelImage(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    int2 const& imageSize,
    unsigned char* imageData)
{
    if (imageSize.x * imageSize.y > _cudaSimulationData->numImageBytes) {
        _cudaSimulationData->resizeImage(imageSize);
    }

/*
    GPU_FUNCTION(cudaDrawImage_pixelStyle, rectUpperLeft, rectLowerRight, imageSize, * _cudaSimulationData);
*/
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        imageData,
        _cudaSimulationData->imageData,
        sizeof(unsigned int) * imageSize.x * imageSize.y,
        cudaMemcpyDeviceToHost));
}

void CudaSimulation::getVectorImage(
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

    if (imageSize.x * imageSize.y > _cudaSimulationData->numImageBytes) {
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
        sizeof(unsigned int) * imageSize.x * imageSize.y,
        cudaMemcpyDeviceToDevice);

    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &cudaResource));
}

void CudaSimulation::getSimulationData(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataAccessTO const& dataTO)
{
    GPU_FUNCTION(cudaGetSimulationAccessData, rectUpperLeft, rectLowerRight, *_cudaSimulationData, *_cudaAccessTO);

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(dataTO.numClusters, _cudaAccessTO->numClusters, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(dataTO.numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numTokens, _cudaAccessTO->numTokens, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(dataTO.numStringBytes, _cudaAccessTO->numStringBytes, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        dataTO.clusters,
        _cudaAccessTO->clusters,
        sizeof(ClusterAccessTO) * (*dataTO.numClusters),
        cudaMemcpyDeviceToHost));
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

void CudaSimulation::setSimulationData(
    int2 const& rectUpperLeft,
    int2 const& rectLowerRight,
    DataAccessTO const& dataTO)
{
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(_cudaAccessTO->numClusters, dataTO.numClusters, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->numCells, dataTO.numCells, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(_cudaAccessTO->numParticles, dataTO.numParticles, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->numTokens, dataTO.numTokens, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpy(_cudaAccessTO->numStringBytes, dataTO.numStringBytes, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        _cudaAccessTO->clusters,
        dataTO.clusters,
        sizeof(ClusterAccessTO) * (*dataTO.numClusters),
        cudaMemcpyHostToDevice));
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

void CudaSimulation::selectData(int2 const& pos)
{
    GPU_FUNCTION(cudaSelectData, pos, *_cudaSimulationData);
}

void CudaSimulation::deselectData()
{
    GPU_FUNCTION(cudaDeselectData, *_cudaSimulationData);
}

void CudaSimulation::applyForce(ApplyForceData const& applyData)
{
    CudaApplyForceData cudaApplyData{applyData.startPos, applyData.endPos, applyData.force, applyData.onlyRotation};
    GPU_FUNCTION(cudaApplyForce, cudaApplyData, *_cudaSimulationData);
}

void CudaSimulation::moveSelection(float2 const& displacement)
{
    GPU_FUNCTION(cudaMoveSelection, displacement, *_cudaSimulationData);
}

CudaConstants CudaSimulation::getCudaConstants() const
{
    return _cudaConstants;
}

MonitorData CudaSimulation::getMonitorData()
{
    GPU_FUNCTION(cudaGetCudaMonitorData, *_cudaSimulationData, *_cudaMonitorData);
    return _cudaMonitorData->getMonitorData(getTimestep());
}

int CudaSimulation::getTimestep() const
{
    return _cudaSimulationData->timestep;
}

void CudaSimulation::setTimestep(int timestep)
{
    _cudaSimulationData->timestep = timestep;
}

void CudaSimulation::setSimulationParameters(SimulationParameters const& parameters)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaSimulationParameters, &parameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void CudaSimulation::setExecutionParameters(ExecutionParameters const& parameters)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaExecutionParameters, &parameters, sizeof(ExecutionParameters), 0, cudaMemcpyHostToDevice));
}

void CudaSimulation::clear()
{
    GPU_FUNCTION(cudaClearData, *_cudaSimulationData);
}

namespace
{
    void calcImageBlurFactors(int* imageBlurFactors)
    {
        imageBlurFactors[0] = 200;
        imageBlurFactors[1] = 20;
        imageBlurFactors[2] = 7;
        imageBlurFactors[3] = 7;
        imageBlurFactors[4] = 7;
        imageBlurFactors[5] = 7;

        int sum = 0;
        int2 relPos;
        for (relPos.x = -5; relPos.x <= 5; ++relPos.x) {
            for (relPos.y = -5; relPos.y <= 5; ++relPos.y) {
                auto r = Math::length(toFloat2(relPos));
                if (r <= 5 + FP_PRECISION) {
                    sum += imageBlurFactors[floorInt(r)];
                }
            }
        }
        imageBlurFactors[6] = sum - 400;
    }
}

void CudaSimulation::setCudaConstants(CudaConstants const& cudaConstants_)
{
    _cudaConstants = cudaConstants_;

    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaConstants, &cudaConstants_, sizeof(CudaConstants), 0, cudaMemcpyHostToDevice));

    int imageBlurFactors[7];
    calcImageBlurFactors(imageBlurFactors);
    CHECK_FOR_CUDA_ERROR(
        cudaMemcpyToSymbol(cudaImageBlurFactors, &imageBlurFactors, sizeof(int) * 7, 0, cudaMemcpyHostToDevice));
}
