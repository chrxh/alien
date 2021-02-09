#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <list>
#include <iostream>
#include <functional>

#include "EngineInterface/SimulationParameters.h"
#include "Base.cuh"

#include "CudaSimulation.cuh"
#include "CudaConstants.h"
#include "ConstantMemory.cuh"
#include "AccessTOs.cuh"
#include "AccessKernels.cuh"
#include "CleanupKernels.cuh"
#include "MonitorKernels.cuh"
#include "RenderingKernels.cuh"
#include "PhysicalActionKernels.cuh"
#include "Entities.cuh"
#include "CudaMemoryManager.cuh"
#include "CudaMonitorData.cuh"

#include "SimulationData.cuh"
#include "Map.cuh"
#include "SimulationKernels.cuh"



#define GPU_FUNCTION(func, ...) func<<<1, 1>>>(##__VA_ARGS__); \
    cudaDeviceSynchronize(); \
    CHECK_FOR_CUDA_ERROR(cudaGetLastError());

namespace
{
    class CudaInitializer
    {
    public:
        static void init()
        {
            static CudaInitializer instance;
        }

        CudaInitializer()
        {
            std::cerr << "[CUDA] start initialization" << std::endl;
            auto result = cudaSetDevice(0);
            if (result != cudaSuccess) {
                throw std::exception("CUDA could not be initialized.");
            }
            std::cerr << "[CUDA] initialization finished" << std::endl;
        }

        ~CudaInitializer()
        {
            cudaDeviceReset();
            std::cerr << "[CUDA] closed" << std::endl;
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
    CudaMemoryManager::getInstance().acquireMemory<ClusterAccessTO>(cudaConstants.MAX_CLUSTERS, _cudaAccessTO->clusters);
    CudaMemoryManager::getInstance().acquireMemory<CellAccessTO>(cudaConstants.MAX_CELLS, _cudaAccessTO->cells);
    CudaMemoryManager::getInstance().acquireMemory<ParticleAccessTO>(cudaConstants.MAX_PARTICLES, _cudaAccessTO->particles);
    CudaMemoryManager::getInstance().acquireMemory<TokenAccessTO>(cudaConstants.MAX_TOKENS, _cudaAccessTO->tokens);
    CudaMemoryManager::getInstance().acquireMemory<char>(cudaConstants.METADATA_DYNAMIC_MEMORY_SIZE, _cudaAccessTO->stringBytes);

    auto const memorySizeAfter = CudaMemoryManager::getInstance().getSizeOfAcquiredMemory();

    std::cerr << "[CUDA] " << (memorySizeAfter - memorySizeBefore) / (1024 * 1024) << "mb memory acquired" << std::endl;
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

    std::cerr << "[CUDA] memory released" << std::endl;

    delete _cudaAccessTO;
    delete _cudaSimulationData;
    delete _cudaMonitorData;

}

void CudaSimulation::calcCudaTimestep()
{
    GPU_FUNCTION(cudaCalcSimulationTimestep, *_cudaSimulationData);
    ++_cudaSimulationData->timestep;
}

void CudaSimulation::DEBUG_printNumEntries()
{
    std::cerr
        << "Particles: " << _cudaSimulationData->entities.particles.retrieveNumEntries() << "; "
        << "Cells: " << _cudaSimulationData->entities.cells.retrieveNumEntries() << "; "
        << "Clusters: " << _cudaSimulationData->entities.clusters.retrieveNumEntries() << "; "
        << "CellPointers: " << _cudaSimulationData->entities.cellPointers.retrieveNumEntries() << "; "
        << "Tokens: " << _cudaSimulationData->entities.tokens.retrieveNumEntries() << "; "
        << "TokenPointers: " << _cudaSimulationData->entities.tokenPointers.retrieveNumEntries() << "; "
        << std::endl;
}

void CudaSimulation::getPixelImage(int2 const & rectUpperLeft, int2 const & rectLowerRight, unsigned char* imageData)
{
    int width = rectLowerRight.x - rectUpperLeft.x + 1;
    int height = rectLowerRight.y - rectUpperLeft.y + 1;

    GPU_FUNCTION(cudaDrawImage_pixelStyle, rectUpperLeft, rectLowerRight, *_cudaSimulationData);
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        imageData, _cudaSimulationData->finalImageData, sizeof(unsigned int) * width * height, cudaMemcpyDeviceToHost));
}

void CudaSimulation::getVectorImage(int2 const & rectUpperLeft, int2 const & rectLowerRight, int2 const& imageSize, 
    double zoom, unsigned char * imageData)
{
    if (imageSize.x * imageSize.y > _cudaSimulationData->numImageBytes) {
        _cudaSimulationData->resizeImage(imageSize);
    }
    GPU_FUNCTION(drawImage_vectorStyle, rectUpperLeft, rectLowerRight, imageSize, static_cast<float>(zoom), *_cudaSimulationData);

    CHECK_FOR_CUDA_ERROR(cudaMemcpy(
        imageData, _cudaSimulationData->finalImageData, sizeof(unsigned int) * imageSize.x * imageSize.y, cudaMemcpyDeviceToHost));
}

void CudaSimulation::getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO)
{
    GPU_FUNCTION(cudaGetSimulationAccessData, rectUpperLeft, rectLowerRight, *_cudaSimulationData, *_cudaAccessTO);

    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numClusters, _cudaAccessTO->numClusters, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numTokens, _cudaAccessTO->numTokens, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.numStringBytes, _cudaAccessTO->numStringBytes, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.clusters, _cudaAccessTO->clusters, sizeof(ClusterAccessTO) * (*dataTO.numClusters), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.cells, _cudaAccessTO->cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.particles, _cudaAccessTO->particles, sizeof(ParticleAccessTO) * (*dataTO.numParticles), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.tokens, _cudaAccessTO->tokens, sizeof(TokenAccessTO) * (*dataTO.numTokens), cudaMemcpyDeviceToHost));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(dataTO.stringBytes, _cudaAccessTO->stringBytes, sizeof(char) * (*dataTO.numStringBytes), cudaMemcpyDeviceToHost));
}

void CudaSimulation::setSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->numClusters, dataTO.numClusters, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->numCells, dataTO.numCells, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->numParticles, dataTO.numParticles, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->numTokens, dataTO.numTokens, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->numStringBytes, dataTO.numStringBytes, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->clusters, dataTO.clusters, sizeof(ClusterAccessTO) * (*dataTO.numClusters), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->cells, dataTO.cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->particles, dataTO.particles, sizeof(ParticleAccessTO) * (*dataTO.numParticles), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->tokens, dataTO.tokens, sizeof(TokenAccessTO) * (*dataTO.numTokens), cudaMemcpyHostToDevice));
    CHECK_FOR_CUDA_ERROR(cudaMemcpy(_cudaAccessTO->stringBytes, dataTO.stringBytes, sizeof(char) * (*dataTO.numStringBytes), cudaMemcpyHostToDevice));

    GPU_FUNCTION(cudaSetSimulationAccessData, rectUpperLeft, rectLowerRight, *_cudaSimulationData, *_cudaAccessTO);
}

void CudaSimulation::selectData(int2 const & pos)
{
    GPU_FUNCTION(cudaSelectData, pos, *_cudaSimulationData);
}

void CudaSimulation::deselectData()
{
    GPU_FUNCTION(cudaDeselectData, *_cudaSimulationData);
}

void CudaSimulation::applyForce(ApplyForceData const& applyData)
{
    CudaApplyForceData cudaApplyData{ applyData.startPos, applyData.endPos, applyData.force, applyData.onlyRotation };
    GPU_FUNCTION(cudaApplyForce, cudaApplyData, *_cudaSimulationData);
}

void CudaSimulation::moveSelection(float2 const & displacement)
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

void CudaSimulation::setSimulationParameters(SimulationParameters const & parameters)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaSimulationParameters, &parameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void CudaSimulation::setExecutionParameters(ExecutionParameters const & parameters)
{
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(
        cudaExecutionParameters,
        &parameters,
        sizeof(ExecutionParameters),
        0,
        cudaMemcpyHostToDevice));
}

void CudaSimulation::clear()
{
    GPU_FUNCTION(cudaClearData, *_cudaSimulationData);
}

namespace
{
    void calcImageBlurFactors(int* imageBlurFactors)
    {
        imageBlurFactors[0] = 300;
        imageBlurFactors[1] = 40;
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

void CudaSimulation::setCudaConstants(CudaConstants const & cudaConstants_)
{
    _cudaConstants = cudaConstants_;

    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(cudaConstants, &cudaConstants_, sizeof(CudaConstants), 0, cudaMemcpyHostToDevice));

    int imageBlurFactors[7];
    calcImageBlurFactors(imageBlurFactors);
    CHECK_FOR_CUDA_ERROR(cudaMemcpyToSymbol(cudaImageBlurFactors, &imageBlurFactors, sizeof(int) * 7, 0, cudaMemcpyHostToDevice));
}
