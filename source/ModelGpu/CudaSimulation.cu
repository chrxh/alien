#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <list>
#include <iostream>
#include <functional>

#include "ModelBasic/SimulationParameters.h"
#include "Base.cuh"
#include "CudaSimulation.cuh"
#include "CudaConstants.h"
#include "ConstantMemory.cuh"
#include "AccessTOs.cuh"
#include "SimulationKernels.cuh"
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


#define GPU_FUNCTION(func, ...) func<<<1, 1>>>(##__VA_ARGS__); \
    cudaDeviceSynchronize(); \
    checkCudaErrors(cudaGetLastError());

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
            cudaSetDevice(0);
            std::cout << "[CUDA] initialized" << std::endl;
        }

        ~CudaInitializer()
        {
            cudaDeviceReset();
            std::cout << "[CUDA] closed" << std::endl;
        }
    };
}

CudaSimulation::CudaSimulation(
    int2 const& size,
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

    _cudaSimulationData->init(size, cudaConstants, timestep);
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

    std::cout << "[CUDA] " << (memorySizeAfter - memorySizeBefore) / (1024 * 1024) << "mb memory acquired" << std::endl;
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

    std::cout << "[CUDA] memory released" << std::endl;

    delete _cudaAccessTO;
    delete _cudaSimulationData;
    delete _cudaMonitorData;

}

void CudaSimulation::calcCudaTimestep()
{
    GPU_FUNCTION(calcSimulationTimestep, *_cudaSimulationData);
    ++_cudaSimulationData->timestep;
}

void CudaSimulation::DEBUG_printNumEntries()
{
    std::cout
        << "Particles: " << _cudaSimulationData->entities.particles.retrieveNumEntries() << "; "
        << "Cells: " << _cudaSimulationData->entities.cells.retrieveNumEntries() << "; "
        << "Clusters: " << _cudaSimulationData->entities.clusters.retrieveNumEntries() << "; "
        << "CellPointers: " << _cudaSimulationData->entities.cellPointers.retrieveNumEntries() << "; "
        << "Tokens: " << _cudaSimulationData->entities.tokens.retrieveNumEntries() << "; "
        << "TokenPointers: " << _cudaSimulationData->entities.tokenPointers.retrieveNumEntries() << "; "
        << std::endl;
}

void CudaSimulation::getSimulationImage(int2 const & rectUpperLeft, int2 const & rectLowerRight, unsigned char* imageData)
{
    int width = rectLowerRight.x - rectUpperLeft.x + 1;
    int height = rectLowerRight.y - rectUpperLeft.y + 1;
    int numPixels = width * height;

    GPU_FUNCTION(drawImage, rectUpperLeft, rectLowerRight, *_cudaSimulationData);
    checkCudaErrors(cudaMemcpy(
        imageData, _cudaSimulationData->finalImageData, sizeof(unsigned int) * numPixels, cudaMemcpyDeviceToHost));
}

void CudaSimulation::getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO)
{
    GPU_FUNCTION(getSimulationAccessData, rectUpperLeft, rectLowerRight, *_cudaSimulationData, *_cudaAccessTO);

    checkCudaErrors(cudaMemcpy(dataTO.numClusters, _cudaAccessTO->numClusters, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.numTokens, _cudaAccessTO->numTokens, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.numStringBytes, _cudaAccessTO->numStringBytes, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.clusters, _cudaAccessTO->clusters, sizeof(ClusterAccessTO) * (*dataTO.numClusters), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.cells, _cudaAccessTO->cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.particles, _cudaAccessTO->particles, sizeof(ParticleAccessTO) * (*dataTO.numParticles), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.tokens, _cudaAccessTO->tokens, sizeof(TokenAccessTO) * (*dataTO.numTokens), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.stringBytes, _cudaAccessTO->stringBytes, sizeof(char) * (*dataTO.numStringBytes), cudaMemcpyDeviceToHost));
}

void CudaSimulation::setSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO)
{
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->numClusters, dataTO.numClusters, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->numCells, dataTO.numCells, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->numParticles, dataTO.numParticles, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->numTokens, dataTO.numTokens, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->numStringBytes, dataTO.numStringBytes, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->clusters, dataTO.clusters, sizeof(ClusterAccessTO) * (*dataTO.numClusters), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->cells, dataTO.cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->particles, dataTO.particles, sizeof(ParticleAccessTO) * (*dataTO.numParticles), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->tokens, dataTO.tokens, sizeof(TokenAccessTO) * (*dataTO.numTokens), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->stringBytes, dataTO.stringBytes, sizeof(char) * (*dataTO.numStringBytes), cudaMemcpyHostToDevice));

    GPU_FUNCTION(setSimulationAccessData, rectUpperLeft, rectLowerRight, *_cudaSimulationData, *_cudaAccessTO);
}

void CudaSimulation::applyForce(ApplyForceData const& applyData)
{
    CudaApplyForceData cudaApplyData{ applyData.startPos, applyData.endPos, applyData.force, applyData.onlyRotation };
    GPU_FUNCTION(cudaApplyForce, cudaApplyData, *_cudaSimulationData);
}

MonitorData CudaSimulation::getMonitorData()
{
    GPU_FUNCTION(getCudaMonitorData, *_cudaSimulationData, *_cudaMonitorData);
    return _cudaMonitorData->getMonitorData();
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
    checkCudaErrors(cudaMemcpyToSymbol(
        cudaSimulationParameters, &parameters, sizeof(SimulationParameters), 0, cudaMemcpyHostToDevice));
}

void CudaSimulation::setExecutionParameters(ExecutionParameters const & parameters)
{
    checkCudaErrors(cudaMemcpyToSymbol(
        cudaExecutionParameters,
        &parameters,
        sizeof(ExecutionParameters),
        0,
        cudaMemcpyHostToDevice));
}

void CudaSimulation::clear()
{
    GPU_FUNCTION(clearData, *_cudaSimulationData);
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
    checkCudaErrors(cudaMemcpyToSymbol(cudaConstants, &cudaConstants_, sizeof(CudaConstants), 0, cudaMemcpyHostToDevice));

    int imageBlurFactors[7];
    calcImageBlurFactors(imageBlurFactors);
    checkCudaErrors(cudaMemcpyToSymbol(cudaImageBlurFactors, &imageBlurFactors, sizeof(int) * 7, 0, cudaMemcpyHostToDevice));
}
