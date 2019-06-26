#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <list>
#include <iostream>
#include <functional>

#include "ModelBasic/SimulationParameters.h"
#include "Base.cuh"
#include "CudaSimulation.cuh"
#include "CudaConstants.cuh"
#include "CudaSimulationParameters.cuh"
#include "CudaAccessTOs.cuh"
#include "SimulationKernels.cuh"
#include "AccessKernels.cuh"
#include "CleanupKernels.cuh"
#include "Entities.cuh"

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

CudaSimulation::CudaSimulation(int2 const &size, SimulationParameters const& parameters)
{

    CudaInitializer::init();

    setSimulationParameters(parameters);

    _internalData = new SimulationData();
    _internalData->init(size);

    _cudaAccessTO = new DataAccessTO();
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->numClusters, sizeof(int)));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->numCells, sizeof(int)));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->numParticles, sizeof(int)));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->numTokens, sizeof(int)));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->clusters, sizeof(ClusterAccessTO)*MAX_CLUSTERS));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->cells, sizeof(CellAccessTO)*MAX_CELLS));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->particles, sizeof(ParticleAccessTO)*MAX_PARTICLES));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->tokens, sizeof(TokenAccessTO)*MAX_TOKENS));

    std::cout << "[CUDA] memory acquired" << std::endl;
}

CudaSimulation::~CudaSimulation()
{
    _internalData->free();

    checkCudaErrors(cudaFree(_cudaAccessTO->numClusters));
    checkCudaErrors(cudaFree(_cudaAccessTO->numCells));
    checkCudaErrors(cudaFree(_cudaAccessTO->numParticles));
    checkCudaErrors(cudaFree(_cudaAccessTO->clusters));
    checkCudaErrors(cudaFree(_cudaAccessTO->cells));
    checkCudaErrors(cudaFree(_cudaAccessTO->particles));
    checkCudaErrors(cudaFree(_cudaAccessTO->tokens));

    std::cout << "[CUDA] freed" << std::endl;

    delete _cudaAccessTO;
    delete _internalData;

}

void CudaSimulation::calcNextTimestep()
{
    GPU_FUNCTION(calcSimulationTimestep, *_internalData);
/*
    std::cout
        << "Clusters: " << _internalData->entities.clusterPointers.retrieveNumEntries() << "; " << _internalData->entities.clusters.retrieveNumEntries() << "  "
        << "Cells: " << _internalData->entities.cellPointers.retrieveNumEntries() << "; " << _internalData->entities.cells.retrieveNumEntries()
        << std::endl;
*/
}

void CudaSimulation::getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO)
{
    GPU_FUNCTION(getSimulationAccessData, rectUpperLeft, rectLowerRight, *_internalData, *_cudaAccessTO);

    checkCudaErrors(cudaMemcpy(dataTO.numClusters, _cudaAccessTO->numClusters, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.numTokens, _cudaAccessTO->numTokens, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.clusters, _cudaAccessTO->clusters, sizeof(ClusterAccessTO) * (*dataTO.numClusters), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.cells, _cudaAccessTO->cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.particles, _cudaAccessTO->particles, sizeof(ParticleAccessTO) * (*dataTO.numParticles), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(dataTO.tokens, _cudaAccessTO->tokens, sizeof(TokenAccessTO) * (*dataTO.numTokens), cudaMemcpyDeviceToHost));
}

void CudaSimulation::setSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO)
{
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->numClusters, dataTO.numClusters, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->numCells, dataTO.numCells, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->numParticles, dataTO.numParticles, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->numTokens, dataTO.numTokens, sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->clusters, dataTO.clusters, sizeof(ClusterAccessTO) * (*dataTO.numClusters), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->cells, dataTO.cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->particles, dataTO.particles, sizeof(ParticleAccessTO) * (*dataTO.numParticles), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_cudaAccessTO->tokens, dataTO.tokens, sizeof(TokenAccessTO) * (*dataTO.numTokens), cudaMemcpyHostToDevice));

    GPU_FUNCTION(setSimulationAccessData, rectUpperLeft, rectLowerRight, *_internalData, *_cudaAccessTO);
}

void CudaSimulation::setSimulationParameters(SimulationParameters const & parameters)
{
    CudaSimulationParameters parametersToCopy;
    parametersToCopy.cellMaxDistance = parameters.cellMaxDistance;
    parametersToCopy.cellMinDistance = parameters.cellMinDistance;
    parametersToCopy.cellMinEnergy = parameters.cellMinEnergy;
    parametersToCopy.cellFusionVelocity = parameters.cellFusionVelocity;
    parametersToCopy.cellMaxForce = parameters.cellMaxForce;
    parametersToCopy.cellMaxForceDecayProb = parameters.cellMaxForceDecayProb;
    parametersToCopy.cellTransformationProb = parameters.cellTransformationProb;
    parametersToCopy.cellMass = 1.0f / parameters.cellMass_Reciprocal;
    parametersToCopy.cellMaxToken = parameters.cellMaxToken;
    parametersToCopy.cellMaxTokenBranchNumber = parameters.cellMaxTokenBranchNumber;
    parametersToCopy.cellFunctionComputerMaxInstructions = parameters.cellFunctionComputerMaxInstructions;
    parametersToCopy.cellFunctionComputerCellMemorySize = parameters.cellFunctionComputerCellMemorySize;
    parametersToCopy.tokenMinEnergy = parameters.tokenMinEnergy;
    parametersToCopy.tokenMemorySize = parameters.tokenMemorySize;
    parametersToCopy.radiationProbability = parameters.radiationProb;
    parametersToCopy.radiationExponent = parameters.radiationExponent;
    parametersToCopy.radiationFactor = parameters.radiationFactor;
    parametersToCopy.radiationVelocityMultiplier = parameters.radiationVelocityMultiplier;
    parametersToCopy.radiationVelocityPerturbation = parameters.radiationVelocityPerturbation;

    checkCudaErrors(cudaMemcpyToSymbol(cudaSimulationParameters, &parametersToCopy, sizeof(CudaSimulationParameters), 0, cudaMemcpyHostToDevice));
}
