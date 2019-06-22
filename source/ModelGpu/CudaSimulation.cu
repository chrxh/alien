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

#include "SimulationData.cuh"
#include "Map.cuh"


#define KERNEL_FUNCTION(numBlocks, numThreadsPerBlock, func, ...) func<<<numBlocks, numThreadsPerBlock>>>(##__VA_ARGS__); \
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
    _internalData->size = size;
    _internalData->clusterPointers = ArrayController<Cluster*>(MAX_CELLCLUSTERPOINTERS);
    _internalData->clusterPointersTemp = ArrayController<Cluster*>(MAX_CELLCLUSTERPOINTERS);
    _internalData->clusters = ArrayController<Cluster>(MAX_CELLCLUSTERS);
    _internalData->cellPointers = ArrayController<Cell*>(MAX_CELLPOINTERS);
    _internalData->cellPointersTemp = ArrayController<Cell*>(MAX_CELLPOINTERS);
    _internalData->cells = ArrayController<Cell>(MAX_CELLS);
    _internalData->cellsTemp = ArrayController<Cell>(MAX_CELLS);
    _internalData->tokenPointers = ArrayController<Token*>(MAX_TOKENPOINTERS);
    _internalData->tokenPointersTemp = ArrayController<Token*>(MAX_TOKENPOINTERS);
    _internalData->tokens = ArrayController<Token>(MAX_TOKENS);
    _internalData->tokensTemp = ArrayController<Token>(MAX_TOKENS);
    _internalData->particles = ArrayController<Particle>(MAX_PARTICLES);
    _internalData->particlesNew = ArrayController<Particle>(MAX_PARTICLES);
    checkCudaErrors(cudaMalloc(&_internalData->cellMap, size.x * size.y * sizeof(Cell*)));
    checkCudaErrors(cudaMalloc(&_internalData->particleMap, size.x * size.y * sizeof(Particle*)));

    std::vector<Cell*> hostCellMap(size.x * size.y, 0);
    std::vector<Particle*> hostParticleMap(size.x * size.y, 0);
    checkCudaErrors(cudaMemcpy(_internalData->cellMap, hostCellMap.data(), sizeof(Cell*)*size.x*size.y, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_internalData->particleMap, hostParticleMap.data(), sizeof(Cell*)*size.x*size.y, cudaMemcpyHostToDevice));
    _internalData->numberGen.init(RANDOM_NUMBER_BLOCK_SIZE);

    _cudaAccessTO = new DataAccessTO();
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->numClusters, sizeof(int)));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->numCells, sizeof(int)));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->numParticles, sizeof(int)));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->numTokens, sizeof(int)));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->clusters, sizeof(ClusterAccessTO)*MAX_CELLCLUSTERS));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->cells, sizeof(CellAccessTO)*MAX_CELLS));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->particles, sizeof(ParticleAccessTO)*MAX_PARTICLES));
    checkCudaErrors(cudaMalloc(&_cudaAccessTO->tokens, sizeof(TokenAccessTO)*MAX_TOKENS));
}

CudaSimulation::~CudaSimulation()
{
    _internalData->clusterPointers.free();
    _internalData->clusterPointersTemp.free();
    _internalData->clusters.free();
    _internalData->cellPointers.free();
    _internalData->cellPointersTemp.free();
    _internalData->cells.free();
    _internalData->cellsTemp.free();
    _internalData->tokenPointers.free();
    _internalData->tokenPointersTemp.free();
    _internalData->tokens.free();
    _internalData->tokensTemp.free();
    _internalData->particles.free();
    _internalData->particlesNew.free();

    checkCudaErrors(cudaFree(_internalData->cellMap));
    checkCudaErrors(cudaFree(_internalData->particleMap));
    _internalData->numberGen.free();

    checkCudaErrors(cudaFree(_cudaAccessTO->numClusters));
    checkCudaErrors(cudaFree(_cudaAccessTO->numCells));
    checkCudaErrors(cudaFree(_cudaAccessTO->numParticles));
    checkCudaErrors(cudaFree(_cudaAccessTO->clusters));
    checkCudaErrors(cudaFree(_cudaAccessTO->cells));
    checkCudaErrors(cudaFree(_cudaAccessTO->particles));
    checkCudaErrors(cudaFree(_cudaAccessTO->tokens));

    delete _cudaAccessTO;
    delete _internalData;

}

void CudaSimulation::calcNextTimestep()
{
    KERNEL_FUNCTION(1, 1, calcSimulationTimestep, *_internalData);
    swap(_internalData->particles, _internalData->particlesNew);
    swap(_internalData->clusterPointers, _internalData->clusterPointersTemp);
}

void CudaSimulation::getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO)
{
    KERNEL_FUNCTION(1, 1, getSimulationAccessData, rectUpperLeft, rectLowerRight, *_internalData, *_cudaAccessTO);

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

    KERNEL_FUNCTION(1, 1, setSimulationAccessData, rectUpperLeft, rectLowerRight, *_internalData, *_cudaAccessTO);
    swap(_internalData->particles, _internalData->particlesNew);
    swap(_internalData->clusterPointers, _internalData->clusterPointersTemp);
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
