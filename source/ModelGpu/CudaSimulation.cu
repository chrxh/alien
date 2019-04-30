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
#include "CudaInterface.cuh"
#include "SimulationFunctions.cuh"
#include "AccessFunctions.cuh"

#include "SimulationData.cuh"
#include "Map.cuh"

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

	cudaStreamCreate(&_cudaStream);
	std::cout << "[CUDA] stream created" << std::endl;

	_internalData = new SimulationData();
	_internalData->size = size;
	_internalData->clustersAC1 = ArrayController<Cluster>(MAX_CELLCLUSTERS);
	_internalData->clustersAC2 = ArrayController<Cluster>(MAX_CELLCLUSTERS);
	_internalData->cellsAC1 = ArrayController<Cell>(MAX_CELLS);
	_internalData->cellsAC2 = ArrayController<Cell>(MAX_CELLS);
	_internalData->particlesAC1 = ArrayController<Particle>(MAX_PARTICLES);
	_internalData->particlesAC2 = ArrayController<Particle>(MAX_PARTICLES);
	_internalData->tokensAC1 = ArrayController<Token>(MAX_TOKENS);
	_internalData->tokensAC2 = ArrayController<Token>(MAX_TOKENS);
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
	_internalData->clustersAC1.free();
	_internalData->clustersAC2.free();
	_internalData->cellsAC1.free();
	_internalData->cellsAC2.free();
	_internalData->particlesAC1.free();
	_internalData->particlesAC2.free();
	_internalData->tokensAC1.free();
	_internalData->tokensAC2.free();

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

	std::cout << "[CUDA] stream closed" << std::endl;
}

void CudaSimulation::calcNextTimestep()
{
	prepareTargetData();

	clusterDynamicsStep1<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_internalData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clusterDynamicsStep2<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_internalData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clusterReorganizing<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_internalData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleDynamicsStep1<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_internalData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleDynamicsStep2<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_internalData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleReorganizing<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_internalData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clearMaps<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_internalData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	swapData();
}

void CudaSimulation::getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight, DataAccessTO const& dataTO)
{
	checkCudaErrors(cudaMemset(_cudaAccessTO->numClusters, 0, sizeof(int)));
	checkCudaErrors(cudaMemset(_cudaAccessTO->numCells, 0, sizeof(int)));
	checkCudaErrors(cudaMemset(_cudaAccessTO->numParticles, 0, sizeof(int)));
	checkCudaErrors(cudaMemset(_cudaAccessTO->numTokens, 0, sizeof(int)));

	getSimulationAccessData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>>(rectUpperLeft, rectLowerRight, *_internalData, *_cudaAccessTO);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

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
	prepareTargetData();

	checkCudaErrors(cudaMemcpy(_cudaAccessTO->numClusters, dataTO.numClusters, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->numCells, dataTO.numCells, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->numParticles, dataTO.numParticles, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->numTokens, dataTO.numTokens, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->clusters, dataTO.clusters, sizeof(ClusterAccessTO) * (*dataTO.numClusters), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->cells, dataTO.cells, sizeof(CellAccessTO) * (*dataTO.numCells), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->particles, dataTO.particles, sizeof(ParticleAccessTO) * (*dataTO.numParticles), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->tokens, dataTO.tokens, sizeof(TokenAccessTO) * (*dataTO.numTokens), cudaMemcpyHostToDevice));

	filterData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (rectUpperLeft, rectLowerRight, *_internalData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	setSimulationAccessData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (rectUpperLeft, rectLowerRight, *_internalData, *_cudaAccessTO);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	swapData();
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
	parametersToCopy.radiationProbability = parameters.radiationProb;
	parametersToCopy.radiationExponent = parameters.radiationExponent;
	parametersToCopy.radiationFactor = parameters.radiationFactor;
	parametersToCopy.radiationVelocityMultiplier = parameters.radiationVelocityMultiplier;
	parametersToCopy.radiationVelocityPerturbation = parameters.radiationVelocityPerturbation;

	checkCudaErrors(cudaMemcpyToSymbol(cudaSimulationParameters, &parametersToCopy, sizeof(CudaSimulationParameters), 0, cudaMemcpyHostToDevice));
}

void CudaSimulation::prepareTargetData()
{
	_internalData->clustersAC2.reset();
	_internalData->cellsAC2.reset();
	_internalData->particlesAC2.reset();
	_internalData->tokensAC2.reset();
}

void CudaSimulation::swapData()
{
	swap(_internalData->clustersAC1, _internalData->clustersAC2);
	swap(_internalData->cellsAC1, _internalData->cellsAC2);
	swap(_internalData->particlesAC1, _internalData->particlesAC2);
	swap(_internalData->tokensAC1, _internalData->tokensAC2);
}
