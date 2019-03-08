#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <list>
#include <iostream>
#include <functional>

#include "Base.cuh"
#include "CudaSimulation.cuh"
#include "CudaConstants.cuh"
#include "SimulationParameters.cuh"
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

CudaSimulation::CudaSimulation(int2 const &size)
{

	CudaInitializer::init();

	setCudaSimulationParameters();

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
	checkCudaErrors(cudaMalloc(&_internalData->cellMap, size.x * size.y * sizeof(Cell*)));
	checkCudaErrors(cudaMalloc(&_internalData->particleMap, size.x * size.y * sizeof(Particle*)));

	std::vector<Cell*> hostCellMap(size.x * size.y, 0);
	std::vector<Particle*> hostParticleMap(size.x * size.y, 0);
	checkCudaErrors(cudaMemcpy(_internalData->cellMap, hostCellMap.data(), sizeof(Cell*)*size.x*size.y, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_internalData->particleMap, hostParticleMap.data(), sizeof(Cell*)*size.x*size.y, cudaMemcpyHostToDevice));
	_internalData->numberGen.init(RANDOM_NUMBER_BLOCK_SIZE);

	_accessTO = new SimulationAccessTO();
	_accessTO->numClusters = new int();
	_accessTO->numCells = new int();
	_accessTO->numParticles = new int();
	_accessTO->clusters = new ClusterAccessTO[MAX_CELLCLUSTERS];
	_accessTO->cells = new CellAccessTO[MAX_CELLS];
	_accessTO->particles = new ParticleAccessTO[MAX_PARTICLES];
	*_accessTO->numClusters = 0;
	*_accessTO->numCells = 0;
	*_accessTO->numParticles = 0;

	_cudaAccessTO = new SimulationAccessTO();
	checkCudaErrors(cudaMalloc(&_cudaAccessTO->numClusters, sizeof(int)));
	checkCudaErrors(cudaMalloc(&_cudaAccessTO->numCells, sizeof(int)));
	checkCudaErrors(cudaMalloc(&_cudaAccessTO->numParticles, sizeof(int)));
	checkCudaErrors(cudaMalloc(&_cudaAccessTO->clusters, sizeof(ClusterAccessTO)*MAX_CELLCLUSTERS));
	checkCudaErrors(cudaMalloc(&_cudaAccessTO->cells, sizeof(CellAccessTO)*MAX_CELLS));
	checkCudaErrors(cudaMalloc(&_cudaAccessTO->particles, sizeof(ParticleAccessTO)*MAX_PARTICLES));
}

CudaSimulation::~CudaSimulation()
{
	_internalData->clustersAC1.free();
	_internalData->clustersAC2.free();
	_internalData->cellsAC1.free();
	_internalData->cellsAC2.free();
	_internalData->particlesAC1.free();
	_internalData->particlesAC2.free();

	checkCudaErrors(cudaFree(_internalData->cellMap));
	checkCudaErrors(cudaFree(_internalData->particleMap));
	_internalData->numberGen.free();

	checkCudaErrors(cudaFree(_cudaAccessTO->numClusters));
	checkCudaErrors(cudaFree(_cudaAccessTO->numCells));
	checkCudaErrors(cudaFree(_cudaAccessTO->numParticles));
	checkCudaErrors(cudaFree(_cudaAccessTO->clusters));
	checkCudaErrors(cudaFree(_cudaAccessTO->cells));
	checkCudaErrors(cudaFree(_cudaAccessTO->particles));

	delete _accessTO->numClusters;
	delete _accessTO->numCells;
	delete _accessTO->numParticles;
	delete[] _accessTO->clusters;
	delete[] _accessTO->cells;
	delete[] _accessTO->particles;

	delete _accessTO;
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

SimulationAccessTO* CudaSimulation::getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight)
{
	_rectUpperLeft = rectUpperLeft;
	_rectLowerRight = rectLowerRight;

	checkCudaErrors(cudaMemset(_cudaAccessTO->numClusters, 0, sizeof(int)));
	checkCudaErrors(cudaMemset(_cudaAccessTO->numCells, 0, sizeof(int)));
	checkCudaErrors(cudaMemset(_cudaAccessTO->numParticles, 0, sizeof(int)));

	getSimulationAccessData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (_rectUpperLeft, _rectLowerRight, *_internalData, *_cudaAccessTO);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(_accessTO->numClusters, _cudaAccessTO->numClusters, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_accessTO->numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_accessTO->numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_accessTO->clusters, _cudaAccessTO->clusters, sizeof(ClusterAccessTO) * (*_accessTO->numClusters), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_accessTO->cells, _cudaAccessTO->cells, sizeof(CellAccessTO) * (*_accessTO->numCells), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_accessTO->particles, _cudaAccessTO->particles, sizeof(ParticleAccessTO) * (*_accessTO->numParticles), cudaMemcpyDeviceToHost));

	return _accessTO;
}

void CudaSimulation::updateSimulationData()
{
 	prepareTargetData();

	checkCudaErrors(cudaMemcpy(_cudaAccessTO->numClusters, _accessTO->numClusters, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->numCells, _accessTO->numCells, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->numParticles, _accessTO->numParticles, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->clusters, _accessTO->clusters, sizeof(ClusterAccessTO) * (*_accessTO->numClusters), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->cells, _accessTO->cells, sizeof(CellAccessTO) * (*_accessTO->numCells), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_cudaAccessTO->particles, _accessTO->particles, sizeof(ParticleAccessTO) * (*_accessTO->numParticles), cudaMemcpyHostToDevice));

	filterData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (_rectUpperLeft, _rectLowerRight, *_internalData);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	setSimulationAccessData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (_rectUpperLeft, _rectLowerRight, *_internalData, *_cudaAccessTO);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	swapData();
}

void CudaSimulation::prepareTargetData()
{
	_internalData->clustersAC2.reset();
	_internalData->cellsAC2.reset();
	_internalData->particlesAC2.reset();
}

void CudaSimulation::swapData()
{
	swap(_internalData->clustersAC1, _internalData->clustersAC2);
	swap(_internalData->cellsAC1, _internalData->cellsAC2);
	swap(_internalData->particlesAC1, _internalData->particlesAC2);
}

void CudaSimulation::setCudaSimulationParameters()
{
	SimulationParameters parametersToCopy;
	parametersToCopy.cellMaxDistance = 1.3f;
	parametersToCopy.cellMinDistance = 0.3f;
	parametersToCopy.cellMinEnergy = 50.0f;
	parametersToCopy.cellFusionVelocity = 0.4f;
	parametersToCopy.cellMaxForce = 0.8f;
	parametersToCopy.cellMaxForceDecayProb = 0.2f;
	parametersToCopy.cellTransformationProb = 0.2f;
	parametersToCopy.cellMass = 1.0;
	parametersToCopy.radiationProbability = 0.2f;
	parametersToCopy.radiationExponent = 1.0f;
	parametersToCopy.radiationFactor = 0.0002f;
	parametersToCopy.radiationVelocityMultiplier = 1.0f;
	parametersToCopy.radiationVelocityPerturbation = 0.5f;

	checkCudaErrors(cudaMemcpyToSymbol(cudaSimulationParameters, &parametersToCopy, sizeof(SimulationParameters) , 0, cudaMemcpyHostToDevice));
}
