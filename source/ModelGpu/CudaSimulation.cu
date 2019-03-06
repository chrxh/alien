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
	cudaDeviceSynchronize();

	setCudaSimulationParameters();

	cudaStreamCreate(&_cudaStream);
	std::cout << "[CUDA] stream created" << std::endl;

	int totalVideoMemory = 0;
	totalVideoMemory += 2 * sizeof(Cluster) * MAX_CELLCLUSTERS;
	totalVideoMemory += 2 * sizeof(Cell) * MAX_CELLS;
	totalVideoMemory += 2 * sizeof(Particle) * MAX_PARTICLES;
	totalVideoMemory += size.x * size.y * sizeof(Particle*);
	totalVideoMemory += size.x * size.y * sizeof(Cell*);
	totalVideoMemory += sizeof(int) * RANDOM_NUMBER_BLOCK_SIZE;
	std::cout << "[CUDA] acquire " << totalVideoMemory/1024/1024 << "mb of video memory" << std::endl;
	cudaDeviceSynchronize();

	_data = new SimulationData();
	_data->size = size;
	_data->clustersAC1 = ArrayController<Cluster>(MAX_CELLCLUSTERS);
	_data->clustersAC2 = ArrayController<Cluster>(MAX_CELLCLUSTERS);
	_data->cellsAC1 = ArrayController<Cell>(MAX_CELLS);
	_data->cellsAC2 = ArrayController<Cell>(MAX_CELLS);
	_data->particlesAC1 = ArrayController<Particle>(MAX_PARTICLES);
	_data->particlesAC2 = ArrayController<Particle>(MAX_PARTICLES);
	cudaMallocManaged(&_data->cellMap, size.x * size.y * sizeof(Cell*));
	cudaMallocManaged(&_data->particleMap, size.x * size.y * sizeof(Particle*));
	checkCudaErrors(cudaGetLastError());
	for (int i = 0; i < size.x * size.y; ++i) {
		_data->cellMap[i] = nullptr;
		_data->particleMap[i] = nullptr;
	}
	_data->numberGen.init(RANDOM_NUMBER_BLOCK_SIZE);

	cudaDeviceSynchronize();

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
	cudaMalloc(&_cudaAccessTO->numClusters, sizeof(int));
	checkCudaErrors(cudaGetLastError());

	cudaMalloc(&_cudaAccessTO->numCells, sizeof(int));
	checkCudaErrors(cudaGetLastError());

	cudaMalloc(&_cudaAccessTO->numParticles, sizeof(int));
	checkCudaErrors(cudaGetLastError());

	cudaMalloc(&_cudaAccessTO->clusters, sizeof(ClusterAccessTO)*MAX_CELLCLUSTERS);
	checkCudaErrors(cudaGetLastError());

	cudaMalloc(&_cudaAccessTO->cells, sizeof(CellAccessTO)*MAX_CELLS);
	checkCudaErrors(cudaGetLastError());

	cudaMalloc(&_cudaAccessTO->particles, sizeof(ParticleAccessTO)*MAX_PARTICLES);
	checkCudaErrors(cudaGetLastError());

	/*
	cudaMallocManaged(&_access->numClusters, sizeof(int*));
	cudaMallocManaged(&_access->numCells, sizeof(int*));
	cudaMallocManaged(&_access->numParticles, sizeof(int*));
	cudaMallocManaged(&_access->clusters, sizeof(ClusterAccessTO*)*MAX_CELLCLUSTERS);
	cudaMallocManaged(&_access->cells, sizeof(CellAccessTO*)*MAX_CELLS);
	cudaMallocManaged(&_access->particles, sizeof(ParticleAccessTO*)*MAX_PARTICLES);
*/
}

CudaSimulation::~CudaSimulation()
{
	cudaDeviceSynchronize();

	_data->clustersAC1.free();
	_data->clustersAC2.free();
	_data->cellsAC1.free();
	_data->cellsAC2.free();
	_data->particlesAC1.free();
	_data->particlesAC2.free();

	cudaFree(_data->cellMap);
	cudaFree(_data->particleMap);
	_data->numberGen.free();

	cudaFree(_cudaAccessTO->numClusters);
	cudaFree(_cudaAccessTO->numCells);
	cudaFree(_cudaAccessTO->numParticles);
	cudaFree(_cudaAccessTO->clusters);
	cudaFree(_cudaAccessTO->cells);
	cudaFree(_cudaAccessTO->particles);

/*
	cudaFree(_access->numClusters);
	cudaFree(_access->numCells);
	cudaFree(_access->numParticles);
	cudaFree(_access->clusters);
	cudaFree(_access->cells);
	cudaFree(_access->particles);
*/
	delete _accessTO->numClusters;
	delete _accessTO->numCells;
	delete _accessTO->numParticles;
	delete[] _accessTO->clusters;
	delete[] _accessTO->cells;
	delete[] _accessTO->particles;

	delete _accessTO;
	delete _cudaAccessTO;
	delete _data;

	std::cout << "[CUDA] stream closed" << std::endl;
}

void CudaSimulation::calcNextTimestep()
{
	prepareTargetData();

	clusterDynamicsStep1<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clusterDynamicsStep2<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clusterReorganizing<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleDynamicsStep1 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream >> > (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleDynamicsStep2<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleReorganizing<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clearMaps<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	swapData();
}

SimulationAccessTO* CudaSimulation::getSimulationData(int2 const& rectUpperLeft, int2 const& rectLowerRight)
{
	_rectUpperLeft = rectUpperLeft;
	_rectLowerRight = rectLowerRight;

	int zero = 0;
	cudaMemcpy(_cudaAccessTO->numClusters, &zero, sizeof(int), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	cudaMemcpy(_cudaAccessTO->numCells, &zero, sizeof(int), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	cudaMemcpy(_cudaAccessTO->numParticles, &zero, sizeof(int), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	getSimulationAccessData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (_rectUpperLeft, _rectLowerRight, *_data, *_cudaAccessTO);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_accessTO->numClusters, _cudaAccessTO->numClusters, sizeof(int), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_accessTO->numCells, _cudaAccessTO->numCells, sizeof(int), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_accessTO->numParticles, _cudaAccessTO->numParticles, sizeof(int), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_accessTO->clusters, _cudaAccessTO->clusters, sizeof(ClusterAccessTO) * (*_accessTO->numClusters), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_accessTO->cells, _cudaAccessTO->cells, sizeof(CellAccessTO) * (*_accessTO->numCells), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_accessTO->particles, _cudaAccessTO->particles, sizeof(ParticleAccessTO) * (*_accessTO->numParticles), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	return _accessTO;

/*
	cudaDeviceSynchronize();
	_access->numClusters = _data->clustersAC1.getNumEntries();
	cudaMemcpy(_access->clusters, _data->clustersAC1.getEntireArray(), sizeof(ClusterData) * _data->clustersAC1.getNumEntries(), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());
	_access->numCells = _data->cellsAC1.getNumEntries();
	cudaMemcpy(_access->cells, _data->cellsAC1.getEntireArray(), sizeof(CellData) * _data->cellsAC1.getNumEntries(), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());
	_access->numParticles = _data->particlesAC1.getNumEntries();
	cudaMemcpy(_access->particles, _data->particlesAC1.getEntireArray(), sizeof(ParticleData) * _data->particlesAC1.getNumEntries(), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();

	int64_t addrShiftCell = int64_t(_access->cells) - int64_t(_data->cellsAC1.getEntireArray());
	int64_t addrShiftCluster = int64_t(_access->clusters) - int64_t(_data->clustersAC1.getEntireArray());
	for (int i = 0; i < _access->numClusters; ++i) {
		correctPointersAfterClusterCopy(&_access->clusters[i], addrShiftCell);
	}
	for (int i = 0; i < _access->numCells; ++i) {
		correctPointersAfterCellCopy(&_access->cells[i], addrShiftCell, addrShiftCluster);
	}

	return *_access;
*/
}

void CudaSimulation::updateSimulationData()
{
	prepareTargetData();

	cudaMemcpy(_cudaAccessTO->numClusters, _accessTO->numClusters, sizeof(int), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_cudaAccessTO->numCells, _accessTO->numCells, sizeof(int), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_cudaAccessTO->numParticles, _accessTO->numParticles, sizeof(int), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_cudaAccessTO->clusters, _accessTO->clusters, sizeof(ClusterAccessTO) * (*_accessTO->numClusters), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_cudaAccessTO->cells, _accessTO->cells, sizeof(CellAccessTO) * (*_accessTO->numCells), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	cudaMemcpy(_cudaAccessTO->particles, _accessTO->particles, sizeof(ParticleAccessTO) * (*_accessTO->numParticles), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	filterData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (_rectUpperLeft, _rectLowerRight, *_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	setSimulationAccessData<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (_rectUpperLeft, _rectLowerRight, *_data, *_cudaAccessTO);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	swapData();

	/*
	cudaDeviceSynchronize();
	*_access = newAccess;
	_data->clustersAC1.setNumEntries(_access->numClusters);
	cudaMemcpy(_data->clustersAC1.getEntireArray(), _access->clusters, sizeof(ClusterData) * _data->clustersAC1.getNumEntries(), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	_data->cellsAC1.setNumEntries(_access->numCells);
	cudaMemcpy(_data->cellsAC1.getEntireArray(), _access->cells, sizeof(CellData) * _data->cellsAC1.getNumEntries(), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	_data->particlesAC1.setNumEntries(_access->numParticles);
	cudaMemcpy(_data->particlesAC1.getEntireArray(), _access->particles, sizeof(ParticleData) * _data->particlesAC1.getNumEntries(), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	int64_t addrShiftCell = int64_t(_data->cellsAC1.getEntireArray()) - int64_t(_access->cells);
	int64_t addrShiftCluster = int64_t(_data->clustersAC1.getEntireArray()) - int64_t(_access->clusters);
	for (int i = 0; i < _data->clustersAC1.getNumEntries(); ++i) {
		correctPointersAfterClusterCopy(_data->clustersAC1.at(i), addrShiftCell);
	}
	for (int i = 0; i < _data->cellsAC1.getNumEntries(); ++i) {
		correctPointersAfterCellCopy(_data->cellsAC1.at(i), addrShiftCell, addrShiftCluster);
	}

	for (int i = 0; i < _data->size.x * _data->size.y; ++i) {
		_data->cellMap[i] = nullptr;
		_data->particleMap[i] = nullptr;
	}
	cudaDeviceSynchronize();
*/
}

void CudaSimulation::prepareTargetData()
{
	_data->clustersAC2.reset();
	_data->cellsAC2.reset();
	_data->particlesAC2.reset();
}

void CudaSimulation::swapData()
{
	swap(_data->clustersAC1, _data->clustersAC2);
	swap(_data->cellsAC1, _data->cellsAC2);
	swap(_data->particlesAC1, _data->particlesAC2);
}

/*
void CudaSimulation::correctPointersAfterCellCopy(CellAccessTO* cell, int64_t addressShiftCell, int64_t addressShiftCluster)
{
	for (int j = 0; j < cell->numConnections; ++j) {
		cell->connections[j] = (CellAccessTO*)(int64_t(cell->connections[j]) + addressShiftCell);
	}
}

void CudaSimulation::correctPointersAfterClusterCopy(ClusterAccessTO* cluster, int64_t addressShiftCell)
{
	cluster->cells = (CellAccessTO*)(int64_t(cluster->cells) + addressShiftCell);
}

*/
void CudaSimulation::setCudaSimulationParameters()
{
	cudaDeviceSynchronize();
	SimulationParameters parametersToCopy;

	parametersToCopy.cellMaxDistance = 1.3f;
	parametersToCopy.cellMinDistance = 0.3f;
	parametersToCopy.cellMinEnergy = 50.0f;
	parametersToCopy.cellFusionVelocity = 0.8f;
	parametersToCopy.cellMaxForce = 0.8f;
	parametersToCopy.cellMaxForceDecayProb = 0.2f;
	parametersToCopy.cellTransformationProb = 0.2f;
	parametersToCopy.cellMass = 1.0;
	parametersToCopy.radiationProbability = 0.2f;
	parametersToCopy.radiationExponent = 1.0f;
	parametersToCopy.radiationFactor = 0.0002f;
	parametersToCopy.radiationVelocityMultiplier = 1.0f;
	parametersToCopy.radiationVelocityPerturbation = 0.5f;

	cudaMemcpyToSymbol(cudaSimulationParameters, &parametersToCopy, sizeof(SimulationParameters) , 0, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();
}
