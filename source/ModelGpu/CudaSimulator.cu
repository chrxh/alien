#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <list>
#include <iostream>
#include <functional>

#include "Base.cuh"
#include "CudaConstants.cuh"
#include "CudaSimulationParameters.cuh"
#include "CudaInterface.cuh"
#include "CudaSimulatorFunctions.cuh"

#include "SimulationDataInternal.cuh"
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

CudaSimulator::CudaSimulator(int2 const &size)
{
	std::cout << "[CUDA debug] begin CudaSimulator::CudaSimulator" << std::endl;

	CudaInitializer::init();
	cudaDeviceSynchronize();

	setCudaSimulationParameters();

	cudaStreamCreate(&_cudaStream);
	std::cout << "[CUDA] stream created" << std::endl;

	int totalVideoMemory = 0;
	totalVideoMemory += 2 * sizeof(ClusterData) * MAX_CELLCLUSTERS;
	totalVideoMemory += 2 * sizeof(CellData) * MAX_CELLS;
	totalVideoMemory += 2 * sizeof(ParticleData) * MAX_PARTICLES;
	totalVideoMemory += size.x * size.y * sizeof(ParticleData*);
	totalVideoMemory += size.x * size.y * sizeof(CellData*);
	totalVideoMemory += sizeof(int) * RANDOM_NUMBER_BLOCK_SIZE;
	std::cout << "[CUDA] acquire " << totalVideoMemory/1024/1024 << "mb of video memory" << std::endl;
	cudaDeviceSynchronize();

	_data = new SimulationDataInternal();
	std::cout << "[CUDA debug] 1" << std::endl;
	_data->size = size;

	std::cout << "[CUDA debug] step 1 CudaSimulator::CudaSimulator" << std::endl;

	_data->clustersAC1 = ArrayController<ClusterData>(MAX_CELLCLUSTERS);
	_data->clustersAC2 = ArrayController<ClusterData>(MAX_CELLCLUSTERS);
	_data->cellsAC1 = ArrayController<CellData>(MAX_CELLS);
	_data->cellsAC2 = ArrayController<CellData>(MAX_CELLS);
	_data->particlesAC1 = ArrayController<ParticleData>(MAX_PARTICLES);
	_data->particlesAC2 = ArrayController<ParticleData>(MAX_PARTICLES);

	std::cout << "[CUDA debug] step 2 CudaSimulator::CudaSimulator" << std::endl;

	cudaMallocManaged(&_data->cellMap, size.x * size.y * sizeof(CellData*));
	cudaMallocManaged(&_data->particleMap, size.x * size.y * sizeof(ParticleData*));
	checkCudaErrors(cudaGetLastError());

	std::cout << "[CUDA debug] step 3 CudaSimulator::CudaSimulator" << std::endl;

	_access = new SimulationDataForAccess();
	_access->clusters = new ClusterData[MAX_CELLCLUSTERS];
	_access->cells = new CellData[MAX_CELLS];
	_access->particles = new ParticleData[MAX_PARTICLES];

	std::cout << "[CUDA debug] step 4 CudaSimulator::CudaSimulator" << std::endl;

	for (int i = 0; i < size.x * size.y; ++i) {
		_data->cellMap[i] = nullptr;
		_data->particleMap[i] = nullptr;
	}
	std::cout << "[CUDA debug] step 5 CudaSimulator::CudaSimulator" << std::endl;

	_data->numberGen.init(RANDOM_NUMBER_BLOCK_SIZE);
	cudaDeviceSynchronize();

	std::cout << "[CUDA debug] end CudaSimulator::CudaSimulator" << std::endl;

}

CudaSimulator::~CudaSimulator()
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

	delete[] _access->clusters;
	delete[] _access->cells;
	delete[] _access->particles;

	delete _data;
	delete _access;
	std::cout << "[CUDA] stream closed" << std::endl;
}

void CudaSimulator::calcNextTimestep()
{
	prepareTargetData();

	clusterDynamicsStep1<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clusterDynamicsStep2<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clusterReassembling<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleDynamicsStep1 << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream >> > (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleDynamicsStep2<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleReassembling<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clearMaps<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, _cudaStream>>> (*_data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	swapData();
}

SimulationDataForAccess const& CudaSimulator::getDataForAccess()
{
	std::cout << "[CUDA debug] begin CudaSimulator::getDataForAccess" << std::endl;

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

	std::cout << "[CUDA debug] end CudaSimulator::getDataForAccess" << std::endl;
	return *_access;
}

void CudaSimulator::setDataForAccess(SimulationDataForAccess const& newAccess)
{
	std::cout << "[CUDA debug] begin CudaSimulator::setDataForAccess" << std::endl;

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
/*
	Map<CellData> map;
	map.init(_data->size, _data->cellMap);
	for (int index = 0; index < _data->cellsAC1.getNumEntries(); ++index) {
		CellData* cell = _data->cellsAC1.at(index);
		auto& absPos = cell->absPos;
		map.set(absPos, cell);
	}
*/
	cudaDeviceSynchronize();

	std::cout << "[CUDA debug] end CudaSimulator::setDataForAccess" << std::endl;
}

void CudaSimulator::prepareTargetData()
{
	_data->clustersAC2.reset();
	_data->cellsAC2.reset();
	_data->particlesAC2.reset();
}

void CudaSimulator::swapData()
{
	swap(_data->clustersAC1, _data->clustersAC2);
	swap(_data->cellsAC1, _data->cellsAC2);
	swap(_data->particlesAC1, _data->particlesAC2);
}

void CudaSimulator::correctPointersAfterCellCopy(CellData* cell, int64_t addressShiftCell, int64_t addressShiftCluster)
{
	cell->cluster = (ClusterData*)(int64_t(cell->cluster) + addressShiftCluster);
	for (int j = 0; j < cell->numConnections; ++j) {
		cell->connections[j] = (CellData*)(int64_t(cell->connections[j]) + addressShiftCell);
	}
	cell->nextTimestep = nullptr;
}

void CudaSimulator::correctPointersAfterClusterCopy(ClusterData* cluster, int64_t addressShiftCell)
{
	cluster->cells = (CellData*)(int64_t(cluster->cells) + addressShiftCell);
}

void CudaSimulator::setCudaSimulationParameters()
{
	std::cout << "[CUDA debug] begin CudaSimulator::setCudaSimulationParameters" << std::endl;

	cudaDeviceSynchronize();
	CudaSimulationParameters parametersToCopy;

	parametersToCopy.cellMaxDistance = 1.3f;
	parametersToCopy.cellMinDistance = 0.3f;
	parametersToCopy.cellMinEnergy = 50.0f;
	parametersToCopy.radiationProbability = 0.2f;
	parametersToCopy.radiationExponent = 1.0f;
	parametersToCopy.radiationFactor = 0.0002f;
	parametersToCopy.radiationVelocityPerturbation = 0.5f;

	cudaMemcpyToSymbol(cudaSimulationParameters, &parametersToCopy, sizeof(CudaSimulationParameters) , 0, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	cudaDeviceSynchronize();

	std::cout << "[CUDA debug] end CudaSimulator::setCudaSimulationParameters" << std::endl;
}