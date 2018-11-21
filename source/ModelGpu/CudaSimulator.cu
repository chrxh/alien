#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <list>
#include <iostream>
#include <functional>

#include "Base.cuh"
#include "Constants.cuh"
#include "CudaInterface.cuh"
#include "CudaSimulatorFunctions.cuh"

#include "CudaInterface.cuh"
#include "SimulationDataInternal.cuh"
#include "Map.cuh"

namespace
{
	class CudaInitializer
	{
	public:
		static CudaInitializer& getInstance()
		{
			static CudaInitializer instance;
			return instance;
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
	CudaInitializer::getInstance();

	cudaStreamCreate(&cudaStream);
	std::cout << "[CUDA] stream created" << std::endl;

	int totalVideoMemory = 0;
	totalVideoMemory += 2 * sizeof(ClusterData) * MAX_CELLCLUSTERS;
	totalVideoMemory += 2 * sizeof(CellData) * MAX_CELLS;
	totalVideoMemory += 2 * sizeof(ParticleData) * MAX_PARTICLES;
	totalVideoMemory += 2 * size.x * size.y * sizeof(ParticleData*);
	totalVideoMemory += 2 * size.x * size.y * sizeof(CellData*);
	totalVideoMemory += sizeof(int) * RANDOM_NUMBER_BLOCK_SIZE;
	std::cout << "[CUDA] acquire " << totalVideoMemory/1024/1024 << "mb of video memory" << std::endl;

	data = new SimulationDataInternal();
	data->size = size;

	data->clustersAC1 = ArrayController<ClusterData>(MAX_CELLCLUSTERS);
	data->clustersAC2 = ArrayController<ClusterData>(MAX_CELLCLUSTERS);
	data->cellsAC1 = ArrayController<CellData>(MAX_CELLS);
	data->cellsAC2 = ArrayController<CellData>(MAX_CELLS);
	data->particlesAC1 = ArrayController<ParticleData>(MAX_PARTICLES);
	data->particlesAC2 = ArrayController<ParticleData>(MAX_PARTICLES);

	cudaMallocManaged(&data->cellMap1, size.x * size.y * sizeof(CellData*));
	cudaMallocManaged(&data->cellMap2, size.x * size.y * sizeof(CellData*));
	cudaMallocManaged(&data->particleMap1, size.x * size.y * sizeof(ParticleData*));
	cudaMallocManaged(&data->particleMap2, size.x * size.y * sizeof(ParticleData*));
	checkCudaErrors(cudaGetLastError());

	access.clusters = static_cast<ClusterData*>(malloc(sizeof(ClusterData) * static_cast<int>(MAX_CELLCLUSTERS)));
	access.cells = static_cast<CellData*>(malloc(sizeof(CellData) * static_cast<int>(MAX_CELLS)));
	access.particles = static_cast<ParticleData*>(malloc(sizeof(ParticleData) * static_cast<int>(MAX_PARTICLES)));

	for (int i = 0; i < size.x * size.y; ++i) {
		data->cellMap1[i] = nullptr;
		data->cellMap2[i] = nullptr;
		data->particleMap1[i] = nullptr;
		data->particleMap2[i] = nullptr;
	}

	data->numberGen.init(RANDOM_NUMBER_BLOCK_SIZE);
}

CudaSimulator::~CudaSimulator()
{
	cudaDeviceSynchronize();

	data->clustersAC1.free();
	data->clustersAC2.free();
	data->cellsAC1.free();
	data->cellsAC2.free();
	data->particlesAC1.free();
	data->particlesAC2.free();

	cudaFree(data->cellMap1);
	cudaFree(data->cellMap2);
	cudaFree(data->particleMap1);
	cudaFree(data->particleMap2);
	data->numberGen.free();

	free(access.clusters);
	free(access.cells);
	free(access.particles);

	delete data;
	std::cout << "[CUDA] stream closed" << std::endl;
}

void CudaSimulator::calcNextTimestep()
{
	prepareTargetData();

	clusterMovement << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream >> > (*data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	particleMovement << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream >> > (*data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	clearMaps << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, cudaStream >> > (*data);
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());

	swapData();
}

SimulationDataForAccess CudaSimulator::getDataForAccess()
{
	access.numClusters = data->clustersAC1.getNumEntries();
	cudaMemcpy(access.clusters, data->clustersAC1.getEntireArray(), sizeof(ClusterData) * data->clustersAC1.getNumEntries(), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());
	access.numCells = data->cellsAC1.getNumEntries();
	cudaMemcpy(access.cells, data->cellsAC1.getEntireArray(), sizeof(CellData) * data->cellsAC1.getNumEntries(), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());
	access.numParticles = data->particlesAC1.getNumEntries();
	cudaMemcpy(access.particles, data->particlesAC1.getEntireArray(), sizeof(ParticleData) * data->particlesAC1.getNumEntries(), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaGetLastError());

	int64_t addrShiftCell = int64_t(access.cells) - int64_t(data->cellsAC1.getEntireArray());
	int64_t addrShiftCluster = int64_t(access.clusters) - int64_t(data->clustersAC1.getEntireArray());
	for (int i = 0; i < access.numClusters; ++i) {
		correctPointersAfterClusterCopy(&access.clusters[i], addrShiftCell);
	}
	for (int i = 0; i < access.numCells; ++i) {
		correctPointersAfterCellCopy(&access.cells[i], addrShiftCell, addrShiftCluster);
	}

	return access;
}

void CudaSimulator::setDataForAccess(SimulationDataForAccess const& newAccess)
{
	access = newAccess;

	data->clustersAC1.setNumEntries(access.numClusters);
	cudaMemcpy(data->clustersAC1.getEntireArray(), access.clusters, sizeof(ClusterData) * data->clustersAC1.getNumEntries(), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	data->cellsAC1.setNumEntries(access.numCells);
	cudaMemcpy(data->cellsAC1.getEntireArray(), access.cells, sizeof(CellData) * data->cellsAC1.getNumEntries(), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());
	data->particlesAC1.setNumEntries(access.numParticles);
	cudaMemcpy(data->particlesAC1.getEntireArray(), access.particles, sizeof(ParticleData) * data->particlesAC1.getNumEntries(), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaGetLastError());

	int64_t addrShiftCell = int64_t(data->cellsAC1.getEntireArray()) - int64_t(access.cells);
	int64_t addrShiftCluster = int64_t(data->clustersAC1.getEntireArray()) - int64_t(access.clusters);
	for (int i = 0; i < data->clustersAC1.getNumEntries(); ++i) {
		correctPointersAfterClusterCopy(data->clustersAC1.at(i), addrShiftCell);
	}
	for (int i = 0; i < data->cellsAC1.getNumEntries(); ++i) {
		correctPointersAfterCellCopy(data->cellsAC1.at(i), addrShiftCell, addrShiftCluster);
	}

	for (int i = 0; i < data->size.x * data->size.y; ++i) {
		data->cellMap1[i] = nullptr;
		data->particleMap1[i] = nullptr;
	}
	Map<CellData> map;
	map.init(data->size, data->cellMap1, data->cellMap2);
	for (int index = 0; index < data->cellsAC1.getNumEntries(); ++index) {
		CellData* cell = data->cellsAC1.at(index);
		auto& absPos = cell->absPos;
		map.setToOrigMap({ int(absPos.x), int(absPos.y) }, cell);
	}

}

void CudaSimulator::prepareTargetData()
{
	data->clustersAC2.reset();
	data->cellsAC2.reset();
	data->particlesAC2.reset();
}

void CudaSimulator::swapData()
{
	swap(data->clustersAC1, data->clustersAC2);
	swap(data->cellsAC1, data->cellsAC2);
	swap(data->particlesAC1, data->particlesAC2);
	swap(data->cellMap1, data->cellMap2);
	swap(data->particleMap1, data->particleMap2);
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

