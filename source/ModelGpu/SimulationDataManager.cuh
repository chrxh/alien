#pragma once

#include <iostream>

#include "CudaInterface.cuh"
#include "SimulationData.cuh"
#include "Map.cuh"

class SimulationDataManager
{
public:
	SimulationData data;
	SimulationDataForAccess access;

	SimulationDataManager(int2 const &size)
	{
		int totalVideoMemory = 0;
		totalVideoMemory += 2 * sizeof(ClusterData) * MAX_CELLCLUSTERS;
		totalVideoMemory += 2 * sizeof(CellData) * MAX_CELLS;
		totalVideoMemory += 2 * sizeof(ParticleData) * MAX_PARTICLES;
		totalVideoMemory += 2 * size.x * size.y * sizeof(ParticleData*);
		totalVideoMemory += 2 * size.x * size.y * sizeof(CellData*);
		totalVideoMemory += sizeof(int) * RANDOM_NUMBER_BLOCK_SIZE;
		std::cout << "Total video memory needed for CUDA: " << totalVideoMemory/1024/1024 << "mb" << std::endl;

		data.size = size;

		data.clustersAC1 = ArrayController<ClusterData>(MAX_CELLCLUSTERS);
		data.clustersAC2 = ArrayController<ClusterData>(MAX_CELLCLUSTERS);
		data.cellsAC1 = ArrayController<CellData>(MAX_CELLS);
		data.cellsAC2 = ArrayController<CellData>(MAX_CELLS);
		data.particlesAC1 = ArrayController<ParticleData>(MAX_PARTICLES);
		data.particlesAC2 = ArrayController<ParticleData>(MAX_PARTICLES);

		cudaMallocManaged(&data.cellMap1, size.x * size.y * sizeof(CellData*));
		cudaMallocManaged(&data.cellMap2, size.x * size.y * sizeof(CellData*));
		cudaMallocManaged(&data.particleMap1, size.x * size.y * sizeof(ParticleData*));
		cudaMallocManaged(&data.particleMap2, size.x * size.y * sizeof(ParticleData*));
		checkCudaErrors(cudaGetLastError());

		access.clusters = static_cast<ClusterData*>(malloc(sizeof(ClusterData) * static_cast<int>(MAX_CELLCLUSTERS)));
		access.cells = static_cast<CellData*>(malloc(sizeof(CellData) * static_cast<int>(MAX_CELLS)));
		access.particles = static_cast<ParticleData*>(malloc(sizeof(ParticleData) * static_cast<int>(MAX_PARTICLES)));

		for (int i = 0; i < size.x * size.y; ++i) {
			data.cellMap1[i] = nullptr;
			data.cellMap2[i] = nullptr;
			data.particleMap1[i] = nullptr;
			data.particleMap2[i] = nullptr;
		}

		data.numberGen.init(RANDOM_NUMBER_BLOCK_SIZE);
	}

	~SimulationDataManager()
	{
		data.clustersAC1.free();
		data.clustersAC2.free();
		data.cellsAC1.free();
		data.cellsAC2.free();
		data.particlesAC1.free();
		data.particlesAC2.free();

		cudaFree(data.cellMap1);
		cudaFree(data.cellMap2);
		cudaFree(data.particleMap1);
		cudaFree(data.particleMap2);
		data.numberGen.free();

		free(access.clusters);
		free(access.cells);
		free(access.particles);
	}

	void swapData()
	{
		swap(data.clustersAC1, data.clustersAC2);
		swap(data.cellsAC1, data.cellsAC2);
		swap(data.particlesAC1, data.particlesAC2);
		swap(data.cellMap1, data.cellMap2);
		swap(data.particleMap1, data.particleMap2);
	}

	void prepareTargetData()
	{
		data.clustersAC2.reset();
		data.cellsAC2.reset();
		data.particlesAC2.reset();
	}

	SimulationDataForAccess getDataForAccess()
	{
		access.numClusters = data.clustersAC1.getNumEntries();
		cudaMemcpy(access.clusters, data.clustersAC1.getEntireArray(), sizeof(ClusterData) * data.clustersAC1.getNumEntries(), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		access.numCells = data.cellsAC1.getNumEntries();
		cudaMemcpy(access.cells, data.cellsAC1.getEntireArray(), sizeof(CellData) * data.cellsAC1.getNumEntries(), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());
		access.numParticles = data.particlesAC1.getNumEntries();
		cudaMemcpy(access.particles, data.particlesAC1.getEntireArray(), sizeof(ParticleData) * data.particlesAC1.getNumEntries(), cudaMemcpyDeviceToHost);
		checkCudaErrors(cudaGetLastError());

		correctPointersAfterCopy(int64_t(access.cells) - int64_t(data.cellsAC1.getEntireArray()));
		return access;
	}

	void setDataForAccess(SimulationDataForAccess const& newAccess)
	{
		access = newAccess;

		data.clustersAC1.setNumEntries(access.numClusters);
		cudaMemcpy(data.clustersAC1.getEntireArray(), access.clusters, sizeof(ClusterData) * data.clustersAC1.getNumEntries(), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		data.cellsAC1.setNumEntries(access.numCells);
		cudaMemcpy(data.cellsAC1.getEntireArray(), access.cells, sizeof(CellData) * data.cellsAC1.getNumEntries(), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());
		data.particlesAC1.setNumEntries(access.numParticles);
		cudaMemcpy(data.particlesAC1.getEntireArray(), access.particles, sizeof(ParticleData) * data.particlesAC1.getNumEntries(), cudaMemcpyHostToDevice);
		checkCudaErrors(cudaGetLastError());

		correctPointersAfterCopy(int64_t(data.cellsAC1.getEntireArray()) - int64_t(access.cells));
	}

	void updateAngularMass(ClusterData* cluster)
	{
		cluster->angularMass = 0.0;
		for (int i = 0; i < cluster->numCells; ++i) {
			auto relPos = cluster->cells[i].relPos;
			cluster->angularMass += dot(relPos, relPos);
		}
	}

	void centerCluster(ClusterData* cluster)
	{
		float2 center = { 0.0, 0.0 };
		for (int i = 0; i < cluster->numCells; ++i) {
			auto const &relPos = cluster->cells[i].relPos;
			center.x += relPos.x;
			center.y += relPos.y;
		}
		center.x /= static_cast<float>(cluster->numCells);
		center.y /= static_cast<float>(cluster->numCells);
		for (int i = 0; i < cluster->numCells; ++i) {
			auto &relPos = cluster->cells[i].relPos;
			relPos.x -= center.x;
			relPos.y -= center.y;
		}
	}

	void updateAbsPos(ClusterData *cluster)
	{

		float rotMatrix[2][2];
		float sinAngle = sinf(cluster->angle*DEG_TO_RAD);
		float cosAngle = cosf(cluster->angle*DEG_TO_RAD);
		rotMatrix[0][0] = cosAngle;
		rotMatrix[0][1] = -sinAngle;
		rotMatrix[1][0] = sinAngle;
		rotMatrix[1][1] = cosAngle;
		for (int i = 0; i < cluster->numCells; ++i) {
			auto &relPos = cluster->cells[i].relPos;
			auto &absPos = cluster->cells[i].absPos;
			absPos.x = relPos.x*rotMatrix[0][0] + relPos.y*rotMatrix[0][1] + cluster->pos.x;
			absPos.y = relPos.x*rotMatrix[1][0] + relPos.y*rotMatrix[1][1] + cluster->pos.y;
		};
	}

	bool isClusterPositionFree(ClusterData* cluster, SimulationData* data)
	{
		for (int i = 0; i < cluster->numCells; ++i) {
			auto &absPos = cluster->cells[i].absPos;
			if (getFromMap({ static_cast<int>(absPos.x),static_cast<int>(absPos.y) }, data->cellMap1, data->size)) {
				return false;
			}
			if (getFromMap({ static_cast<int>(absPos.x - 1),static_cast<int>(absPos.y) }, data->cellMap1, data->size)) {
				return false;
			}
			if (getFromMap({ static_cast<int>(absPos.x + 1),static_cast<int>(absPos.y) }, data->cellMap1, data->size)) {
				return false;
			}
			if (getFromMap({ static_cast<int>(absPos.x),static_cast<int>(absPos.y - 1) }, data->cellMap1, data->size)) {
				return false;
			}
			if (getFromMap({ static_cast<int>(absPos.x),static_cast<int>(absPos.y + 1) }, data->cellMap1, data->size)) {
				return false;
			}
		}
		return true;
	}

	void drawClusterToMap(ClusterData* cluster, SimulationData* data)
	{
		for (int i = 0; i < cluster->numCells; ++i) {
			auto &absPos = cluster->cells[i].absPos;
			setToMap({ static_cast<int>(absPos.x),static_cast<int>(absPos.y) }, &cluster->cells[i], data->cellMap1, data->size);
		}
	}

	void createCluster(SimulationData &data, ClusterData* cluster, float2 pos, float2 vel, float angle, float angVel, float energy, int2 clusterSize, int2 const &size)
	{
		cluster->id = data.numberGen.newId();
		cluster->pos = pos;
		cluster->vel = vel;
		cluster->angle = angle;
		cluster->angularVel = angVel;
		cluster->numCells = clusterSize.x * clusterSize.y;
		cluster->cells = data.cellsAC1.getArray(clusterSize.x*clusterSize.y);

		for (int x = 0; x < clusterSize.x; ++x) {
			for (int y = 0; y < clusterSize.y; ++y) {
				CellData *cell = &cluster->cells[x + y*clusterSize.x];
				cell->id = data.numberGen.newId();
				cell->relPos = { static_cast<float>(x), static_cast<float>(y) };
				cell->cluster = cluster;
				cell->nextTimestep = nullptr;
				cell->protectionCounter = 0;
				cell->numConnections = 0;
				cell->energy = energy;
				cell->setProtectionCounterForNextTimestep = false;
				cell->alive = true;
				if (x > 0) {
					cell->connections[cell->numConnections++] = &cluster->cells[x - 1 + y * clusterSize.x];
				}
				if (y > 0) {
					cell->connections[cell->numConnections++] = &cluster->cells[x + (y - 1) * clusterSize.x];
				}
				if (x < clusterSize.x - 1) {
					cell->connections[cell->numConnections++] = &cluster->cells[x + 1 + y * clusterSize.x];
				}
				if (y < clusterSize.y - 1) {
					cell->connections[cell->numConnections++] = &cluster->cells[x + (y + 1) * clusterSize.x];
				}
			}
		}
		centerCluster(cluster);
		updateAbsPos(cluster);
		updateAngularMass(cluster);
	}

private:
	void correctPointersAfterCopy(int64_t addressShift)
	{
		auto cellPtrCorrection = int64_t(access.cells) - int64_t(data.cellsAC1.getEntireArray());
		for (int i = 0; i < access.numClusters; ++i) {
			access.clusters[i].cells = (CellData*)(int64_t(access.clusters[i].cells) + addressShift);
		}

		for (int i = 0; i < access.numCells; ++i) {
			auto &cell = access.cells[i];
			for (int j = 0; j < cell.numConnections; ++j) {
				cell.connections[j] = (CellData*)(int64_t(cell.connections[j]) + addressShift);
			}
		}


	}
};

