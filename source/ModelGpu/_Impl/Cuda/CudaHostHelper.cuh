#pragma once

#include "CudaShared.cuh"

class CudaSimulationManager
{
public:
	CudaSimulation data;

	CudaDataForAccess access;

	CudaSimulationManager(int2 const &size)
	{
		data.size = size;

		data.clustersAC1 = ArrayController<CudaCellCluster>(MAX_CELLCLUSTERS);
		data.clustersAC2 = ArrayController<CudaCellCluster>(MAX_CELLCLUSTERS);
		data.cellsAC1 = ArrayController<CudaCell>(MAX_CELLS);
		data.cellsAC2 = ArrayController<CudaCell>(MAX_CELLS);
		data.particlesAC1 = ArrayController<CudaEnergyParticle>(MAX_ENERGY_PARTICLES);
		data.particlesAC2 = ArrayController<CudaEnergyParticle>(MAX_ENERGY_PARTICLES);

		size_t mapSize = size.x * size.y * sizeof(CudaCell*);
		cudaMallocManaged(&data.map1, mapSize);
		cudaMallocManaged(&data.map2, mapSize);
		checkCudaErrors(cudaGetLastError());

		access.cells = static_cast<CudaCell*>(malloc(sizeof(CudaCell) * static_cast<int>(MAX_CELLS)));
		access.particles = static_cast<CudaEnergyParticle*>(malloc(sizeof(CudaEnergyParticle) * static_cast<int>(MAX_ENERGY_PARTICLES)));

		for (int i = 0; i < size.x * size.y; ++i) {
			data.map1[i] = nullptr;
			data.map2[i] = nullptr;
		}

		data.randomGen.init(RANDOM_NUMBER_BLOCK_SIZE);
	}

	~CudaSimulationManager()
	{
		data.clustersAC1.free();
		data.clustersAC2.free();
		data.cellsAC1.free();
		data.cellsAC2.free();
		data.particlesAC1.free();
		data.particlesAC2.free();

		cudaFree(data.map1);
		cudaFree(data.map2);
		data.randomGen.free();

		free(access.cells);
		free(access.particles);
	}

	void swapData()
	{
		swap(data.clustersAC1, data.clustersAC2);
		swap(data.cellsAC1, data.cellsAC2);
		swap(data.particlesAC1, data.particlesAC2);
		swap(data.map1, data.map2);
	}

	void prepareTargetData()
	{
		data.clustersAC2.reset();
		data.cellsAC2.reset();
		data.particlesAC2.reset();
	}

	CudaDataForAccess getDataForAccess()
	{
		access.numCells = data.cellsAC2.getNumEntries();
		cudaMemcpy(access.cells, data.cellsAC2.getEntireArray(), sizeof(CudaCell) * data.cellsAC2.getNumEntries(), cudaMemcpyDeviceToHost);
		access.numParticles = data.particlesAC2.getNumEntries();
		cudaMemcpy(access.particles, data.particlesAC2.getEntireArray(), sizeof(CudaEnergyParticle) * data.particlesAC2.getNumEntries(), cudaMemcpyDeviceToHost);
		return access;
	}
};

void updateAngularMass(CudaCellCluster* cluster)
{
	cluster->angularMass = 0.0;
	for (int i = 0; i < cluster->numCells; ++i) {
		auto relPos = cluster->cells[i].relPos;
		cluster->angularMass += dot(relPos, relPos);
	}
}

void centerCluster(CudaCellCluster* cluster)
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

void updateAbsPos(CudaCellCluster *cluster)
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

bool isClusterPositionFree(CudaCellCluster* cluster, CudaSimulation* data)
{
	for (int i = 0; i < cluster->numCells; ++i) {
		auto &absPos = cluster->cells[i].absPos;
		if (getCellFromMap({ static_cast<int>(absPos.x),static_cast<int>(absPos.y) }, data->map1, data->size)) {
			return false;
		}
		if (getCellFromMap({ static_cast<int>(absPos.x-1),static_cast<int>(absPos.y) }, data->map1, data->size)) {
			return false;
		}
		if (getCellFromMap({ static_cast<int>(absPos.x+1),static_cast<int>(absPos.y) }, data->map1, data->size)) {
			return false;
		}
		if (getCellFromMap({ static_cast<int>(absPos.x),static_cast<int>(absPos.y-1) }, data->map1, data->size)) {
			return false;
		}
		if (getCellFromMap({ static_cast<int>(absPos.x),static_cast<int>(absPos.y+1) }, data->map1, data->size)) {
			return false;
		}
	}
	return true;
}

void drawClusterToMap(CudaCellCluster* cluster, CudaSimulation* data)
{
	for (int i = 0; i < cluster->numCells; ++i) {
		auto &absPos = cluster->cells[i].absPos;
		setCellToMap({ static_cast<int>(absPos.x),static_cast<int>(absPos.y) }, &cluster->cells[i], data->map1, data->size);
	}
}
