#pragma once

#include "CudaShared.cuh"

class CudaSimulationManager
{
public:
	CudaSimulation data;

	CudaCell* cellsForAccess;

	CudaSimulationManager(int2 const &size)
	{
		data.size = size;

		data.clustersAC1 = ArrayController<CudaCellCluster>(static_cast<int>(NUM_CLUSTERS * 1.1));
		data.clustersAC2 = ArrayController<CudaCellCluster>(static_cast<int>(NUM_CLUSTERS * 1.1));
//		data.clustersAC3 = ArrayController<CudaCellCluster>(static_cast<int>(NUM_CLUSTERS * 1.1));
		data.cellsAC1 = ArrayController<CudaCell>(static_cast<int>(NUM_CLUSTERS * 30 * 30 * 1.1));
		data.cellsAC2 = ArrayController<CudaCell>(static_cast<int>(NUM_CLUSTERS * 30 * 30 * 1.1));
//		data.cellsAC3 = ArrayController<CudaCell>(static_cast<int>(NUM_CLUSTERS * 30 * 30 * 1.1));


		size_t mapSize = size.x * size.y * sizeof(CudaCell*);
		cudaMallocManaged(&data.map1, mapSize);
		cudaMallocManaged(&data.map2, mapSize);
		checkCudaErrors(cudaGetLastError());

		cellsForAccess = (CudaCell*)malloc(sizeof(CudaCell) * static_cast<int>(NUM_CLUSTERS * 30 * 30 * 1.1));

		for (int i = 0; i < size.x * size.y; ++i) {
			data.map1[i] = nullptr;
			data.map2[i] = nullptr;
		}

	}

	~CudaSimulationManager()
	{
		data.clustersAC1.free();
		data.clustersAC2.free();
		data.cellsAC1.free();
		data.cellsAC2.free();

		cudaFree(data.map1);
		cudaFree(data.map2);

		free(cellsForAccess);
	}

	void swapData()
	{
		swap(data.clustersAC1, data.clustersAC2);
		swap(data.cellsAC1, data.cellsAC2);
		swap(data.map1, data.map2);
	}

	void prepareTargetData()
	{
		data.clustersAC2.reset();
		data.cellsAC2.reset();
	}

	CudaDataForAccess getDataForAccess()
	{
		CudaDataForAccess result;
		result.numCells = data.cellsAC2.getNumEntries();
		cudaMemcpy(cellsForAccess, data.cellsAC2.getEntireArray(), sizeof(CudaCell) * data.cellsAC2.getNumEntries(), cudaMemcpyDeviceToHost);
		result.cells = cellsForAccess;
		return result;
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
