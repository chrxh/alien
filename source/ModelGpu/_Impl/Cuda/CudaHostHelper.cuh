#pragma once

#include "CudaShared.cuh"

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

bool isClusterPositionFree(CudaCellCluster* cluster, CudaData* data)
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

void drawClusterToMap(CudaCellCluster* cluster, CudaData* data)
{
	for (int i = 0; i < cluster->numCells; ++i) {
		auto &absPos = cluster->cells[i].absPos;
		setCellToMap({ static_cast<int>(absPos.x),static_cast<int>(absPos.y) }, &cluster->cells[i], data->map1, data->size);
	}
}
