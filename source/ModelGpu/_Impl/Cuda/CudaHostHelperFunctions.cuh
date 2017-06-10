#pragma once

#include "CudaShared.cuh"

void updateAngularMass(ClusterCuda* cluster)
{
	cluster->angularMass = 0.0;
	for (int i = 0; i < cluster->numCells; ++i) {
		auto relPos = cluster->cells[i].relPos;
		cluster->angularMass += relPos.x*relPos.x + relPos.y*relPos.y;
	}
}

void updateRelPos(ClusterCuda* cluster)
{
	double2 center = { 0.0, 0.0 };
	for (int i = 0; i < cluster->numCells; ++i) {
		auto const &relPos = cluster->cells[i].relPos;
		center.x += relPos.x;
		center.y += relPos.y;
	}
	center.x /= static_cast<double>(cluster->numCells);
	center.y /= static_cast<double>(cluster->numCells);
	for (int i = 0; i < cluster->numCells; ++i) {
		auto &relPos = cluster->cells[i].relPos;
		relPos.x -= center.x;
		relPos.y -= center.y;
	}
}

void updateAbsPos(CellCuda *cell)
{

	double rotMatrix[2][2];
	ClusterCuda *cluster = cell->cluster;
	double sinAngle = sin(cluster->angle*DEG_TO_RAD);
	double cosAngle = cos(cluster->angle*DEG_TO_RAD);
	rotMatrix[0][0] = cosAngle;
	rotMatrix[0][1] = sinAngle;
	rotMatrix[1][0] = -sinAngle;
	rotMatrix[1][1] = cosAngle;
	for (int i = 0; i < cluster->numCells; ++i) {
		auto &relPos = cluster->cells[i].relPos;
		auto &absPos = cluster->cells[i].absPos;
		absPos.x = relPos.x*rotMatrix[0][0] + relPos.y*rotMatrix[0][1] + cluster->pos.x;
		absPos.y = relPos.x*rotMatrix[1][0] + relPos.y*rotMatrix[1][1] + cluster->pos.y;
	};
}
