#pragma once

#include "device_functions.h"

__host__ __device__ void mapPosCorrection(int2 &pos, int2 const &size)
{
	pos = { ((pos.x % size.x) + size.x) % size.x, ((pos.y % size.y) + size.y) % size.y };
}

__host__ __device__ void mapPosCorrection(double2 &pos, int2 const &size)
{
	int2 intPart{ (int)pos.x, (int)pos.y };
	double2 fracPart = { pos.x - intPart.x, pos.y - intPart.y };
	mapPosCorrection(intPart, size);
	pos = { static_cast<double>(intPart.x) + fracPart.x, static_cast<double>(intPart.y) + fracPart.y };
}

__device__ void mapDisplacementCorrection_Kernel(double2 &disp, int2 const &size)
{
}

__device__ inline bool isCellPresentAtMap_Kernel(int2 posInt, CellCuda * __restrict__ cell, CellCuda ** __restrict__ map, int2 const &size)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	return map[mapEntry] == cell;

}

__host__ __device__ CellCuda* getCellFromMap(int2 posInt, CellCuda ** __restrict__ map, int2 const &size)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	return map[mapEntry];
}

__host__ __device__ void inline setCellToMap(int2 posInt, CellCuda * __restrict__ cell, CellCuda ** __restrict__ map, int2 const &size)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	map[mapEntry] = cell;
}

__device__ inline double mapDistanceSquared_Kernel(double2 const &p, double2 const &q, int2 const &size)
{
	double2 d = { p.x - q.x, p.y - q.y };
	mapDisplacementCorrection_Kernel(d, size);
	return d.x*d.x + d.y*d.y;
}
