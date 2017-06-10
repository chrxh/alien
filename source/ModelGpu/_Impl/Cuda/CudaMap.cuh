#pragma once

#include "device_functions.h"

__device__ void mapPosCorrection_Kernel(int2 &pos, int2 const &size)
{
	pos = { ((pos.x % size.x) + size.x) % size.x, ((pos.y % size.y) + size.y) % size.y };
}

__device__ void mapPosCorrection_Kernel(double2 &pos, int2 const &size)
{
	int2 intPart{ (int)pos.x, (int)pos.y };
	double2 fracPart = { pos.x - intPart.x, pos.y - intPart.y };
	mapPosCorrection_Kernel(intPart, size);
	pos = { (double)intPart.x + fracPart.x, (double)intPart.y + fracPart.y };
}

__device__ void mapDisplacementCorrection_Kernel(double2 &disp, int2 const &size)
{
}

__device__ inline bool isCellPresentAtMap_Kernel(int2 posInt, CellCuda *cell, CellCuda ** __restrict__ map, int2 const &size)
{
	mapPosCorrection_Kernel(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	return map[mapEntry] == cell;

}

__device__ void inline setCellToMap_Kernel(int2 const &posInt, CellCuda *cell, CellCuda ** __restrict__ map, int2 const &size)
{
	auto mapEntry = posInt.x + posInt.y * size.x;
	map[mapEntry] = cell;
}

__device__ inline double mapDistanceSquared_Kernel(double2 const &p, double2 const &q, int2 const &size)
{
	double2 d = { p.x - q.x, p.y - q.y };
	mapDisplacementCorrection_Kernel(d, size);
	return d.x*d.x + d.y*d.y;
}
