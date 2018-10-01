#pragma once

#include "device_functions.h"

__host__ __device__ void mapPosCorrection(int2 &pos, int2 const &size)
{
	pos = { ((pos.x % size.x) + size.x) % size.x, ((pos.y % size.y) + size.y) % size.y };
}

__host__ __device__ void mapPosCorrection(float2 &pos, int2 const &size)
{
	int2 intPart{ (int)pos.x, (int)pos.y };
	float2 fracPart = { pos.x - intPart.x, pos.y - intPart.y };
	mapPosCorrection(intPart, size);
	pos = { static_cast<float>(intPart.x) + fracPart.x, static_cast<float>(intPart.y) + fracPart.y };
}

__device__ void angleCorrection(int &angle)
{
	angle = ((angle % 360) + 360) % 360;
}

__device__ void angleCorrection(float &angle)
{
	int intPart = (int)angle;
	float fracPart = angle - intPart;
	angleCorrection(intPart);
	angle = (float)intPart + fracPart;
}

__device__ void mapDisplacementCorrection(float2 &disp, int2 const &size)
{
}

__device__ float mapDistanceSquared(float2 const &p, float2 const &q, int2 const &size)
{
	float2 d = { p.x - q.x, p.y - q.y };
	mapDisplacementCorrection(d, size);
	return d.x*d.x + d.y*d.y;
}

__device__ bool isCellPresentAtMap(int2 posInt, CellData * cell, CellData ** map, int2 const &size)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	return map[mapEntry] == cell;

}

template<typename T>
__host__ __device__ CellData* getFromMap(int2 posInt, T ** __restrict__ map, int2 const &size)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	return map[mapEntry];
}

template<typename T>
__host__ __device__ void setToMap(int2 posInt, T * __restrict__ entity, T ** __restrict__ map, int2 const &size)
{
	mapPosCorrection(posInt, size);
	auto mapEntry = posInt.x + posInt.y * size.x;
	map[mapEntry] = entity;
}

