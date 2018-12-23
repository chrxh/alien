#pragma once

#include "device_functions.h"

class BasicMap
{
public:
	__inline__ __host__ __device__ void init(int2 const& size)
	{
		_size = size;
	}

	__inline__ __host__ __device__ void mapPosCorrection(int2 &pos) const
	{
		pos = { ((pos.x % _size.x) + _size.x) % _size.x, ((pos.y % _size.y) + _size.y) % _size.y };
	}

	__inline__ __host__ __device__ void mapPosCorrection(float2 &pos) const
	{
		int2 intPart{ (int)pos.x, (int)pos.y };
		float2 fracPart = { pos.x - intPart.x, pos.y - intPart.y };
		mapPosCorrection(intPart);
		pos = { static_cast<float>(intPart.x) + fracPart.x, static_cast<float>(intPart.y) + fracPart.y };
	}

	__inline__ __device__ void mapDisplacementCorrection(float2 &disp) const
	{
	}

	__inline__ __device__ float mapDistanceSquared(float2 const &p, float2 const &q) const
	{
		float2 d = { p.x - q.x, p.y - q.y };
		mapDisplacementCorrection(d);
		return d.x*d.x + d.y*d.y;
	}

protected:
	int2 _size;
};

template<typename T>
class Map
	: public BasicMap
{
public:
	__inline__ __host__ __device__ void init(int2 const& size, T ** map1, T ** map2)
	{
		BasicMap::init(size);
		_map1 = map1;
		_map2 = map2;
	}

	__inline__ __host__ __device__ bool isEntityPresentAtOrigMap(int2 posInt, T* entity) const
	{
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * BasicMap::_size.x;
		return _map1[mapEntry] == entity;	

	}

	__inline__ __host__ __device__ T* getFromOrigMap(int2 posInt) const
	{
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * _size.x;
		return _map1[mapEntry];
	}

	__inline__ __host__ __device__ T* getFromNewMap(int2 posInt) const
	{
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * _size.x;
		return _map2[mapEntry];
	}

	__inline__ __host__ __device__ void setToOrigMap(int2 posInt, T* entity)
	{
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * _size.x;
		_map1[mapEntry] = entity;
	}

	__inline__ __host__ __device__ void setToNewMap(int2 posInt, T* entity)
	{
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * _size.x;
		_map2[mapEntry] = entity;
	}

private:
	T ** _map1;
	T ** _map2;
};
