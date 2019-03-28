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
		int2 intPart{ floorInt(pos.x), floorInt(pos.y) };
		float2 fracPart = { pos.x - intPart.x, pos.y - intPart.y };
		mapPosCorrection(intPart);
		pos = { static_cast<float>(intPart.x) + fracPart.x, static_cast<float>(intPart.y) + fracPart.y };
	}

	__inline__ __device__ void mapDisplacementCorrection(float2 &disp) const
	{
		disp.x = remainderf(disp.x, _size.x / 2);
		disp.y = remainderf(disp.y, _size.y / 2);
	}

	__inline__ __device__ float mapDistance(float2 const &p, float2 const &q) const
	{
		float2 d = { p.x - q.x, p.y - q.y };
		mapDisplacementCorrection(d);
		return sqrt(d.x*d.x + d.y*d.y);
	}

	__inline__ __device__ float2 correctionIncrement(float2 pos1, float2 pos2) const
	{
		float2 result{ 0.0f, 0.0f };
		if (pos2.x - pos1.x > _size.x / 2) {
			result.x = -_size.x;
		}
		if (pos1.x - pos2.x > _size.x / 2) {
			result.x = _size.x;
		}
		if (pos2.y - pos1.y > _size.y / 2) {
			result.y = -_size.y;
		}
		if (pos1.y - pos2.y > _size.y / 2) {
			result.y = _size.y;
		}
		return result;
	}


protected:
	int2 _size;
};

template<typename T>
class Map
	: public BasicMap
{
public:
	__inline__ __host__ __device__ void init(int2 const& size, T ** map)
	{
		BasicMap::init(size);
		_map = map;
	}

	__inline__ __host__ __device__ bool isEntityPresent(float2 const& pos, T* entity) const
	{
		int2 posInt = { floorInt(pos.x), floorInt(pos.y) };
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * BasicMap::_size.x;
		return _map[mapEntry] == entity;
	}

	__inline__ __host__ __device__ T* get(float2 const& pos) const
	{
		int2 posInt = { floorInt(pos.x), floorInt(pos.y) };
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * _size.x;
		return _map[mapEntry];
	}

	__inline__ __host__ __device__ void set(float2 const& pos, T* entity)
	{
		int2 posInt = { floorInt(pos.x), floorInt(pos.y) };
		mapPosCorrection(posInt);
		auto mapEntry = posInt.x + posInt.y * _size.x;
		_map[mapEntry] = entity;
	}

private:
	T ** _map;
};

template<typename T>
class BiMap
{
public:
	__inline__ __host__ __device__ void init(int2 const& size, T ** map1, T ** map2)
	{
		_map1.init(size, map1);
		_map2.init(size, map2);
	}

	__inline__ __host__ __device__ bool isEntityPresentAtOrigMap(float2 const& pos, T* entity) const
	{
		return _map1.isEntityPresent(pos, entity);

	}

	__inline__ __host__ __device__ T* getFromOrigMap(float2 const& pos) const
	{
		return _map1.get(pos);
	}

	__inline__ __host__ __device__ T* getFromNewMap(float2 const& pos) const
	{
		return _map2.get(pos);
	}

	__inline__ __host__ __device__ void setToOrigMap(float2 const& pos, T* entity)
	{
		_map1.setToNewMap(pos, entity);
	}

	__inline__ __host__ __device__ void setToNewMap(float2 const& pos, T* entity)
	{
		_map2.setToNewMap(pos, entity);
	}

private:
	Map<T> _map1;
	Map<T> _map2;
};
