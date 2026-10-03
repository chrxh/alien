#pragma once

#include "Base.cuh"
#include "Math.cuh"

// The world is a torus: positions are wrapped at the world boundaries and directions follow the shortest way
class WorldGeometry
{
public:
    __inline__ __host__ __device__ void init(int2 const& size)
    {
        _size = size;
        _sizeFloat = {toFloat(size.x), toFloat(size.y)};
        _invSizeFloat = {1.0f / _sizeFloat.x, 1.0f / _sizeFloat.y};
    }

    __inline__ __host__ __device__ int2 getSize() const { return _size; }

    __inline__ __host__ __device__ void correctPosition(int2& pos) const { pos = {wrapCoordinate(pos.x, _size.x), wrapCoordinate(pos.y, _size.y)}; }

    __inline__ __host__ __device__ void correctPosition(float2& pos) const
    {
        int2 intPart{floorInt(pos.x), floorInt(pos.y)};
        float2 fracPart = {pos.x - toFloat(intPart.x), pos.y - toFloat(intPart.y)};
        correctPosition(intPart);
        pos = {static_cast<float>(intPart.x) + fracPart.x, static_cast<float>(intPart.y) + fracPart.y};
    }

    __inline__ __device__ float2 getCorrectedPosition(float2 const& pos) const
    {
        auto copy = pos;
        correctPosition(copy);
        return copy;
    }

    __inline__ __device__ void correctDirection(float2& disp) const
    {
        disp.x = wrapDisplacement(disp.x, _sizeFloat.x, _invSizeFloat.x);
        disp.y = wrapDisplacement(disp.y, _sizeFloat.y, _invSizeFloat.y);
    }

    __inline__ __device__ float2 getCorrectedDirection(float2 const& disp) const
    {
        return {wrapDisplacement(disp.x, _sizeFloat.x, _invSizeFloat.x), wrapDisplacement(disp.y, _sizeFloat.y, _invSizeFloat.y)};
    }

    __inline__ __device__ float getDistance(float2 const& p, float2 const& q) const
    {
        float2 d = {p.x - q.x, p.y - q.y};
        correctDirection(d);
        return sqrt(d.x * d.x + d.y * d.y);
    }

    __inline__ __device__ float2 getCorrectionIncrement(float2 pos1, float2 pos2) const
    {
        auto delta = pos1 - pos2 + toFloat2(_size) / 2;
        return {delta.x - Math::modulo(delta.x, toFloat(_size.x)), delta.y - Math::modulo(delta.y, toFloat(_size.y))};
    }

    __inline__ __device__ int getMaxRadius() const { return min(_size.x, _size.y) / 4; }

private:
    __inline__ __host__ __device__ static int wrapCoordinate(int value, int size)
    {
        if (value < 0) {
            value += size;
            return value >= 0 ? value : ((value % size) + size) % size;
        }
        if (value >= size) {
            value -= size;
            return value < size ? value : value % size;
        }
        return value;
    }

    // Minimum image convention, equivalent to remainderf(disp, size)
    __inline__ __device__ static float wrapDisplacement(float disp, float size, float invSize) { return fmaf(-size, rintf(disp * invSize), disp); }

    int2 _size;
    float2 _sizeFloat;
    float2 _invSizeFloat;
};
