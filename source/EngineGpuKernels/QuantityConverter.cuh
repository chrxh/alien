#pragma once

#include "Base.cuh"

class QuantityConverter
{
public:
    //Notice: all angles below are in DEG
    __inline__ __device__ static float convertDataToAngle(unsigned char b);
    __inline__ __device__ static unsigned char convertAngleToData(float a);
    __inline__ __device__ static float convertDataToDistance(unsigned char b);
    __inline__ __device__ static unsigned char convertDistanceToData(float len);
    __inline__ __device__ static unsigned char convertURealToData(float r);
    __inline__ __device__ static float convertDataToUReal(unsigned char d);
    __inline__ __device__ static unsigned char convertIntToData(int i);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ float QuantityConverter::convertDataToAngle(unsigned char b)
{
    //0 to 127 => 0 to 179 degree
    //128 to 255 => -179 to 0 degree
    if (b < 128) {
        return (0.5f + static_cast<float>(b))*(180.0f / 128.0f);
    }
    else {
        return (-256.0f - 0.5f + static_cast<float>(b))*(180.0f / 128.0f);
    }
}

__inline__ __device__ unsigned char QuantityConverter::convertAngleToData(float a)
{
    //0 to 180 degree => 0 to 128
    //-180 to 0 degree => 128 to 256 (= 0)
    a = remainderf(remainderf(a, 360.0f) + 360.0f, 360.0f);    //get angle between 0 and 360
    if (a > 180.0f) {
        a -= 360.0f;
    }
    int result = static_cast<int>(a * 128.0f / 180.0f);
    return static_cast<unsigned char>(result);

}

__inline__ __device__ float QuantityConverter::convertDataToDistance(unsigned char b)
{
    return (0.5f + static_cast<float>(b)) / 100.0f;

}

__inline__ __device__ unsigned char QuantityConverter::convertDistanceToData(float len)
{
    if (static_cast<uint32_t>(len*100.0f) >= 256) {
        return 255;
    }
    return static_cast<unsigned char>(len*100.0f);
}

__inline__ __device__ unsigned char QuantityConverter::convertURealToData(float r)
{
    if (r < 0.0f) {
        return 0;
    }
    if (r > 127.0f) {
        return 127;
    }
    return floorInt(r);
}

__inline__ __device__ float QuantityConverter::convertDataToUReal(unsigned char d)
{
    return static_cast<float>(d);
}

__inline__ __device__ unsigned char QuantityConverter::convertIntToData(int i)
{
    if (i > 127) {
        return i;
    }
    if (i < -128) {
        return static_cast<unsigned char>(-128);
    }
    return i;
}
