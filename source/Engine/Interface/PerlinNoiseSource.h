#pragma once

// Perlin noise shared by the CUDA force field kernel and the GLSL location shader, so that the rendered
// background matches the height map that drives the force field. The code is written in the common subset
// of CUDA C++ and GLSL and is used in two ways:
//   PERLIN_NOISE_SOURCE(FUNC) expands to the function definitions, where FUNC is the function qualifier
//   (__device__ __inline__ in CUDA, empty in GLSL).
//   PERLIN_NOISE_GLSL yields the same definitions as a string literal that can be concatenated into a
//   shader source.
#define PERLIN_NOISE_SOURCE(FUNC) \
    FUNC int perlinHash(int x) \
    { \
        x = ((x >> 16) ^ x) * 0x45d9f3b; \
        x = ((x >> 16) ^ x) * 0x45d9f3b; \
        x = (x >> 16) ^ x; \
        return x; \
    } \
    FUNC float perlinGradientDot(int ix, int iy, int seed, float dx, float dy) \
    { \
        int gradientIndex = perlinHash(perlinHash(ix) + iy + seed) & 3; \
        if (gradientIndex == 0) { \
            return dx; \
        } \
        if (gradientIndex == 1) { \
            return -dx; \
        } \
        if (gradientIndex == 2) { \
            return dy; \
        } \
        return -dy; \
    } \
    FUNC float perlinFade(float t) \
    { \
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); \
    } \
    FUNC float perlinNoise(float x, float y, int seed) \
    { \
        int ix = int(floor(x)); \
        int iy = int(floor(y)); \
        float fx = x - float(ix); \
        float fy = y - float(iy); \
        float u = perlinFade(fx); \
        float v = perlinFade(fy); \
        float n00 = perlinGradientDot(ix, iy, seed, fx, fy); \
        float n10 = perlinGradientDot(ix + 1, iy, seed, fx - 1.0f, fy); \
        float n01 = perlinGradientDot(ix, iy + 1, seed, fx, fy - 1.0f); \
        float n11 = perlinGradientDot(ix + 1, iy + 1, seed, fx - 1.0f, fy - 1.0f); \
        float nx0 = n00 + u * (n10 - n00); \
        float nx1 = n01 + u * (n11 - n01); \
        return nx0 + v * (nx1 - nx0); \
    }

#define PERLIN_NOISE_STRINGIFY_IMPL(source) #source
#define PERLIN_NOISE_STRINGIFY(source) PERLIN_NOISE_STRINGIFY_IMPL(source)
#define PERLIN_NOISE_GLSL PERLIN_NOISE_STRINGIFY(PERLIN_NOISE_SOURCE())
