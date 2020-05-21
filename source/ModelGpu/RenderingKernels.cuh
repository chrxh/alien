#pragma once

#include "ModelBasic/Colors.h"

#include "device_functions.h"
#include "sm_60_atomic_functions.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "Map.cuh"
#include "EntityFactory.cuh"
#include "CleanupKernels.cuh"

#include "SimulationData.cuh"

__global__ void clearImageMap(unsigned int* imageData, int size)
{
    auto const block = calcPartition(size, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = block.startIndex; index <= block.endIndex; ++index) {
        imageData[index] = 0xff00001b;
    }
}

__device__ __inline__ int mapUniversePosToImageIndex(int2 const& imageSize, int2 const& rectUpperLeft, int2 const& intPos)
{
    return (intPos.x - rectUpperLeft.x) + (intPos.y - rectUpperLeft.y) * imageSize.x;
}

__device__ __inline__ unsigned int calcColor(Cell* cell)
{
    unsigned char colorCode = cell->metadata.color;
    
    unsigned int result;
    switch (colorCode % 7)
    {
    case 0: {
        result = Const::IndividualCellColor1;
        break;
    }
    case 1: {
        result = Const::IndividualCellColor2;
        break;
    }
    case 2: {
        result = Const::IndividualCellColor3;
        break;
    }
    case 3: {
        result = Const::IndividualCellColor4;
        break;
    }
    case 4: {
        result = Const::IndividualCellColor5;
        break;
    }
    case 5: {
        result = Const::IndividualCellColor6;
        break;
    }
    case 6: {
        result = Const::IndividualCellColor7;
        break;
    }
    }

    auto factor = static_cast<int>(min(150.0f, cell->getEnergy() / 2 + 20.0f));
    auto r = ((result >> 16) & 0xff) * factor / 150;
    auto g = ((result >> 8) & 0xff) * factor / 150;
    auto b = (result & 0xff) * factor / 150;
    return 0xff000000 | (r << 16) | (g << 8) | b;
}

__device__ __inline__ unsigned int calcColor(Particle* particle)
{
    auto e = min(150, (static_cast<int>(particle->getEnergy()) + 10) * 5);
    return (e << 16) | 0xff000030;
}

__device__ __inline__ unsigned int calcColor(Token* token)
{
    return 0xffffffff;
}

__device__ __inline__ void addingColor(unsigned int& color, unsigned int const& colorToAdd)
{
    auto newColor = (color & 0xfefefe) + (colorToAdd & 0xfefefe);
    if ((newColor & 0x1000000) != 0) {
        newColor |= 0xff0000;
    }
    if ((newColor & 0x10000) != 0) {
        newColor |= 0xff00;
    }
    if ((newColor & 0x100) != 0) {
        newColor |= 0xff;
    }
    color = newColor | 0xff000000;
}

__device__ __inline__ void drawEntity(unsigned int* imageData, int2 const& imageSize, int const& index, unsigned int color)
{
    color = (color >> 1) & 0x7e7e7e;
    addingColor(imageData[index], color);

    color = (color >> 1) & 0x7e7e7e;
    addingColor(imageData[index - 1], color);
    addingColor(imageData[index + 1], color);
    addingColor(imageData[index - imageSize.x], color);
    addingColor(imageData[index + imageSize.x], color);
}

__global__ void drawClusters(
    int2 universeSize,
    int2 rectUpperLeft,
    int2 rectLowerRight,
    Array<Cluster*> clusters,
    unsigned int* imageData,
    int2 imageSize)
{
    auto const clusterBlock =
        calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);

    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {

        auto const& cluster = clusters.at(clusterIndex);
        if (nullptr == cluster) {
            continue;
        }

        __shared__ MapInfo map;
        if (0 == threadIdx.x) {
            map.init(universeSize);
        }
        __syncthreads();

        auto const cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto const& cell = cluster->cellPointers[cellIndex];
            auto intPos = toInt2(cell->absPos);
            map.mapPosCorrection(intPos);
            if (isContainedInRect(rectUpperLeft, rectLowerRight, intPos, 1)) {
                auto const index = mapUniversePosToImageIndex(imageSize, rectUpperLeft, intPos);
                auto const color = calcColor(cell);
                drawEntity(imageData, imageSize, index, color);
            }
        }
        __syncthreads();

        auto const tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (auto tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto const& token = cluster->tokenPointers[tokenIndex];
            auto const& cell = token->cell;
            auto intPos = toInt2(cell->absPos);
            map.mapPosCorrection(intPos);
            if (isContainedInRect(rectUpperLeft, rectLowerRight, intPos, 1)) {
                auto const index = mapUniversePosToImageIndex(imageSize, rectUpperLeft, intPos);
                auto const color = calcColor(token);
                drawEntity(imageData, imageSize, index, color);
            }
        }
        __syncthreads();
    }
}

__global__ void drawParticles(
    int2 universeSize,
    int2 rectUpperLeft,
    int2 rectLowerRight,
    Array<Particle*> particles,
    unsigned int* imageData,
    int2 imageSize)
{
    auto const particleBlock =
        calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = particles.at(index);
        auto intPos = toInt2(particle->absPos);
        if (isContainedInRect(rectUpperLeft, rectLowerRight, intPos, 1)) {
            auto const index = mapUniversePosToImageIndex(imageSize, rectUpperLeft, intPos);
            auto const color = calcColor(particle);
            drawEntity(imageData, imageSize, index, color);
        }
    }
}

__global__ void drawImage(int2 rectUpperLeft, int2 rectLowerRight, SimulationData data)
{
    int width = rectLowerRight.x - rectUpperLeft.x + 1;
    int height = rectLowerRight.y - rectUpperLeft.y + 1;
    int numPixels = width * height;

    KERNEL_CALL(clearImageMap, data.imageData, numPixels);
    KERNEL_CALL(drawClusters, data.size, rectUpperLeft, rectLowerRight, data.entities.clusterPointers, data.imageData, { width, height });
    if (data.entities.clusterFreezedPointers.getNumEntries() > 0) {
        KERNEL_CALL(drawClusters, data.size, rectUpperLeft, rectLowerRight, data.entities.clusterFreezedPointers, data.imageData, { width, height });
    }
    KERNEL_CALL(drawParticles, data.size, rectUpperLeft, rectLowerRight, data.entities.particlePointers, data.imageData, { width, height });
}
