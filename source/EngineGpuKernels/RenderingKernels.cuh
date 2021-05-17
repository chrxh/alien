#pragma once

#include "EngineInterface/Colors.h"
#include "EngineInterface/ZoomLevels.h"

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "CleanupKernels.cuh"
#include "EntityFactory.cuh"
#include "Map.cuh"
#include "SimulationData.cuh"
#include "cuda_runtime_api.h"
#include "sm_60_atomic_functions.h"

__global__ void clearImageMap(unsigned int* imageData, int2 size, int2 outsideRectUpperLeft, int2 outsideRectLowerRight)
{
    auto const block = calcPartition(size.x * size.y, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = block.startIndex; index <= block.endIndex; ++index) {
        auto x = index % size.x;
        auto y = index / size.x;
        if (x < outsideRectUpperLeft.x || y < outsideRectUpperLeft.y || x >= outsideRectLowerRight.x
            || y >= outsideRectLowerRight.y) {
            imageData[index] = Const::NothingnessColor;
        } else {
            imageData[index] = Const::SpaceColor;
        }
    }
}

__device__ __inline__ float2 mapUniversePosToVectorImagePos(float2 const& rectUpperLeft, float2 const& pos, float zoom)
{
    return float2{(pos.x - rectUpperLeft.x) * zoom, (pos.y - rectUpperLeft.y) * zoom};
}

__device__ __inline__ float2 mapUniversePosToPixelImagePos(float2 const& rectUpperLeft, float2 const& pos, float zoom)
{
    float2 offset{rectUpperLeft.x - toFloat(toInt(rectUpperLeft.x)), rectUpperLeft.y - toFloat(toInt(rectUpperLeft.y))};
    return {
        (toFloat(toInt(pos.x - toInt(rectUpperLeft.x))) - offset.x) * zoom,
        (toFloat(toInt(pos.y - toInt(rectUpperLeft.y))) - offset.y) * zoom};
}

__device__ __inline__ float3 calcColor(Cell* cell, bool selected)
{
    unsigned int cellColor;
    switch (cell->metadata.color % 7) {
    case 0: {
        cellColor = Const::IndividualCellColor1;
        break;
    }
    case 1: {
        cellColor = Const::IndividualCellColor2;
        break;
    }
    case 2: {
        cellColor = Const::IndividualCellColor3;
        break;
    }
    case 3: {
        cellColor = Const::IndividualCellColor4;
        break;
    }
    case 4: {
        cellColor = Const::IndividualCellColor5;
        break;
    }
    case 5: {
        cellColor = Const::IndividualCellColor6;
        break;
    }
    case 6: {
        cellColor = Const::IndividualCellColor7;
        break;
    }
    }

    float factor = min(100.0f, sqrt(cell->getEnergy()) * 5 + 20.0f) / 100.0f;
    if (!selected) {
        factor *= 0.5f;
    }

    return {
        toFloat((cellColor >> 16) & 0xff) / 256.0f * factor,
        toFloat((cellColor >> 8) & 0xff) / 256.0f * factor,
        toFloat(cellColor & 0xff) / 256.0f * factor};
}

__device__ __inline__ float3 calcColor(Particle* particle, bool selected)
{
    auto intensity = max(min((toInt(particle->getEnergy()) + 10) * 5, 150), 20) / 256.0f;
    if (!selected) {
        intensity *= 0.5f;
    }

    return {intensity, 0, 0.08f};

}

__device__ __inline__ float3 calcColor(Token* token, bool selected)
{
    return selected ? float3{1.0f, 1.0f, 1.0f} : float3{0.5f, 0.5f, 0.5f};
}

__device__ __inline__ void addingColor(unsigned int& pixel, float3 const& colorToAdd)
{
    unsigned int colorFlat =
        toInt(colorToAdd.z * 255.0f) << 16 | toInt(colorToAdd.y * 255.0f) << 8 | toInt(colorToAdd.x * 255.0f);

    auto newColor = (pixel & 0xfefefe) + (colorFlat & 0xfefefe);
    if ((newColor & 0x1000000) != 0) {
        newColor |= 0xff0000;
    }
    if ((newColor & 0x10000) != 0) {
        newColor |= 0xff00;
    }
    if ((newColor & 0x100) != 0) {
        newColor |= 0xff;
    }
    pixel = newColor | 0xff000000;
}

__device__ __inline__ void
drawDot(unsigned int* imageData, int2 const& imageSize, float2 const& pos, float3 const& colorToAdd)
{
    int2 intPos{toInt(pos.x), toInt(pos.y)};
    if (intPos.x >= 0 && intPos.x < imageSize.x - 1 && intPos.y >= 0 && intPos.y < imageSize.y - 1) {

        float2 posFrac{pos.x - intPos.x, pos.y - intPos.y};
        unsigned int index = intPos.x + intPos.y * imageSize.x;

        float3 colorToAdd1 = colorToAdd * (1.0f - posFrac.x) * (1.0f - posFrac.y);
        addingColor(imageData[index], colorToAdd1);

        float3 colorToAdd2 = colorToAdd * posFrac.x * (1.0f - posFrac.y);
        addingColor(imageData[index + 1], colorToAdd2);

        float3 colorToAdd3 = colorToAdd * (1.0f - posFrac.x) * posFrac.y;
        addingColor(imageData[index + imageSize.x], colorToAdd3);

        float3 colorToAdd4 = colorToAdd * posFrac.x * posFrac.y;
        addingColor(imageData[index + imageSize.x + 1], colorToAdd4);
    }
}

__device__ __inline__ void
drawCircle(unsigned int* imageData, int2 const& imageSize, float2 pos, float3 color, float radius)
{
    if (radius > 1.0 - FP_PRECISION) {
        auto radiusSquared = radius * radius;
        for (float x = -radius; x <= radius; x += 1.0f) {
            for (float y = -radius; y <= radius; y += 1.0f) {
                if (x * x + y * y <= radiusSquared) {
                    drawDot(imageData, imageSize, pos + float2{x, y}, color);
                }
            }
        }
    } else {
        color = color * radius * 2;
        drawDot(imageData, imageSize, pos, color);
        color = color * 0.3f;
        drawDot(imageData, imageSize, pos + float2{1, 0}, color);
        drawDot(imageData, imageSize, pos + float2{-1, 0}, color);
        drawDot(imageData, imageSize, pos + float2{0, 1}, color);
        drawDot(imageData, imageSize, pos + float2{0, -1}, color);
    }
}

__global__ void drawClusters(
    int2 universeSize,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    Array<Cluster*> clusters,
    unsigned int* imageData,
    int2 imageSize,
    float zoom)
{
    auto const clusterBlock = calcPartition(clusters.getNumEntries(), blockIdx.x, gridDim.x);

    for (int clusterIndex = clusterBlock.startIndex; clusterIndex <= clusterBlock.endIndex; ++clusterIndex) {

        auto const& cluster = clusters.at(clusterIndex);
        if (nullptr == cluster) {
            continue;
        }

        __shared__ MapInfo map;
        __shared__ bool isSelected;
        if (0 == threadIdx.x) {
            map.init(universeSize);
            isSelected = cluster->isSelected();
        }
        __syncthreads();

        auto const cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto const& cell = cluster->cellPointers[cellIndex];

            auto cellPos = cell->absPos;
            map.mapPosCorrection(cellPos);
            if (isContainedInRect(rectUpperLeft, rectLowerRight, cellPos)) {
                auto cellImagePos = mapUniversePosToVectorImagePos(rectUpperLeft, cellPos, zoom);
                auto color = calcColor(cell, isSelected);
                drawCircle(imageData, imageSize, cellImagePos, color, zoom / 3);

                auto posCorrection = cellPos - cell->absPos;
                if (zoom > 1 - FP_PRECISION) {
                    color = color * min((zoom - 1.0f), 1.0f);
                    for (int i = 0; i < cell->numConnections; ++i) {
                        auto const otherCell = cell->connections[i];
                        auto const otherCellPos = otherCell->absPos + posCorrection;
                        auto const otherCellImagePos =
                            mapUniversePosToVectorImagePos(rectUpperLeft, otherCellPos, zoom);
                        float dist = Math::length(otherCellImagePos - cellImagePos);
                        float2 const v = {
                            static_cast<float>(otherCellImagePos.x - cellImagePos.x) / dist * 1.8f,
                            static_cast<float>(otherCellImagePos.y - cellImagePos.y) / dist * 1.8f};
                        float2 pos = cellImagePos;


                        for (float d = 0; d <= dist; d += 1.8f) {
                            auto const intPos = toInt2(pos);
                            drawDot(imageData, imageSize, pos, color);
                            pos = pos + v;
                        }
                    }
                }
            }
        }
        __syncthreads();

        auto const tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (auto tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto const& token = cluster->tokenPointers[tokenIndex];
            auto const& cell = token->cell;

            auto cellPos = cell->absPos;
            map.mapPosCorrection(cellPos);
            auto const cellImagePos = mapUniversePosToVectorImagePos(rectUpperLeft, cellPos, zoom);
            if (isContainedInRect({0, 0}, imageSize, cellImagePos)) {
                auto const color = calcColor(token, cell->cluster->isSelected());
                drawCircle(imageData, imageSize, cellImagePos, color, zoom / 2);
            }
        }
        __syncthreads();
    }
}

__global__ void drawParticles(
    int2 universeSize,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    Array<Particle*> particles,
    unsigned int* imageData,
    int2 imageSize,
    float zoom)
{
    auto const particleBlock =
        calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = particles.at(index);

        auto const particleImagePos = mapUniversePosToVectorImagePos(rectUpperLeft, particle->absPos, zoom);
        if (isContainedInRect({0, 0}, imageSize, particleImagePos)) {
            auto const color = calcColor(particle, particle->isSelected());
            drawCircle(imageData, imageSize, particleImagePos, color, zoom / 4);
        }
    }
}


/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void drawImage(
    float2 rectUpperLeft,
    float2 rectLowerRight,
    int2 imageSize,
    float zoom,
    SimulationData data)
{
    unsigned int* targetImage = data.imageData;

    int2 outsideRectUpperLeft{-min(toInt(rectUpperLeft.x * zoom), 0), -min(toInt(rectUpperLeft.y * zoom), 0)};
    int2 outsideRectLowerRight{
        imageSize.x - max(toInt((rectLowerRight.x - data.size.x) * zoom), 0),
        imageSize.y - max(toInt((rectLowerRight.y - data.size.y) * zoom), 0)};
    KERNEL_CALL(clearImageMap, targetImage, imageSize, outsideRectUpperLeft, outsideRectLowerRight);
    KERNEL_CALL(
        drawClusters,
        data.size,
        rectUpperLeft,
        rectLowerRight,
        data.entities.clusterPointers,
        targetImage,
        imageSize,
        zoom);
    if (data.entities.clusterFreezedPointers.getNumEntries() > 0) {
        KERNEL_CALL(
            drawClusters,
            data.size,
            rectUpperLeft,
            rectLowerRight,
            data.entities.clusterFreezedPointers,
            targetImage,
            imageSize,
            zoom);
    }
    KERNEL_CALL(
        drawParticles,
        data.size,
        rectUpperLeft,
        rectLowerRight,
        data.entities.particlePointers,
        targetImage,
        imageSize,
        zoom);
}

