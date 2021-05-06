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

__global__ void clearImageMap(unsigned int* imageData, int size)
{
    auto const block = calcPartition(size, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = block.startIndex; index <= block.endIndex; ++index) {
        imageData[index] = 0xff1b0000;
    }
}

__device__ __inline__ int2 mapUniversePosToVectorImagePos(float2 const& rectUpperLeft, float2 const& pos, float zoom)
{
    return {toInt((pos.x - rectUpperLeft.x) * zoom), toInt((pos.y - rectUpperLeft.y) * zoom)};
}

__device__ __inline__ float2 mapUniversePosToPixelImagePos(float2 const& rectUpperLeft, float2 const& pos, float zoom)
{
    float2 offset{rectUpperLeft.x - toFloat(toInt(rectUpperLeft.x)), rectUpperLeft.y - toFloat(toInt(rectUpperLeft.y))};
    return {
        (toFloat(toInt(pos.x - toInt(rectUpperLeft.x))) - offset.x) * zoom,
        (toFloat(toInt(pos.y - toInt(rectUpperLeft.y))) - offset.y) * zoom};
}

__device__ __inline__ unsigned int calcColor(Cell* cell, bool selected)
{
    unsigned char colorCode = cell->metadata.color;

    unsigned int cellColor;
    switch (colorCode % 7) {
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

    auto factor = static_cast<int>(min(150.0f, cell->getEnergy() / 2 + 20.0f));
    auto r = ((cellColor >> 16) & 0xff) * factor / 150;
    auto g = ((cellColor >> 8) & 0xff) * factor / 150;
    auto b = (cellColor & 0xff) * factor / 150;

    auto result = (b << 16) | (g << 8) | r;

    if (!selected) {
        result = (result >> 1) & 0x7e7e7e;
    }

    return 0xff000000 | result;
}

__device__ __inline__ unsigned int calcColor(Particle* particle, bool selected)
{
    auto result = min(150, (static_cast<int>(particle->getEnergy()) + 10) * 5);

    if (!selected) {
        result = (result >> 1) & 0x7e7e7e;
    }

    return result | 0xff130013;
}

__device__ __inline__ unsigned int calcColor(Token* token, bool selected)
{
    return selected ? 0xffffffff : 0xff7f7f7f;
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

__device__ __inline__ void drawDot(
    unsigned int* imageData,
    int2 const& imageSize,
    int const& index,
    unsigned int color,
    bool darken = false)
{
    if (darken) {
        color = (color >> 1) & 0x7e7e7e;
    }
    addingColor(imageData[index], color);

    color = (color >> 1) & 0x7e7e7e;
    addingColor(imageData[index - 1], color);
    addingColor(imageData[index + 1], color);
    addingColor(imageData[index - imageSize.x], color);
    addingColor(imageData[index + imageSize.x], color);
}

__device__ __inline__ void
drawCircle(unsigned int* imageData, int2 const& imageSize, int index, unsigned int color, bool darken)
{
    drawDot(imageData, imageSize, index, color, darken);

    index -= 1 + 2 * imageSize.x;
    drawDot(imageData, imageSize, index, color, darken);
    ++index;
    drawDot(imageData, imageSize, index, color, darken);
    ++index;
    drawDot(imageData, imageSize, index, color, darken);

    index += 1 + imageSize.x;
    drawDot(imageData, imageSize, index, color, darken);
    index += imageSize.x;
    drawDot(imageData, imageSize, index, color, darken);
    index += imageSize.x;
    drawDot(imageData, imageSize, index, color, darken);

    index += imageSize.x - 1;
    drawDot(imageData, imageSize, index, color, darken);
    --index;
    drawDot(imageData, imageSize, index, color, darken);
    --index;
    drawDot(imageData, imageSize, index, color, darken);

    index -= 1 + imageSize.x;
    drawDot(imageData, imageSize, index, color, darken);
    index -= imageSize.x;
    drawDot(imageData, imageSize, index, color, darken);
    index -= imageSize.x;
    drawDot(imageData, imageSize, index, color, darken);
}

__global__ void drawClusters_vectorStyle(
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
        __shared__ bool isLowZoom;
        __shared__ bool isHighZoom;
        if (0 == threadIdx.x) {
            map.init(universeSize);
            isSelected = cluster->isSelected();
            isLowZoom = zoom < 4.1;
            isHighZoom = zoom > 8.1;
        }
        __syncthreads();

        auto const cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto const& cell = cluster->cellPointers[cellIndex];

            auto cellPos = cell->absPos;
            map.mapPosCorrection(cellPos);
            if (isContainedInRect(rectUpperLeft, rectLowerRight, cellPos)) {
                auto const cellImagePos = mapUniversePosToVectorImagePos(rectUpperLeft, cellPos, zoom);
                auto const color = calcColor(cell, isSelected);
                if (isContainedInRect({0, 0}, imageSize, cellImagePos, 3)) {
                    auto index = cellImagePos.x + cellImagePos.y * imageSize.x;

                    if (isHighZoom) {
                        drawCircle(imageData, imageSize, index, color, false);
                    } else {
                        --index;
                        drawDot(imageData, imageSize, index, color, isLowZoom);
                        index += 2;
                        drawDot(imageData, imageSize, index, color, isLowZoom);
                        index -= 1 + imageSize.x;
                        drawDot(imageData, imageSize, index, color, isLowZoom);
                        index += 2 * imageSize.x;
                        drawDot(imageData, imageSize, index, color, isLowZoom);
                    }
                }

                auto const posCorrection = cellPos - cell->absPos;

                for (int i = 0; i < cell->numConnections; ++i) {
                    auto const otherCell = cell->connections[i];
                    auto const otherCellPos = otherCell->absPos + posCorrection;
                    auto const otherCellImagePos = mapUniversePosToVectorImagePos(rectUpperLeft, otherCellPos, zoom);
                    float dist = Math::length(otherCellImagePos - cellImagePos);
                    float2 const v = {
                        static_cast<float>(otherCellImagePos.x - cellImagePos.x) / dist * 1.8f,
                        static_cast<float>(otherCellImagePos.y - cellImagePos.y) / dist * 1.8f};
                    float2 pos = toFloat2(cellImagePos);

                    for (float d = 0; d <= dist; d += 1.8f) {
                        auto const intPos = toInt2(pos);
                        if (isContainedInRect({0, 0}, imageSize, intPos, 2)) {
                            auto const index = intPos.x + intPos.y * imageSize.x;
                            drawDot(imageData, imageSize, index, color, true);
                        }
                        pos = pos + v;
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
            if (isContainedInRect({0, 0}, imageSize, cellImagePos, 4)) {
                auto index = cellImagePos.x + cellImagePos.y * imageSize.x;
                auto const color = calcColor(token, cell->cluster->isSelected());

                drawCircle(imageData, imageSize, index, color, isLowZoom);
            }
        }
        __syncthreads();
    }
}

__global__ void drawParticles_vectorStyle(
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
        if (isContainedInRect({0, 0}, imageSize, particleImagePos, 4)) {
            auto index = particleImagePos.x + particleImagePos.y * imageSize.x;
            auto const color = calcColor(particle, particle->isSelected());
            drawCircle(imageData, imageSize, index, color, zoom < 4.1);
        }
    }
}


__device__ __inline__ void drawPixel(
    unsigned int* imageData,
    int2 const& imageSize,
    float2 const& pos,
    unsigned int color,
    float zoom)
{
    int zoomIntPart = toInt(zoom);
    float zoomFracPart = zoom - toFloat(zoomIntPart);

    auto r = toFloat(color & 0xff);
    auto g = toFloat((color >> 8) & 0xff);
    auto b = toFloat((color >> 16) & 0xff);
 
    auto intPosX = toInt(pos.x - zoom / 2.0f);
    auto intPosY = toInt(pos.y - zoom / 2.0f);
    auto posFracX = pos.x - zoom / 2.0f - intPosX;
    auto posFracY = pos.y - zoom / 2.0f - intPosY;
    
    auto factor1 = (1.0f - posFracX) * (1.0f - posFracY);
    auto color1 = 0xff000000 | toInt(r * factor1) | (toInt(g * factor1) << 8) | (toInt(b * factor1) << 16);
    auto colorFrac1 = 0xff000000 | toInt(r * factor1 * zoomFracPart) | (toInt(g * factor1 * zoomFracPart) << 8)
        | (toInt(b * factor1 * zoomFracPart) << 16);
    
    auto factor2 = posFracX * (1.0f - posFracY);
    auto color2 = 0xff000000 | toInt(r * factor2) | (toInt(g * factor2) << 8) | (toInt(b * factor2) << 16);
    auto colorFrac2 = 0xff000000 | toInt(r * factor2 * zoomFracPart) | (toInt(g * factor2 * zoomFracPart) << 8)
        | (toInt(b * factor2 * zoomFracPart) << 16);
    
    auto factor3 = (1.0f - posFracX) * posFracY;
    auto color3 = 0xff000000 | toInt(r * factor3) | (toInt(g * factor3) << 8) | (toInt(b * factor3) << 16);
    auto colorFrac3 = 0xff000000 | toInt(r * factor3 * zoomFracPart) | (toInt(g * factor3 * zoomFracPart) << 8)
        | (toInt(b * factor3 * zoomFracPart) << 16);

    auto factor4 = posFracX * posFracY;
    auto color4 = 0xff000000 | toInt(r * factor4) | (toInt(g * factor4) << 8) | (toInt(b * factor4) << 16);
    auto colorFrac4 = 0xff000000 | toInt(r * factor4 * zoomFracPart) | (toInt(g * factor4 * zoomFracPart) << 8)
        | (toInt(b * factor4 * zoomFracPart) << 16);

    for (int x = 0; x <= zoomIntPart + 1; ++x) {
        for (int y = 0; y <= zoomIntPart + 1; ++y) {
            auto resultPosX = intPosX + x;
            auto resultPosY = intPosY + y;
            if (resultPosX >= 5 && resultPosX < imageSize.x - 1 && resultPosY >= 5 && resultPosY < imageSize.y - 1) {
                auto resultIndex = resultPosX + resultPosY * imageSize.x;
                if (x == zoomIntPart + 1 || y == zoomIntPart + 1) {
                    addingColor(imageData[resultIndex], colorFrac1);
                    addingColor(imageData[resultIndex + 1], colorFrac2);
                    addingColor(imageData[resultIndex + imageSize.x], colorFrac3);
                    addingColor(imageData[resultIndex + imageSize.x + 1], colorFrac4);
                } else {
                    addingColor(imageData[resultIndex], color1);
                    addingColor(imageData[resultIndex + 1], color2);
                    addingColor(imageData[resultIndex + imageSize.x], color3);
                    addingColor(imageData[resultIndex + imageSize.x + 1], color4);
                }
            }
        }
    }
}

__global__ void drawClusters_pixelStyle(
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
        if (0 == threadIdx.x) {
            map.init(universeSize);
        }
        __syncthreads();

        auto const cellBlock = calcPartition(cluster->numCellPointers, threadIdx.x, blockDim.x);
        for (auto cellIndex = cellBlock.startIndex; cellIndex <= cellBlock.endIndex; ++cellIndex) {
            auto const& cell = cluster->cellPointers[cellIndex];
            auto pos = cell->absPos;
            map.mapPosCorrection(pos);
            auto imagePos = mapUniversePosToPixelImagePos(rectUpperLeft, pos, zoom);
            if (isContainedInRect({0, 0}, imageSize, imagePos)) {
                auto color = calcColor(cell, cell->cluster->isSelected());
                drawPixel(imageData, imageSize, imagePos, color, zoom);
            }
        }
        __syncthreads();

        auto const tokenBlock = calcPartition(cluster->numTokenPointers, threadIdx.x, blockDim.x);
        for (auto tokenIndex = tokenBlock.startIndex; tokenIndex <= tokenBlock.endIndex; ++tokenIndex) {
            auto const& token = cluster->tokenPointers[tokenIndex];
            auto const& cell = token->cell;
            auto pos = cell->absPos;
            map.mapPosCorrection(pos);
            auto imagePos = mapUniversePosToPixelImagePos(rectUpperLeft, pos, zoom);
            if (isContainedInRect({0, 0}, imageSize, imagePos)) {
                auto color = calcColor(token, cell->cluster->isSelected());
                drawPixel(imageData, imageSize, imagePos, color, zoom);
            }
        }
        __syncthreads();
    }
}


__global__ void drawParticles_pixelStyle(
    int2 universeSize,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    Array<Particle*> particles,
    unsigned int* imageData,
    int2 imageSize,
    float zoom)
{
    auto particleBlock =
        calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    for (int index = particleBlock.startIndex; index <= particleBlock.endIndex; ++index) {
        auto const& particle = particles.at(index);

        auto particleImagePos = mapUniversePosToPixelImagePos(rectUpperLeft, particle->absPos, zoom);
        if (isContainedInRect({0, 0}, imageSize, particleImagePos)) {
            auto color = calcColor(particle, particle->isSelected());
            drawPixel(imageData, imageSize, particleImagePos, color, zoom);
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

    KERNEL_CALL(clearImageMap, targetImage, imageSize.x * imageSize.y);
    if (zoom > Const::ZoomLevelForAutomaticVectorViewSwitch) {
        KERNEL_CALL(
            drawClusters_vectorStyle,
            data.size,
            rectUpperLeft,
            rectLowerRight,
            data.entities.clusterPointers,
            targetImage,
            imageSize,
            zoom);
        if (data.entities.clusterFreezedPointers.getNumEntries() > 0) {
            KERNEL_CALL(
                drawClusters_vectorStyle,
                data.size,
                rectUpperLeft,
                rectLowerRight,
                data.entities.clusterFreezedPointers,
                targetImage,
                imageSize,
                zoom);
        }
        KERNEL_CALL(
            drawParticles_vectorStyle,
            data.size,
            rectUpperLeft,
            rectLowerRight,
            data.entities.particlePointers,
            targetImage,
            imageSize,
            zoom);
    } else {
        KERNEL_CALL(
            drawClusters_pixelStyle,
            data.size,
            rectUpperLeft,
            rectLowerRight,
            data.entities.clusterPointers,
            targetImage,
            imageSize,
            zoom);
        if (data.entities.clusterFreezedPointers.getNumEntries() > 0) {
            KERNEL_CALL(
                drawClusters_pixelStyle,
                data.size,
                rectUpperLeft,
                rectLowerRight,
                data.entities.clusterFreezedPointers,
                targetImage,
                imageSize,
                zoom);
        }
        KERNEL_CALL(
            drawParticles_pixelStyle,
            data.size,
            rectUpperLeft,
            rectLowerRight,
            data.entities.particlePointers,
            targetImage,
            imageSize,
            zoom);
    }
}

