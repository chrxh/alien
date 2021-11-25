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

__device__ __inline__ void drawPixel(unsigned int* imageData, unsigned int index, float3 const& color)
{
    unsigned int greenRed = toInt(color.y * 225.0f) << 16 | toInt(color.x * 225.0f);
    unsigned int blue = toInt(color.z * 225.0f);

    imageData[2 * index] = greenRed;
    imageData[2 * index + 1] = blue;
}

__device__ __inline__ void drawAddingPixel(unsigned int* imageData, unsigned int index, float3 const& colorToAdd)
{
    unsigned int greenRed = toInt(colorToAdd.y * 255.0f) << 16 | toInt(colorToAdd.x * 255.0f);
    unsigned int blue = toInt(colorToAdd.z * 255.0f);

    atomicAdd(&imageData[2 * index], greenRed);
    atomicAdd(&imageData[2 * index + 1], blue);
}

__device__ float3 colorToFloat3(unsigned int value)
{
    return float3{toFloat(value & 0xff) / 255, toFloat((value >> 8) & 0xff) / 255, toFloat((value >> 16) & 0xff) / 255};
}

__device__ float3 mix(float3 const& a, float3 const& b, float factor)
{
    return float3{
        a.x * factor + b.x * (1 - factor), a.y * factor + b.y * (1 - factor), a.z * factor + b.z * (1 - factor)};
}

__device__ float3 mix(float3 const& a, float3 const& b, float3 const& c, float factor1, float factor2)
{
    float weight1 = factor1 * factor2;
    float weight2 = 1 - factor1;
    float weight3 = 1 - factor2;
    float sum = weight1 + weight2 + weight3;
    weight1 /= sum;
    weight2 /= sum;
    weight3 /= sum;
    return float3{
        a.x * weight1 + b.x * weight2 + c.x * weight3,
        a.y * weight1 + b.y * weight2 + c.y * weight3,
        a.z * weight1 + b.z * weight2 + c.z * weight3};
}

__global__ void drawBackground(
    unsigned int* imageData,
    int2 imageSize,
    int2 worldSize,
    float zoom,
    float2 rectUpperLeft,
    float2 rectLowerRight)
{
    int2 outsideRectUpperLeft{-min(toInt(rectUpperLeft.x * zoom), 0), -min(toInt(rectUpperLeft.y * zoom), 0)};
    int2 outsideRectLowerRight{
        imageSize.x - max(toInt((rectLowerRight.x - worldSize.x) * zoom), 0),
        imageSize.y - max(toInt((rectLowerRight.y - worldSize.y) * zoom), 0)};

    MapInfo map;
    map.init(worldSize);
    auto spaceColor = colorToFloat3(Const::SpaceColor);
    auto spotColor1 = colorToFloat3(cudaSimulationParametersSpots.spots[0].color);
    auto spotColor2 = colorToFloat3(cudaSimulationParametersSpots.spots[1].color);

    auto const block = calcPartition(imageSize.x * imageSize.y, threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (int index = block.startIndex; index <= block.endIndex; ++index) {
        auto x = index % imageSize.x;
        auto y = index / imageSize.x;
        if (x < outsideRectUpperLeft.x || y < outsideRectUpperLeft.y || x >= outsideRectLowerRight.x
            || y >= outsideRectLowerRight.y) {
            imageData[2 * index] = 0;
            imageData[2 * index + 1] = 0;
        } else {
            if (0 == cudaSimulationParametersSpots.numSpots) {
                drawPixel(imageData, index, spaceColor);
            }
            if (1 == cudaSimulationParametersSpots.numSpots) {
                float2 worldPos = {toFloat(x) / zoom + rectUpperLeft.x, toFloat(y) / zoom + rectUpperLeft.y};
                auto distance = map.mapDistance(
                    worldPos,
                    {cudaSimulationParametersSpots.spots[0].posX, cudaSimulationParametersSpots.spots[0].posY});
                auto coreRadius = cudaSimulationParametersSpots.spots[0].coreRadius;
                auto fadeoutRadius = cudaSimulationParametersSpots.spots[0].fadeoutRadius + 1;
                auto factor = distance < coreRadius ? 0.0f : min(1.0f, (distance - coreRadius) / fadeoutRadius);
                auto resultingColor = mix(spaceColor, spotColor1, factor);
                drawPixel(imageData, index, resultingColor);
            }
            if (2 == cudaSimulationParametersSpots.numSpots) {
                float2 worldPos = {toFloat(x) / zoom + rectUpperLeft.x, toFloat(y) / zoom + rectUpperLeft.y};
                auto distance1 = map.mapDistance(
                    worldPos,
                    {cudaSimulationParametersSpots.spots[0].posX, cudaSimulationParametersSpots.spots[0].posY});
                auto distance2 = map.mapDistance(
                    worldPos,
                    {cudaSimulationParametersSpots.spots[1].posX, cudaSimulationParametersSpots.spots[1].posY});

                auto coreRadius1 = cudaSimulationParametersSpots.spots[0].coreRadius;
                auto fadeoutRadius1 = cudaSimulationParametersSpots.spots[0].fadeoutRadius + 1;
                auto factor1 = distance1 < coreRadius1 ? 0.0f : min(1.0f, (distance1 - coreRadius1) / fadeoutRadius1);
                auto coreRadius2 = cudaSimulationParametersSpots.spots[1].coreRadius;
                auto fadeoutRadius2 = cudaSimulationParametersSpots.spots[1].fadeoutRadius + 1;
                auto factor2 = distance2 < coreRadius2 ? 0.0f : min(1.0f, (distance2 - coreRadius2) / fadeoutRadius2);

                auto resultingColor = mix(spaceColor, spotColor1, spotColor2, factor1, factor2);
                drawPixel(imageData, index, resultingColor);
            }

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

    float factor = min(300.0f, cell->energy) / 220.0f;
    if (!selected) {
        factor *= 0.75f;
    }

    return {
        toFloat((cellColor >> 16) & 0xff) / 256.0f * factor,
        toFloat((cellColor >> 8) & 0xff) / 256.0f * factor,
        toFloat(cellColor & 0xff) / 256.0f * factor};
}

__device__ __inline__ float3 calcColor(Particle* particle, bool selected)
{
/*
    unsigned int particleColor;
    switch (particle->metadata.color % 7) {
    case 0: {
        particleColor = Const::IndividualCellColor1;
        break;
    }
    case 1: {
        particleColor = Const::IndividualCellColor2;
        break;
    }
    case 2: {
        particleColor = Const::IndividualCellColor3;
        break;
    }
    case 3: {
        particleColor = Const::IndividualCellColor4;
        break;
    }
    case 4: {
        particleColor = Const::IndividualCellColor5;
        break;
    }
    case 5: {
        particleColor = Const::IndividualCellColor6;
        break;
    }
    case 6: {
        particleColor = Const::IndividualCellColor7;
        break;
    }
    }

    float factor = min(100.0f, sqrt(particle->energy) * 5 + 20.0f) / 100.0f;
    if (!selected) {
        factor *= 0.75f;
    }

    return {
        toFloat((particleColor >> 16) & 0xff) / 256.0f * factor,
        toFloat((particleColor >> 8) & 0xff) / 256.0f * factor,
        toFloat(particleColor & 0xff) / 256.0f * factor};
*/
    auto intensity = max(min((toInt(particle->energy) + 10) * 5, 150), 20) / 200.0f;
    if (!selected) {
        intensity *= 0.75f;
    }

    return {intensity, 0, 0.08f};
}

__device__ __inline__ float3 calcColor(Token* token, bool selected)
{
    return selected ? float3{1.0f, 1.0f, 1.0f} : float3{0.75f, 0.75f, 0.75f};
}

__device__ __inline__ void
drawDot(unsigned int* imageData, int2 const& imageSize, float2 const& pos, float3 const& colorToAdd)
{
    int2 intPos{toInt(pos.x), toInt(pos.y)};
    if (intPos.x >= 1 && intPos.x < imageSize.x - 1 && intPos.y >= 1 && intPos.y < imageSize.y - 1) {

        float2 posFrac{pos.x - intPos.x, pos.y - intPos.y};
        unsigned int index = intPos.x + intPos.y * imageSize.x;

        float3 colorToAdd1 = colorToAdd * (1.0f - posFrac.x) * (1.0f - posFrac.y);
        drawAddingPixel(imageData, index, colorToAdd1);

        float3 colorToAdd2 = colorToAdd * posFrac.x * (1.0f - posFrac.y);
        drawAddingPixel(imageData, index, colorToAdd2);

        float3 colorToAdd3 = colorToAdd * (1.0f - posFrac.x) * posFrac.y;
        drawAddingPixel(imageData, index + imageSize.x, colorToAdd3);

        float3 colorToAdd4 = colorToAdd * posFrac.x * posFrac.y;
        drawAddingPixel(imageData, index + imageSize.x + 1, colorToAdd4);
    }
}

__device__ __inline__ void
drawCircle(unsigned int* imageData, int2 const& imageSize, float2 pos, float3 color, float radius, bool inverted = false)
{
    if (radius > 1.5 - FP_PRECISION) {
        auto radiusSquared = radius * radius;
        for (float x = -radius; x <= radius; x += 1.0f) {
            for (float y = -radius; y <= radius; y += 1.0f) {
                auto rSquared = x * x + y * y;
                if (rSquared <= radiusSquared) {
                    auto factor = inverted ? (rSquared / radiusSquared) * 2 : (1.0f - rSquared / radiusSquared) * 2;
                    drawDot(imageData, imageSize, pos + float2{x, y}, color * min(factor, 1.0f));
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

__global__ void drawCells(
    int2 universeSize,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    Array<Cell*> cells,
    unsigned int* imageData,
    int2 imageSize,
    float zoom)
{
    auto const partition =
        calcPartition(cells.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    MapInfo map;
    map.init(universeSize);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = cells.at(index);

        auto cellPos = cell->absPos;
        map.mapPosCorrection(cellPos);
        if (isContainedInRect(rectUpperLeft, rectLowerRight, cellPos)) {
            auto cellImagePos = mapUniversePosToVectorImagePos(rectUpperLeft, cellPos, zoom);
            auto color = calcColor(cell, 0 != cell->selected);
            drawCircle(imageData, imageSize, cellImagePos, color, zoom / 3, true);

            if (zoom > 1 - FP_PRECISION) {
                color = color * min((zoom - 1.0f) / 3, 1.0f);
                for (int i = 0; i < cell->numConnections; ++i) {
                    auto const otherCell = cell->connections[i].cell;
                    auto const otherCellPos = otherCell->absPos;
                    auto topologyCorrection = map.correctionIncrement(cellPos, otherCellPos);
                    if (Math::lengthSquared(topologyCorrection) < FP_PRECISION) {
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
    }
}

__global__ void drawTokens(
    int2 universeSize,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    Array<Token*> tokens,
    unsigned int* imageData,
    int2 imageSize,
    float zoom)
{
    MapInfo map;
    map.init(universeSize);

    auto partition =
        calcPartition(tokens.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);
    for (auto tokenIndex = partition.startIndex; tokenIndex <= partition.endIndex; ++tokenIndex) {
        auto const& token = tokens.at(tokenIndex);
        auto const& cell = token->cell;

        auto cellPos = cell->absPos;
        map.mapPosCorrection(cellPos);
        auto const cellImagePos = mapUniversePosToVectorImagePos(rectUpperLeft, cellPos, zoom);
        if (isContainedInRect({0, 0}, imageSize, cellImagePos)) {
            auto const color = calcColor(token, false);
            drawCircle(imageData, imageSize, cellImagePos, color, zoom / 2);
        }
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
            auto const color = calcColor(particle, 0 != particle->selected);
            drawCircle(imageData, imageSize, particleImagePos, color, zoom / 3);
        }
    }
}

__device__ void drawFlowCenters(unsigned int* targetImage, float2 const& rectUpperLeft, int2 imageSize, float zoom)
{
    if (cudaFlowFieldSettings.active) {
        for (int i = 0; i < cudaFlowFieldSettings.numCenters; ++i) {
            auto const& radialFlowData = cudaFlowFieldSettings.centers[i];
            int screenPosX = toInt(radialFlowData.posX * zoom) - rectUpperLeft.x * zoom;
            int screenPosY = toInt(radialFlowData.posY * zoom) - rectUpperLeft.y * zoom;
            auto drawX = screenPosX;
            auto drawY = screenPosY;
            if (0 <= drawX && drawX < imageSize.x && 0 <= drawY && drawY < imageSize.y) {
                int index = drawX + drawY * imageSize.x;
                targetImage[index * 2] = 0;
                targetImage[index * 2 + 1] = 0xfffff;
            }
        }
    }
}

/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void
drawImageKernel(float2 rectUpperLeft, float2 rectLowerRight, int2 imageSize, float zoom, SimulationData data)
{
    unsigned int* targetImage = data.imageData;

    KERNEL_CALL(drawBackground, targetImage, imageSize, data.size, zoom, rectUpperLeft, rectLowerRight);

    KERNEL_CALL(
        drawCells, data.size, rectUpperLeft, rectLowerRight, data.entities.cellPointers, targetImage, imageSize, zoom);

    KERNEL_CALL(
        drawTokens, data.size, rectUpperLeft, rectLowerRight, data.entities.tokenPointers, targetImage, imageSize, zoom);

    KERNEL_CALL(
        drawParticles,
        data.size,
        rectUpperLeft,
        rectLowerRight,
        data.entities.particlePointers,
        targetImage,
        imageSize,
        zoom);

    drawFlowCenters(targetImage, rectUpperLeft, imageSize, zoom);
}


