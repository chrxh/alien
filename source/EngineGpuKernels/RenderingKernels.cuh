#pragma once

#include "AccessTOs.cuh"
#include "Base.cuh"
#include "CleanupKernels.cuh"
#include "EngineInterface/Colors.h"
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

__device__ __inline__ int
mapUniversePosToImageIndex(int2 const& imageSize, int2 const& rectUpperLeft, int2 const& intPos)
{
    return (intPos.x - rectUpperLeft.x) + (intPos.y - rectUpperLeft.y) * imageSize.x;
}

__device__ __inline__ int2 mapUniversePosToImagePos(int2 const& rectUpperLeft, float2 const& pos, float zoom)
{
    return {static_cast<int>((pos.x - rectUpperLeft.x) * zoom), static_cast<int>((pos.y - rectUpperLeft.y) * zoom)};
}

__device__ __inline__ int2 mapUniversePosToImagePos(float2 const& rectUpperLeft, float2 const& pos, float zoom)
{
    return {static_cast<int>((pos.x - rectUpperLeft.x) * zoom), static_cast<int>((pos.y - rectUpperLeft.y) * zoom)};
}

__device__ __inline__ unsigned int calcColor(Cell* cell)
{
    unsigned char colorCode = cell->metadata.color;

    unsigned int result;
    switch (colorCode % 7) {
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
    return 0xff000000 | (b << 16) | (g << 8) | r;
}

__device__ __inline__ unsigned int calcColor(Particle* particle)
{
    auto e = min(150, (static_cast<int>(particle->getEnergy()) + 10) * 5);
    return e | 0xff300000;
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

__device__ __inline__ void drawDot(
    unsigned int* imageData,
    int2 const& imageSize,
    int const& index,
    unsigned int color,
    bool selected,
    bool darken = false)
{
    if (!selected) {
        color = (color >> 1) & 0x7e7e7e;
    }
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
drawCircle(unsigned int* imageData, int2 const& imageSize, int index, unsigned int color, bool selected, bool darken)
{
    drawDot(imageData, imageSize, index, color, selected, darken);

    index -= 1 + 2 * imageSize.x;
    drawDot(imageData, imageSize, index, color, selected, darken);
    ++index;
    drawDot(imageData, imageSize, index, color, selected, darken);
    ++index;
    drawDot(imageData, imageSize, index, color, selected, darken);

    index += 1 + imageSize.x;
    drawDot(imageData, imageSize, index, color, selected, darken);
    index += imageSize.x;
    drawDot(imageData, imageSize, index, color, selected, darken);
    index += imageSize.x;
    drawDot(imageData, imageSize, index, color, selected, darken);

    index += imageSize.x - 1;
    drawDot(imageData, imageSize, index, color, selected, darken);
    --index;
    drawDot(imageData, imageSize, index, color, selected, darken);
    --index;
    drawDot(imageData, imageSize, index, color, selected, darken);

    index -= 1 + imageSize.x;
    drawDot(imageData, imageSize, index, color, selected, darken);
    index -= imageSize.x;
    drawDot(imageData, imageSize, index, color, selected, darken);
    index -= imageSize.x;
    drawDot(imageData, imageSize, index, color, selected, darken);
}

__global__ void drawClusters_pixelStyle(
    int2 universeSize,
    int2 rectUpperLeft,
    int2 rectLowerRight,
    Array<Cluster*> clusters,
    unsigned int* imageData,
    int2 imageSize)
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
            auto intPos = toInt2(cell->absPos);
            map.mapPosCorrection(intPos);
            if (isContainedInRect(rectUpperLeft, rectLowerRight, intPos, 1)) {
                auto const index = mapUniversePosToImageIndex(imageSize, rectUpperLeft, intPos);
                auto const color = calcColor(cell);
                drawDot(imageData, imageSize, index, color, cell->cluster->isSelected());
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
                drawDot(imageData, imageSize, index, color, cell->cluster->isSelected());
            }
        }
        __syncthreads();
    }
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
                auto const cellImagePos = mapUniversePosToImagePos(rectUpperLeft, cellPos, zoom);
                auto const color = calcColor(cell);
                if (isContainedInRect({0, 0}, imageSize, cellImagePos, 3)) {
                    auto index = cellImagePos.x + cellImagePos.y * imageSize.x;

                    if (isHighZoom) {
                        drawCircle(imageData, imageSize, index, color, isSelected, false);
                    } else {
                        --index;
                        drawDot(imageData, imageSize, index, color, isSelected, isLowZoom);
                        index += 2;
                        drawDot(imageData, imageSize, index, color, isSelected, isLowZoom);
                        index -= 1 + imageSize.x;
                        drawDot(imageData, imageSize, index, color, isSelected, isLowZoom);
                        index += 2 * imageSize.x;
                        drawDot(imageData, imageSize, index, color, isSelected, isLowZoom);
                    }
                }

                auto const posCorrection = cellPos - cell->absPos;

                for (int i = 0; i < cell->numConnections; ++i) {
                    auto const otherCell = cell->connections[i];
                    auto const otherCellPos = otherCell->absPos + posCorrection;
                    auto const otherCellImagePos = mapUniversePosToImagePos(rectUpperLeft, otherCellPos, zoom);
                    float dist = Math::length(otherCellImagePos - cellImagePos);
                    float2 const v = {
                        static_cast<float>(otherCellImagePos.x - cellImagePos.x) / dist * 1.8f,
                        static_cast<float>(otherCellImagePos.y - cellImagePos.y) / dist * 1.8f};
                    float2 pos = toFloat2(cellImagePos);

                    for (float d = 0; d <= dist; d += 1.8f) {
                        auto const intPos = toInt2(pos);
                        if (isContainedInRect({0, 0}, imageSize, intPos, 2)) {
                            auto const index = intPos.x + intPos.y * imageSize.x;
                            drawDot(imageData, imageSize, index, color, isSelected, true);
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
            auto const cellImagePos = mapUniversePosToImagePos(rectUpperLeft, cellPos, zoom);
            if (isContainedInRect({0, 0}, imageSize, cellImagePos, 4)) {
                auto index = cellImagePos.x + cellImagePos.y * imageSize.x;
                auto const color = calcColor(token);

                drawCircle(imageData, imageSize, index, color, cell->cluster->isSelected(), isLowZoom);
            }
        }
        __syncthreads();
    }
}

__global__ void drawParticles_pixelStyle(
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
            drawDot(imageData, imageSize, index, color, particle->isSelected());
        }
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

        auto const particleImagePos = mapUniversePosToImagePos(rectUpperLeft, particle->absPos, zoom);
        if (isContainedInRect({0, 0}, imageSize, particleImagePos, 4)) {
            auto index = particleImagePos.x + particleImagePos.y * imageSize.x;
            auto const color = calcColor(particle);
            drawCircle(imageData, imageSize, index, color, particle->isSelected(), zoom < 4.1);
        }
    }
}

__global__ void blurImage(unsigned int* sourceImage, unsigned int* targetImage, int2 imageSize)
{
    auto constexpr Radius = 5;

    auto const pixelBlock = calcPartition(imageSize.x * imageSize.y, blockIdx.x, gridDim.x);
    for (int index = pixelBlock.startIndex; index <= pixelBlock.endIndex; ++index) {

        __shared__ int red, green, blue;
        if (0 == threadIdx.x && 0 == threadIdx.y) {
            red = 0;
            green = 0;
            blue = 0;
        }
        __syncthreads();

        int2 pos{index % imageSize.x, index / imageSize.x};
        int2 relPos{static_cast<int>(threadIdx.x) - 3, static_cast<int>(threadIdx.y) - 3};

        auto scanPos = pos - relPos;
        if (scanPos.x >= 0 && scanPos.y >= 0 && scanPos.x < imageSize.x && scanPos.y < imageSize.y) {
            auto r = Math::length(toFloat2(relPos));
            if (r <= Radius + FP_PRECISION) {
                auto const pixel = sourceImage[scanPos.x + scanPos.y * imageSize.x];
                auto const scanRed = (pixel >> 16) & 0xff;
                auto const scanGreen = (pixel >> 8) & 0xff;
                auto const scanBlue = pixel & 0xff;
                auto const factor = cudaImageBlurFactors[floorInt(r)];
                atomicAdd_block(&red, scanRed * factor);
                atomicAdd_block(&green, scanGreen * factor);
                atomicAdd_block(&blue, scanBlue * factor);
            }
        }
        __syncthreads();

        if (0 == threadIdx.x && 0 == threadIdx.y) {
            auto sum = cudaImageBlurFactors[6];
            red = min(255, red / sum);
            green = min(255, green / sum);
            blue = min(255, blue / sum);
            targetImage[pos.x + pos.y * imageSize.x] = 0xff000000 | (red << 16) | (green << 8) | blue;
        }
        __syncthreads();
    }
}


/************************************************************************/
/* Main      															*/
/************************************************************************/

__global__ void cudaDrawImage_pixelStyle(int2 rectUpperLeft, int2 rectLowerRight, int2 imageSize, SimulationData data)
{
/*    int width = rectLowerRight.x - rectUpperLeft.x + 1;
    int height = rectLowerRight.y - rectUpperLeft.y + 1;
    int numPixels = width * height;
    int2 imageSize{width, height};
*/
    unsigned int* targetImage;
    if (cudaExecutionParameters.imageGlow) {
        targetImage = data.rawImageData;
    } else {
        targetImage = data.finalImageData;
    }
    KERNEL_CALL(clearImageMap, targetImage, imageSize.x * imageSize.y);
/*    KERNEL_CALL(
        drawClusters_pixelStyle,
        data.size,
        rectUpperLeft,
        rectLowerRight,
        data.entities.clusterPointers,
        targetImage,
        imageSize);
    if (data.entities.clusterFreezedPointers.getNumEntries() > 0) {
        KERNEL_CALL(
            drawClusters_pixelStyle,
            data.size,
            rectUpperLeft,
            rectLowerRight,
            data.entities.clusterFreezedPointers,
            targetImage,
            imageSize);
    }
    KERNEL_CALL(
        drawParticles_pixelStyle,
        data.size,
        rectUpperLeft,
        rectLowerRight,
        data.entities.particlePointers,
        targetImage,
        imageSize);

    if (cudaExecutionParameters.imageGlow) {
        auto const numBlocks = cudaConstants.NUM_BLOCKS * cudaConstants.NUM_THREADS_PER_BLOCK / 8;
        blurImage<<<numBlocks, dim3{11, 11}>>>(data.rawImageData, data.finalImageData, imageSize);
        cudaDeviceSynchronize();
    }
    */
}

__global__ void drawImage_vectorStyle(
    float2 rectUpperLeft,
    float2 rectLowerRight,
    int2 imageSize,
    float zoom,
    SimulationData data)
{
    unsigned int* targetImage;
    if (cudaExecutionParameters.imageGlow) {
        targetImage = data.rawImageData;
    } else {
        targetImage = data.finalImageData;
    }

    KERNEL_CALL(clearImageMap, targetImage, imageSize.x * imageSize.y);
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

    if (cudaExecutionParameters.imageGlow) {
        auto const numBlocks = cudaConstants.NUM_BLOCKS * cudaConstants.NUM_THREADS_PER_BLOCK / 8;
        blurImage<<<numBlocks, dim3{7, 7}>>>(data.rawImageData, data.finalImageData, imageSize);
        cudaDeviceSynchronize();
    }
}

