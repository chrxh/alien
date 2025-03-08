#include "RenderingKernels.cuh"

#include "SignalProcessor.cuh"
#include "SpotCalculator.cuh"

namespace
{
    auto constexpr ZoomLevelForConnections = 1.0f;
    auto constexpr ZoomLevelForSignalFlow = 15.0f;
    auto constexpr ZoomLevelForShadedCells = 10.0f;

    __device__ __inline__ void drawPixel(uint64_t* imageData, unsigned int index, float3 const& color)
    {
        imageData[index] = toUInt64(color.y * 225.0f) << 16 | toUInt64(color.x * 225.0f) << 0 | toUInt64(color.z * 225.0f) << 32;
    }

    __device__ __inline__ void drawAddingPixel(uint64_t* imageData, unsigned int const& numPixels, unsigned int index, float3 const& colorToAdd)
    {
        if (index < numPixels) {
            uint64_t rawColorToAdd = toUInt64(colorToAdd.y * 255.0f) << 16 | toUInt64(colorToAdd.x * 255.0f) << 0 | toUInt64(colorToAdd.z * 255.0f) << 32;
            alienAtomicAdd64(&imageData[index], rawColorToAdd);
        }
    }

    __device__ __inline__ float3 colorToFloat3(unsigned int value)
    {
        return float3{toFloat(value & 0xff) / 255, toFloat((value >> 8) & 0xff) / 255, toFloat((value >> 16) & 0xff) / 255};
    }

    __device__ __inline__ float2 mapWorldPosToImagePos(float2 const& rectUpperLeft, float2 const& pos, float2 const& universeImageSize, float zoom)
    {
        auto result = float2{(pos.x - rectUpperLeft.x) * zoom, (pos.y - rectUpperLeft.y) * zoom};
        if (cudaSimulationParameters.borderlessRendering) {
            result.x = Math::modulo(result.x, universeImageSize.x);
            result.y = Math::modulo(result.y, universeImageSize.y);
        }
        return result;
    }

    __device__ __inline__ int3 convertHSVtoRGB(float h, float s, float v)
    {
        auto c = v * s;
        auto x = c * (1 - abs(fmodf((h / 60), 2) - 1));
        auto m = v - c;

        float r_, g_, b_;
        if (0 <= h && h < 60.0f) {
            r_ = c;
            g_ = x;
            b_ = 0;
        }
        if (60.0f <= h && h < 120.0f) {
            r_ = x;
            g_ = c;
            b_ = 0;
        }
        if (120.0f <= h && h < 180.0f) {
            r_ = 0;
            g_ = c;
            b_ = x;
        }
        if (180.0f <= h && h < 240.0f) {
            r_ = 0;
            g_ = x;
            b_ = c;
        }
        if (240.0f <= h && h < 300.0f) {
            r_ = x;
            g_ = 0;
            b_ = c;
        }
        if (300.0f <= h && h <= 360.0f) {
            r_ = c;
            g_ = 0;
            b_ = x;
        }
        return {toInt((r_ + m) * 255), toInt((g_ + m) * 255), toInt((b_ + m) * 255)};
    }

    __device__ __inline__ float3 calcColor(Cell* cell, int selected, CellColoring cellColoring, bool primary)
    {
        float factor = max(30.0f, min(300.0f, cell->energy)) / 340.0f;
        if (1 == selected) {
            factor *= 2.5f;
        }
        if (2 == selected) {
            factor *= 1.75f;
        }

        uint32_t cellColor;
        if (cellColoring == CellColoring_None) {
            cellColor = 0xbfbfbf;
        }
        if (cellColoring == CellColoring_CellColor) {
            switch (calcMod(cell->color, 7)) {
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
        }
        if (cellColoring == CellColoring_MutationId || (cellColoring == CellColoring_MutationId_AllCellTypes && primary)) {
            auto colorNumber = cell->mutationId == 0 ? 30 : (cell->mutationId == 1 ? 18 : cell->mutationId + 17);   //6 for zero mutant color
            auto h = abs(toInt((colorNumber * 12107) % 360));
            auto s = 0.6f + toFloat(abs(toInt(colorNumber * 13111)) % 400) / 1000;
            auto rgb = convertHSVtoRGB(toFloat(h), s, 1.0f);
            cellColor = (rgb.x << 16) | (rgb.y << 8) | rgb.z;
        }
        if (cellColoring == CellColoring_LivingState) {
            switch (cell->livingState) {
            case LivingState_Ready:
                cellColor = 0x1010ff;
                break;
            case LivingState_UnderConstruction:
                cellColor = 0x10ff10;
                break;
            case LivingState_Activating:
                cellColor = 0xffffff;
                break;
            case LivingState_Detaching:
                cellColor = 0xbf4040;
                break;
            case LivingState_Reviving:
                cellColor = 0x4040bf;
                break;
            case LivingState_Dying:
                cellColor = 0xff1010;
                break;
            default:
                cellColor = 0x000000;
                break;
            }
        }

        if (cellColoring == CellColoring_GenomeSize) {
            auto rgb = convertHSVtoRGB(toFloat(min(360.0f, 240.0f + powf(cell->genomeComplexity, 0.3f) * 15.0f)),  1.0f, 1.0f);
            cellColor = (rgb.x << 16) | (rgb.y << 8) | rgb.z;
        }

        if (cellColoring == CellColoring_CellType) {
            if (cell->cellType == cudaSimulationParameters.highlightedCellType) {
                auto h = (toFloat(cell->cellType) / toFloat(CellType_Count - 1)) * 360.0f;
                auto rgb = convertHSVtoRGB(toFloat(h), 0.7f, 1.0f);
                cellColor = (rgb.x << 16) | (rgb.y << 8) | rgb.z;
                factor = 2.0f;
            } else {
                cellColor = 0x404040;
            }
        }

        if (cellColoring == CellColoring_AllCellTypes || (cellColoring == CellColoring_MutationId_AllCellTypes && !primary)) {
            auto h = (toFloat(cell->cellType) / toFloat(CellType_Count - 1)) * 360.0f;
            auto rgb = convertHSVtoRGB(toFloat(h), 0.7f, 1.0f);
            cellColor = (rgb.x << 16) | (rgb.y << 8) | rgb.z;
        }

        return {
            toFloat((cellColor >> 16) & 0xff) / 256.0f * factor,
            toFloat((cellColor >> 8) & 0xff) / 256.0f * factor,
            toFloat(cellColor & 0xff) / 256.0f * factor};
    }

    __device__ __inline__ float3 calcColor(Particle* particle, bool selected)
    {
        auto intensity = max(min((toInt(particle->energy) + 10.0f) * 5, 450.0f), 20.0f) / 1000.0f;
        if (selected) {
            intensity *= 2.5f;
        }
        intensity = max(0.08f, intensity);
        return {intensity, intensity, intensity / 2};
    }

    __device__ __inline__ void drawDot(uint64_t* imageData, int2 const& imageSize, float2 const& pos, float3 const& colorToAdd)
    {
        int2 intPos{toInt(pos.x), toInt(pos.y)};
        if (intPos.x >= 0 && intPos.y >= 0 && intPos.y < imageSize.y) {

            float2 posFrac{pos.x - intPos.x, pos.y - intPos.y};
            unsigned int index = intPos.x + intPos.y * imageSize.x;
            auto numPixels = imageSize.x * imageSize.y;

            if (intPos.x < imageSize.x) {
                float3 colorToAdd1 = colorToAdd * (1.0f - posFrac.x) * (1.0f - posFrac.y);
                drawAddingPixel(imageData, numPixels, index, colorToAdd1);

                float3 colorToAdd3 = colorToAdd * (1.0f - posFrac.x) * posFrac.y;
                drawAddingPixel(imageData, numPixels, index + imageSize.x, colorToAdd3);
            }
            if (intPos.x + 1 < imageSize.x) {
                float3 colorToAdd2 = colorToAdd * posFrac.x * (1.0f - posFrac.y);
                drawAddingPixel(imageData, numPixels, index + 1, colorToAdd2);

                float3 colorToAdd4 = colorToAdd * posFrac.x * posFrac.y;
                drawAddingPixel(imageData, numPixels, index + imageSize.x + 1, colorToAdd4);
            }
        }
    }

    __device__ __inline__ void drawCircle(uint64_t* imageData, int2 const& imageSize, float2 pos, float3 color, float radius, bool shaded = true, bool inverted = false)
    {
        if (radius > 2.0 - NEAR_ZERO) {
            auto radiusSquared = radius * radius;
            for (float x = -radius; x <= radius; x += 1.0f) {
                for (float y = -radius; y <= radius; y += 1.0f) {
                    auto rSquared = x * x + y * y;
                    if (rSquared <= radiusSquared) {
                        auto factor = inverted ? (rSquared / radiusSquared) * 2 : (1.0f - rSquared / radiusSquared) * 2;
                        auto angle = Math::angleOfVector({x, y});
                        if (shaded) {
                            angle -= 45.0f;
                            if (angle > 180.0f) {
                                angle -= 360.0f;
                            }
                            if (angle < -180.0f) {
                                angle += 360.0f;
                            }
                            factor *= 40.0f / (abs(angle) + 1.0f);

                            factor = min(factor, 1.0f);
                        }
                        if (inverted && sqrt(rSquared) > radius - 2.0f) {
                            factor = 1.5f;
                        }
                        drawDot(imageData, imageSize, pos + float2{x, y}, color * factor);
                    }
                }
            }
        } else {
            color *= radius * 2;
            drawDot(imageData, imageSize, pos, color);
            color *= 0.45f;
            drawDot(imageData, imageSize, pos + float2{1, 0}, color);
            drawDot(imageData, imageSize, pos + float2{-1, 0}, color);
            drawDot(imageData, imageSize, pos + float2{0, 1}, color);
            drawDot(imageData, imageSize, pos + float2{0, -1}, color);
        }
    }

    __device__ __inline__ void
    drawSection(uint64_t* imageData, int2 const& imageSize, float2 pos, float3 color, float innerRadius, float outerRadius, float angle1, float angle2)
    {
        if (outerRadius > 2.0 - NEAR_ZERO) {
            angle1 = Math::normalizedAngle(angle1, 0.0f);
            angle2 = Math::normalizedAngle(angle2, 0.0f);
            auto outerRadiusSquared = outerRadius * outerRadius;
            auto innerRadiusSquared = innerRadius * innerRadius;
            for (float x = -outerRadius; x <= outerRadius; x += 1.0f) {
                for (float y = -outerRadius; y <= outerRadius; y += 1.0f) {
                    auto rSquared = x * x + y * y;
                    if (rSquared <= outerRadiusSquared && rSquared >= innerRadiusSquared) {
                        auto angle = Math::angleOfVector({x, y});
                        if (Math::isAngleInBetween(angle1, angle2, angle)) {
                            drawDot(imageData, imageSize, pos + float2{x, y}, color);
                        }
                    }
                }
            }
        }
    }

    __device__ __inline__ void drawCircle_block(uint64_t* imageData, int2 const& imageSize, float2 pos, float3 color, float radius)
    {
        if (radius > 2.0 - NEAR_ZERO) {
            auto radiusSquared = radius * radius;
            auto length = 2 * toInt(radius) + 1;
            auto const partition = calcPartition(length * length, threadIdx.x, blockDim.x);
            for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
                auto x = toFloat(index % length) - radius;
                auto y = toFloat(index / length) - radius;
                auto rSquared = x * x + y * y;
                if (rSquared <= radiusSquared) {
                    auto factor = (1.0f - rSquared / radiusSquared) * 2;
                    drawDot(imageData, imageSize, pos + float2{x, y}, color * factor);
                }
            }
        } else {
            if (threadIdx.x == 0) {
                color = color * radius * 2;
                drawDot(imageData, imageSize, pos, color);
                color = color * 0.3f;
                drawDot(imageData, imageSize, pos + float2{1, 0}, color);
                drawDot(imageData, imageSize, pos + float2{-1, 0}, color);
                drawDot(imageData, imageSize, pos + float2{0, 1}, color);
                drawDot(imageData, imageSize, pos + float2{0, -1}, color);
            }
        }
    }

    __device__ __inline__ void drawDisc(uint64_t* imageData, int2 const& imageSize, float2 pos, float3 color, float radius1, float radius2)
    {
        if (radius2 > 2.5 - NEAR_ZERO) {
            auto radiusSquared1 = radius1 * radius1;
            auto radiusSquared2 = radius2 * radius2;
            auto middleRadiusSquared = (radiusSquared1 + radiusSquared2) / 2;

            for (float x = -radius2; x <= radius2; x += 1.0f) {
                for (float y = -radius2; y <= radius2; y += 1.0f) {
                    auto rSquared = x * x + y * y;
                    if (rSquared <= radiusSquared2 && rSquared >= radiusSquared1) {
                        auto weight = rSquared < middleRadiusSquared ? (rSquared - radiusSquared1) / (middleRadiusSquared - radiusSquared1)
                                                                     : (radiusSquared2 - rSquared) / (radiusSquared2 - middleRadiusSquared);
                        drawDot(imageData, imageSize, pos + float2{x, y}, color * weight * 0.6f);
                    }
                }
            }
        } else {
            color = color * (radius2 - radius1) * 2;
            drawDot(imageData, imageSize, pos, color);
            color = color * 0.3f;
            drawDot(imageData, imageSize, pos + float2{1, 0}, color);
            drawDot(imageData, imageSize, pos + float2{-1, 0}, color);
            drawDot(imageData, imageSize, pos + float2{0, 1}, color);
            drawDot(imageData, imageSize, pos + float2{0, -1}, color);
        }
    }

    __device__ __inline__ void
    drawLine(float2 const& start, float2 const& end, float3 const& color, uint64_t* imageData, int2 imageSize, float pixelDistance = 1.5f)
    {
        float dist = Math::length(end - start);
        float2 const v = {static_cast<float>(end.x - start.x) / dist * pixelDistance, static_cast<float>(end.y - start.y) / dist * pixelDistance};
        float2 pos = start;

        for (float d = 0; d <= dist; d += pixelDistance) {
            drawDot(imageData, imageSize, pos, color);
            pos = pos + v;
        }
    }

    __device__ __inline bool isLineVisible(float2 const& startImagePos, float2 const& endImagePos, float2 const& universeImageSize)
    {
        return abs(startImagePos.x - endImagePos.x) < universeImageSize.x / 2 && abs(startImagePos.y - endImagePos.y) < universeImageSize.y / 2;
    }
}

/************************************************************************/
/* Main      															*/
/************************************************************************/
__global__ void cudaPrepareFilteringForRendering(Array<Cell*> filteredCells, Array<Particle*> filteredParticles)
{
    filteredCells.reset();
    filteredParticles.reset();
}

__global__ void cudaFilterCellsForRendering(int2 worldSize, float2 rectUpperLeft, Array<Cell*> cells, Array<Cell*> filteredCells, int2 imageSize, float zoom)
{
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());
    auto universeImageSize = toFloat2(worldSize) * zoom;

    BaseMap map;
    map.init(worldSize);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = cells.at(index);
        auto imagePos = mapWorldPosToImagePos(rectUpperLeft, cell->pos, universeImageSize, zoom);
        if (isContainedInRect({0, 0}, toFloat2(imageSize), imagePos)) {
            auto cellPtr = filteredCells.getNewElement();
            *cellPtr = cell;
        }
    }
}

__global__ void cudaFilterParticlesForRendering(
    int2 worldSize,
    float2 rectUpperLeft,
    Array<Particle*> particles,
    Array<Particle*> filteredParticles,
    int2 imageSize,
    float zoom)
{
    auto const partition = calcAllThreadsPartition(particles.getNumEntries());
    auto universeImageSize = toFloat2(worldSize) * zoom;

    BaseMap map;
    map.init(worldSize);

    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& particle = particles.at(index);
        auto imagePos = mapWorldPosToImagePos(rectUpperLeft, particle->absPos, universeImageSize, zoom);
        if (isContainedInRect({0, 0}, toFloat2(imageSize), imagePos)) {
            auto cellPtr = filteredParticles.getNewElement();
            *cellPtr = particle;
        }
    }
}

__global__ void cudaDrawCells(
    uint64_t timestep,
    int2 worldSize,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    Array<Cell*> cells,
    uint64_t* imageData,
    int2 imageSize,
    float zoom)
{
    auto const partition = calcAllThreadsPartition(cells.getNumEntries());

    BaseMap map;
    map.init(worldSize);

    auto shadedCells = zoom >= ZoomLevelForShadedCells;
    auto universeImageSize = toFloat2(worldSize) * zoom;
    auto coloring = cudaSimulationParameters.cellColoring;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = cells.at(index);

        auto cellPos = cell->pos;
        auto cellImagePos = mapWorldPosToImagePos(rectUpperLeft, cellPos, universeImageSize, zoom);

        auto cellRadius = zoom * cudaSimulationParameters.cellRadius;

        // draw primary color for cell
        auto primaryColor = calcColor(cell, cell->selected, coloring, true) * 0.85f;
        drawCircle(imageData, imageSize, cellImagePos, primaryColor * 0.45f, cellRadius * 7 / 5, false, false);

        // draw secondary color for cell
        auto secondaryColor =
            coloring == CellColoring_MutationId_AllCellTypes ? calcColor(cell, cell->selected, coloring, false) * 0.5f : primaryColor * 0.6f;
        drawCircle(imageData, imageSize, cellImagePos, secondaryColor, cellRadius, shadedCells, true);

        // draw signal restrictions
        if (zoom >= ZoomLevelForSignalFlow && cell->numConnections > 0 && cell->cellType != CellType_Structure && cell->cellType != CellType_Free) {
            float signalAngleRestrictionStart;
            float signalAngleRestrictionEnd;
            if (cell->signalRoutingRestriction.active) {
                signalAngleRestrictionStart = 180.0f + cell->signalRoutingRestriction.baseAngle - cell->signalRoutingRestriction.openingAngle / 2;
                signalAngleRestrictionEnd = 180.0f + cell->signalRoutingRestriction.baseAngle + cell->signalRoutingRestriction.openingAngle / 2;
            } else {
                signalAngleRestrictionStart = 0;
                signalAngleRestrictionEnd = 359.9f;
            }
            auto refAngle = Math::angleOfVector(cell->connections[0].cell->pos - cell->pos);
            drawSection(
                imageData,
                imageSize,
                cellImagePos,
                secondaryColor * 1.5f,
                cellRadius * 11 / 10,
                cellRadius * 13 / 10,
                refAngle + signalAngleRestrictionStart,
                refAngle + signalAngleRestrictionEnd);
        }

        // draw signal activity
        if (zoom >= cudaSimulationParameters.zoomLevelNeuronalActivity) {
            if (cell->signalRelaxationTime == 2) {
                drawCircle(imageData, imageSize, cellImagePos, float3{0.3f, 0.3f, 0.3f}, cellRadius, shadedCells);
            }
            if (cell->signalRelaxationTime == 1) {
                drawCircle(imageData, imageSize, cellImagePos, float3{0.3f, 0.3f, 0.3f}, cellRadius / 2, shadedCells);
            }
        }

        // draw events
        if (cell->eventCounter > 0) {
            if (cudaSimulationParameters.attackVisualization && cell->event == CellEvent_Attacking) {
                drawDisc(imageData, imageSize, cellImagePos, {0.0f, 0.5f, 0.0f}, cellRadius * 1.4f, cellRadius * 2.0f);
            }
            if (cudaSimulationParameters.attackVisualization && cell->event == CellEvent_Attacked) {
                float3 color{0.5f, 0.0f, 0.0f};
                drawDisc(imageData, imageSize, cellImagePos, color, cellRadius * 1.4f, cellRadius * 2.0f);

                auto const endImagePos = mapWorldPosToImagePos(rectUpperLeft, cell->eventPos, universeImageSize, zoom);
                if (isLineVisible(cellImagePos, endImagePos, universeImageSize) && Math::length(cell->eventPos - cell->pos) < 10.0f) {
                    drawLine(cellImagePos, endImagePos, color, imageData, imageSize);
                }
            }
        }

        // draw muscle movements
        if (cudaSimulationParameters.muscleMovementVisualization && cell->cellType == CellType_Muscle
            && (cell->cellTypeData.muscle.lastMovementX != 0 || cell->cellTypeData.muscle.lastMovementY != 0)) {
            float2 lastMovement{cell->cellTypeData.muscle.lastMovementX, cell->cellTypeData.muscle.lastMovementY};
            auto lastMovementLength = Math::length(lastMovement);
            if (lastMovementLength > 0.05f) {
                lastMovement = lastMovement / lastMovementLength * 0.05f;
            }

            auto color = float3{0.7f, 0.7f, 0.7f} * min(1.0f, zoom * 0.1f);
            auto endPos = cell->pos + lastMovement * 100;
            auto endImagePos = mapWorldPosToImagePos(rectUpperLeft, endPos, universeImageSize, zoom);
            if (isLineVisible(cellImagePos, endImagePos, universeImageSize)) {
                drawLine(cellImagePos, endImagePos, color, imageData, imageSize);
            }

            auto arrowPos1 = endPos + float2{-lastMovement.x + lastMovement.y, -lastMovement.x - lastMovement.y} * 20;
            auto arrowImagePos1 = mapWorldPosToImagePos(rectUpperLeft, arrowPos1, universeImageSize, zoom);
            if (isLineVisible(arrowImagePos1, endImagePos, universeImageSize)) {
                drawLine(arrowImagePos1, endImagePos, color, imageData, imageSize);
            }

            auto arrowPos2 = endPos + float2{-lastMovement.x - lastMovement.y, +lastMovement.x - lastMovement.y} * 20;
            auto arrowImagePos2 = mapWorldPosToImagePos(rectUpperLeft, arrowPos2, universeImageSize, zoom);
            if (isLineVisible(arrowImagePos2, endImagePos, universeImageSize)) {
                drawLine(arrowImagePos2, endImagePos, color, imageData, imageSize);
            }
        }

        // draw detonation
        if (cell->cellType == CellType_Detonator) {
            auto const& detonator = cell->cellTypeData.detonator;
            if (detonator.state == DetonatorState_Activated && detonator.countdown < 2) {
                auto radius = toFloat(6 - detonator.countdown * 6);
                radius *= radius;
                radius *= cudaSimulationParameters.cellTypeDetonatorRadius[cell->color] * zoom / 36;
                drawCircle(imageData, imageSize, cellImagePos, float3{0.3f, 0.3f, 0.0f}, radius, shadedCells);
            }
        }

        // draw connections
        auto lineColor = primaryColor * min((zoom - 1.0f) / 3, 1.0f) * 2 * 0.7f;
        if (zoom >= ZoomLevelForConnections) {
            for (int i = 0; i < cell->numConnections; ++i) {
                auto const otherCell = cell->connections[i].cell;
                auto otherCellPos = otherCell->pos;
                auto topologyCorrection = map.getCorrectionIncrement(cellPos, otherCellPos);
                otherCellPos += topologyCorrection;

                auto distFromCellCenter = Math::normalized(otherCellPos - cellPos) * cudaSimulationParameters.cellRadius * 6 / 5 * 11 / 10;
                auto const startImagePos = mapWorldPosToImagePos(rectUpperLeft, cellPos + distFromCellCenter, universeImageSize, zoom);
                auto const endImagePos = mapWorldPosToImagePos(rectUpperLeft, otherCellPos - distFromCellCenter, universeImageSize, zoom);
                if (isLineVisible(startImagePos, endImagePos, universeImageSize)) {
                    drawLine(startImagePos, endImagePos, lineColor, imageData, imageSize);
                }
            }
        }

        // draw arrows
        if (zoom >= ZoomLevelForSignalFlow) {
            auto signalAngleRestrictionStart = 180.0f + cell->signalRoutingRestriction.baseAngle - cell->signalRoutingRestriction.openingAngle / 2;
            auto signalAngleRestrictionEnd = 180.0f + cell->signalRoutingRestriction.baseAngle + cell->signalRoutingRestriction.openingAngle / 2;
            signalAngleRestrictionStart = Math::normalizedAngle(signalAngleRestrictionStart, 0.0f);
            signalAngleRestrictionEnd = Math::normalizedAngle(signalAngleRestrictionEnd, 0.0f);

            auto summedAngle = 0.0f;
            for (int i = 0; i < cell->numConnections; ++i) {
                if (i > 0) {
                    summedAngle += cell->connections[i].angleFromPrevious;
                }
                auto const& otherCell = cell->connections[i].cell;
                if (!cell->signalRoutingRestriction.active
                    || Math::isAngleStrictInBetween(signalAngleRestrictionStart, signalAngleRestrictionEnd, summedAngle)) {
                    auto otherCellPos = otherCell->pos;
                    auto topologyCorrection = map.getCorrectionIncrement(cellPos, otherCellPos);
                    otherCellPos += topologyCorrection;
                    auto distFromCellCenter = Math::normalized(otherCellPos - cellPos) * cudaSimulationParameters.cellRadius * 6 / 5 * 11 / 10;
                    auto const startImagePos = mapWorldPosToImagePos(rectUpperLeft, cellPos + distFromCellCenter, universeImageSize, zoom);
                    auto const endImagePos = mapWorldPosToImagePos(rectUpperLeft, otherCellPos - distFromCellCenter, universeImageSize, zoom);
                    auto direction = Math::normalized(endImagePos - startImagePos);
                    {
                        float2 arrowPartStart = {-direction.x + direction.y, -direction.x - direction.y};
                        arrowPartStart = arrowPartStart * zoom / 14 + endImagePos;
                        if (isLineVisible(arrowPartStart, endImagePos, universeImageSize)) {
                            drawLine(arrowPartStart, endImagePos, lineColor, imageData, imageSize);
                        }
                    }
                    {
                        float2 arrowPartStart = {-direction.x - direction.y, direction.x - direction.y};
                        arrowPartStart = arrowPartStart * zoom / 14 + endImagePos;
                        if (isLineVisible(arrowPartStart, endImagePos, universeImageSize)) {
                            drawLine(arrowPartStart, endImagePos, lineColor, imageData, imageSize);
                        }
                    }
                }
            }
        }
    }
}


__global__ void cudaDrawCellGlow(int2 worldSize, float2 rectUpperLeft, Array<Cell*> cells, uint64_t* imageData, int2 imageSize, float zoom)
{
    __shared__ float2 cellImagePos;
    __shared__ bool isContained;
    __shared__ float3 color;

    auto const partition = calcBlockPartition(cells.getNumEntries());

    BaseMap map;
    if (threadIdx.x == 0) {
        map.init(worldSize);
    }
    __syncthreads();

    auto universeImageSize = toFloat2(worldSize) * zoom;
    auto coloring = cudaSimulationParameters.cellGlowColoring;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& cell = cells.at(index);

        if (threadIdx.x == 0) {
            cellImagePos = mapWorldPosToImagePos(rectUpperLeft, cell->pos, universeImageSize, zoom);
            isContained = isContainedInRect({0, 0}, toFloat2(imageSize), cellImagePos);
        }
        __syncthreads();

        if (isContained) {

            //draw background for cell
            if (threadIdx.x == 0) {
                color = calcColor(cell, cell->selected, coloring, true);
            }
            __syncthreads();

            drawCircle_block(
                imageData, imageSize, cellImagePos, color * cudaSimulationParameters.cellGlowStrength * 0.1f, zoom * cudaSimulationParameters.cellGlowRadius);
        }
    }
}

__global__ void cudaDrawParticles(int2 worldSize, float2 rectUpperLeft, float2 rectLowerRight, Array<Particle*> particles, uint64_t* imageData, int2 imageSize, float zoom)
{
    BaseMap map;
    map.init(worldSize);

    auto const partition = calcPartition(particles.getNumEntries(), threadIdx.x + blockIdx.x * blockDim.x, blockDim.x * gridDim.x);

    auto universeImageSize = toFloat2(worldSize) * zoom;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto const& particle = particles.at(index);
        auto particlePos = particle->absPos;
        map.correctPosition(particlePos);

        auto const particleImagePos = mapWorldPosToImagePos(rectUpperLeft, particlePos, universeImageSize, zoom);
        auto const color = calcColor(particle, 0 != particle->selected);
        auto radius = zoom / 3;
        drawCircle(imageData, imageSize, particleImagePos, color, radius);
    }
}

__global__ void cudaDrawRadiationSources(uint64_t* targetImage, float2 rectUpperLeft, int2 worldSize, int2 imageSize, float zoom)
{
    auto universeImageSize = toFloat2(worldSize) * zoom;
    for (int i = 0; i < cudaSimulationParameters.numRadiationSources; ++i) {
        float2 sourcePos{cudaSimulationParameters.radiationSource[i].posX, cudaSimulationParameters.radiationSource[i].posY};
        auto imagePos = mapWorldPosToImagePos(rectUpperLeft, sourcePos, universeImageSize, zoom);
        if (isContainedInRect({0, 0}, toFloat2(imageSize), imagePos)) {
            for (int dx = -5; dx <= 5; ++dx) {
                auto drawX = toInt(imagePos.x) + dx;
                auto drawY = toInt(imagePos.y);
                if (0 <= drawX && drawX < imageSize.x && 0 <= drawY && drawY < imageSize.y) {
                    int index = drawX + drawY * imageSize.x;
                    targetImage[index] = 0x0000000001ff;
                }
            }
            for (int dy = -5; dy <= 5; ++dy) {
                auto drawX = toInt(imagePos.x);
                auto drawY = toInt(imagePos.y) + dy;
                if (0 <= drawX && drawX < imageSize.x && 0 <= drawY && drawY < imageSize.y) {
                    int index = drawX + drawY * imageSize.x;
                    targetImage[index] = 0x0000000001ff;
                }
            }
        }
    }
}

__global__ void cudaDrawRepetition(int2 worldSize, int2 imageSize, float2 rectUpperLeft, float2 rectLowerRight, uint64_t* imageData, float zoom)
{
    BaseMap map;
    map.init(worldSize);

    auto universeImageSize = toFloat2(worldSize) * zoom;
    auto const partition = calcAllThreadsPartition(imageSize.x * imageSize.y);
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto x = index % imageSize.x;
        auto y = index / imageSize.x;
        if (x >= toInt(universeImageSize.x) || y >= toInt(universeImageSize.y)) {
            float2 worldPos = {toFloat(x) / zoom + rectUpperLeft.x, toFloat(y) / zoom + rectUpperLeft.y};

            auto refPos = mapWorldPosToImagePos(rectUpperLeft, worldPos, universeImageSize, zoom);
            int2 refIntPos{toInt(refPos.x), toInt(refPos.y)};
            if (refIntPos.x >= 0 && refIntPos.x < imageSize.x && refIntPos.y >= 0 && refIntPos.y < imageSize.y) {
                alienAtomicExch64(&imageData[index], imageData[refIntPos.x + refIntPos.y * imageSize.x]);
            }
        }
    }
}

__global__ void cudaDrawSpotsAndGridlines(uint64_t* imageData, int2 imageSize, int2 worldSize, float zoom, float2 rectUpperLeft, float2 rectLowerRight)
{
    BaseMap map;
    map.init(worldSize);

    int2 outsideRectUpperLeft{-min(toInt(rectUpperLeft.x * zoom), 0), -min(toInt(rectUpperLeft.y * zoom), 0)};
    int2 outsideRectLowerRight{
        imageSize.x - max(toInt((rectLowerRight.x - toFloat(worldSize.x)) * zoom), 0),
        imageSize.y - max(toInt((rectLowerRight.y - toFloat(worldSize.y)) * zoom), 0)};

    auto baseColor = colorToFloat3(cudaSimulationParameters.backgroundColor);
    float3 spotColors[MAX_ZONES];
    for (int i = 0; i < cudaSimulationParameters.numZones; ++i) {
        spotColors[i] = colorToFloat3(cudaSimulationParameters.zone[i].color);
    }

    auto const partition = calcAllThreadsPartition(imageSize.x * imageSize.y * sizeof(unsigned int));
    auto const viewWidth = max(1.0f, rectLowerRight.x - rectUpperLeft.x);
    auto const PixelInWorldSize = viewWidth / toFloat(worldSize.x);
    auto const gridDistance = powf(10.0f, truncf(log10f(viewWidth))) / 10.0f;
    auto const maxGridDistance = viewWidth / 10;
    auto const gridRemainder = (maxGridDistance - gridDistance) / maxGridDistance;
    for (int index = partition.startIndex; index <= partition.endIndex; ++index) {
        auto x = index % imageSize.x;
        auto y = index / imageSize.x;
        float2 worldPos = {toFloat(x) / zoom + rectUpperLeft.x, toFloat(y) / zoom + rectUpperLeft.y};

        if (!cudaSimulationParameters.borderlessRendering
            && (x < outsideRectUpperLeft.x || y < outsideRectUpperLeft.y || x >= outsideRectLowerRight.x || y >= outsideRectLowerRight.y)) {
            imageData[index] = 0;
        } else {
            auto color = SpotCalculator::calcResultingValue(map, worldPos, baseColor, spotColors);
            drawPixel(imageData, index, color);
        }

        if (cudaSimulationParameters.gridLines) {
            {
                auto distanceX = Math::modulo(worldPos.x + gridDistance / 2, gridDistance) - gridDistance / 2;
                auto distanceY = Math::modulo(worldPos.y + gridDistance / 2, gridDistance) - gridDistance / 2;
                if (abs(distanceX) <= PixelInWorldSize * 8) {

                    auto viewDistance = max(0.0f, 0.1f - abs(distanceX) * zoom / 10) * gridRemainder * 0.7f;
                    drawAddingPixel(imageData, imageSize.x * imageSize.y, index, {viewDistance, viewDistance, viewDistance});
                }
                if (abs(distanceY) <= PixelInWorldSize * 8) {

                    auto viewDistance = max(0.0f, 0.1f - abs(distanceY) * zoom / 10) * gridRemainder * 0.7f;
                    drawAddingPixel(imageData, imageSize.x * imageSize.y, index, {viewDistance, viewDistance, viewDistance});
                }
            }
            {
                auto distanceX = Math::modulo(worldPos.x + gridDistance / 20, gridDistance / 10) - gridDistance / 20;
                auto distanceY = Math::modulo(worldPos.y + gridDistance / 20, gridDistance / 10) - gridDistance / 20;
                if (abs(distanceX) <= PixelInWorldSize * 8) {

                    auto viewDistance = max(0.0f, 0.1f - abs(distanceX) * zoom / 10) * (1.0f - gridRemainder) * 0.7f;
                    drawAddingPixel(imageData, imageSize.x * imageSize.y, index, {viewDistance, viewDistance, viewDistance});
                }
                if (abs(distanceY) <= PixelInWorldSize * 8) {

                    auto viewDistance = max(0.0f, 0.1f - abs(distanceY) * zoom / 10) * (1.0f - gridRemainder) * 0.7f;
                    drawAddingPixel(imageData, imageSize.x * imageSize.y, index, {viewDistance, viewDistance, viewDistance});
                }
            }
        }
    }
}
