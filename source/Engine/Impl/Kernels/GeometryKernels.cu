#include "GeometryKernels.cuh"

#include <Data/ObjectColoring.h>

#include "ParameterCalculator.cuh"

namespace
{
    // The radiation center shows the light color of its source darkened, so that it stays in the background
    __device__ float constexpr RadiationCenterBrightness = 0.45f;

    // The rendered activity approaches signal #0 with this time constant, so that signals changing with every cell function pass do not let it
    // flicker
    __device__ float constexpr ActivityResponseTimesteps = 30.0f;

    // The outlines of solids follow the lattice of their particles in steps. Where an outline changes between bending outwards and inwards, its
    // particles are rendered closer to the middle of their neighbors on the outline. Outlines that bend in one direction only, such as those of
    // small round solids, keep their shape.
    __device__ float constexpr SolidOutlineSmoothing = 0.5f;

    // Bends of an outline below this angle in radians count as straight
    __device__ float constexpr MinSolidOutlineBend = 0.17f;

    // The steps of the lattice are smaller than this distance in world units, so particles that stick out further stay close to their position
    __device__ float constexpr MaxSolidOutlineShift = 0.5f;

    __device__ __inline__ void correctPositionForRendering(float2& pos, float2 const& visibleUpperLeft, int2 const& worldSize)
    {
        if (cudaSimulationParameters.borderlessRendering.value) {
            pos.x -= visibleUpperLeft.x;
            pos.y -= visibleUpperLeft.y;
            pos.x = Math::modulo(pos.x, toFloat(worldSize.x));
            pos.y = Math::modulo(pos.y, toFloat(worldSize.y));
            pos.x += visibleUpperLeft.x;
            pos.y += visibleUpperLeft.y;
        } else {
            pos.x = Math::modulo(pos.x, toFloat(worldSize.x));
            pos.y = Math::modulo(pos.y, toFloat(worldSize.y));
        }
    }
}

__global__ void cudaCorrectPositionsForRendering(SimulationData data, float2 visibleTopLeft)
{
    {
        auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& object = data.entities.objects.at(index);
            correctPositionForRendering(object->pos, visibleTopLeft, data.world.getSize());
        }
    }
    {
        auto const& partition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& particle = data.entities.energies.at(index);
            correctPositionForRendering(particle->pos, visibleTopLeft, data.world.getSize());
        }
    }
}


namespace
{
    __device__ __inline__ uint32_t getCustomizationColor(int colorCode)
    {
        auto const& color = cudaSimulationParameters.customizationColors.value[calcMod(colorCode, MAX_COLORS)];
        auto const toInt = [](float value) {
            if (value < 0.0f) {
                value = 0.0f;
            } else if (value > 1.0f) {
                value = 1.0f;
            }
            return min(255u, static_cast<uint32_t>(value * 255.0f + 0.5f));
        };
        return (toInt(color.r) << 16) | (toInt(color.g) << 8) | toInt(color.b);
    }

    __device__ __inline__ uint32_t getObjectColor(Object* object)
    {
        auto coloring = cudaSimulationParameters.objectColoring.value[calcMod(object->color, MAX_COLORS)];
        if (coloring == CellColoring_None) {
            return 0x7f7f7f;
        }
        if (coloring == CellColoring_LineageAndCustomization) {
            if (object->type != ObjectType_Cell) {
                return getCustomizationColor(object->color);
            }
            auto const& color = cudaSimulationParameters.customizationColors.value[calcMod(object->color, MAX_COLORS)];
            float r = fminf(1.0f, fmaxf(0.0f, color.r));
            float g = fminf(1.0f, fmaxf(0.0f, color.g));
            float b = fminf(1.0f, fmaxf(0.0f, color.b));
            float h, s, v;
            ObjectColoring::rgbToHsv(r, g, b, h, s, v);
            auto lineageId = object->typeData.cell.creature->lineageId;
            uint32_t hash = lineageId * 2654435761u;
            float hueOffset = (static_cast<float>(hash & 0xFFFFu) / 65535.0f) * 0.2f - 0.1f;
            h += hueOffset;
            h -= floorf(h);
            return ObjectColoring::hsvToRgb(h, s, v);
        }
        if (coloring == CellColoring_Lineage) {
            if (object->type != ObjectType_Cell) {
                return getCustomizationColor(object->color);
            }
            return ObjectColoring::getColorFromId(object->typeData.cell.creature->lineageId);
        }
        if (coloring == CellColoring_Creature) {
            if (object->type != ObjectType_Cell) {
                return getCustomizationColor(object->color);
            }
            return ObjectColoring::getColorFromId(static_cast<uint32_t>(object->typeData.cell.creature->id));
        }
        return getCustomizationColor(object->color);
    }

    __device__ __inline__ bool isInVisibleRect(float2 const& pos, GeometryExtractionContext const& context)
    {
        return pos.x >= context.visibleTopLeft.x - context.cullingMargin && pos.x <= context.visibleBottomRight.x + context.cullingMargin
            && pos.y >= context.visibleTopLeft.y - context.cullingMargin && pos.y <= context.visibleBottomRight.y + context.cullingMargin;
    }

    // Only visible cells are updated, hence the elapsed simulation time is measured per cell. Paused simulations keep their rendered activity.
    __device__ __inline__ void updateRenderedActivity(Cell& cell, uint64_t timestep)
    {
        auto timestepBits = static_cast<uint16_t>(timestep);
        auto elapsedTimesteps = static_cast<uint16_t>(timestepBits - cell.renderedActivityTimestep);
        if (elapsedTimesteps == 0) {
            return;
        }
        cell.renderedActivityTimestep = timestepBits;
        auto activity = toInt(cell.renderedActivity);
        auto target = toInt(cell.getRenderedActivityTarget());
        auto newActivity = target + toInt(roundf(toFloat(activity - target) * expf(-toFloat(elapsedTimesteps) / ActivityResponseTimesteps)));

        // Differences below one step of the 8 bit activity would otherwise remain forever
        if (newActivity == activity && activity != target) {
            newActivity += activity < target ? 1 : -1;
        }
        cell.renderedActivity = static_cast<uint8_t>(newActivity);
    }
}

__global__ void cudaExtractObjectData(SimulationData data, ObjectVertexData* objectData, uint64_t* numObjects, GeometryExtractionContext context)
{
    auto const& objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);
        if (!isInVisibleRect(object->pos, context)) {
            continue;
        }

        auto idx = alienAtomicAdd64(numObjects, uint64_t(1));
        if (objectData != nullptr) {
            int isIsolatedOrTail = (object->numConnections <= 1) ? 1 : 0;

            auto const& pos = object->pos;

            auto const& cellColor = getObjectColor(object);

            float luminance;
            float zOffset = 0.0f;
            int cellType = 0;

            if (object->type == ObjectType_Cell) {
                luminance = object->typeData.cell.getEnergy() / 350.0f;
                zOffset = toFloat(object->typeData.cell.creature->id % 1000) / 2000;
                cellType = object->typeData.cell.cellType;
            } else if (object->type == ObjectType_FreeCell) {
                luminance = object->typeData.freeCell.energy / 300.0f;
            } else if (object->type == ObjectType_Fluid) {
                luminance = object->typeData.fluid.energy / 300.0f;
            } else {
                // Solid - use energy for luminance if available
                luminance = object->typeData.solid.energy / 300.0f;
            }

            auto white = luminance / 10.0f;
            if (object->selected == 1) {
                luminance = (luminance + 0.05f) * 1.15f;
                white = 0.1f;
            }
            if (object->selected == 2) {
                luminance = luminance * 1.15f;
            }
            luminance = min(3.0f, luminance);
            white = min(2.0f, white);

            // Calculate deterministic z-position based on cell id for lighting
            // Use a simple hash function to get a pseudo-random value in range [0, 1]
            uint64_t hash = object->id * 2654435761u;  // Knuth's multiplicative hash
            hash = (hash ^ (hash >> 16)) * 0x85ebca6b;
            hash = (hash ^ (hash >> 13)) * 0xc2b2ae35;
            hash = hash ^ (hash >> 16);
            float normalizedHash = toFloat(hash & 0xFFFFFF) / toFloat(0xFFFFFF);
            float zPos = normalizedHash * 0.05f;

            objectData[idx].pos[0] = pos.x;
            objectData[idx].pos[1] = pos.y;
            objectData[idx].pos[2] = zPos + zOffset;
            objectData[idx].color[0] = toFloat((cellColor >> 16) & 0xff) / 255.0f * luminance + white;
            objectData[idx].color[1] = toFloat((cellColor >> 8) & 0xff) / 255.0f * luminance + white;
            objectData[idx].color[2] = toFloat(cellColor & 0xff) / 255.0f * luminance + white;

            // Compute highlight intensity and activity from cell
            float highlightIntensity = 0.0f;
            float activity = 0.0f;
            if (object->type == ObjectType_Cell) {
                auto& cell = object->typeData.cell;
                highlightIntensity = toFloat(cell.highlightIntensity) / 255.0f;
                updateRenderedActivity(cell, *data.timestep);
                activity = toFloat(cell.renderedActivity) / 255.0f;
            }

            // Pack cellType (bits 0-7), objectType (bits 8-15), and isIsolatedOrTail (bit 16) into state field
            objectData[idx].state = cellType | (object->type << 8) | (isIsolatedOrTail << 16);
            objectData[idx].highlightIntensity = highlightIntensity;
            objectData[idx].activity = activity;

            object->tempValue1.as_uint64 = idx;
        }
    }
}

namespace
{
    // Isolated particles, the ends of chains and the particles with the maximum number of connections, which lie inside, keep their positions
    __device__ __inline__ bool isMovableOnOutline(Object* object)
    {
        return object->type == ObjectType_Solid && object->numConnections >= 2 && object->numConnections < MAX_OBJECT_CONNECTIONS;
    }

    // The passes of the smoothing alternate between the positions in the vertex data and temporary positions of the particles that may move
    struct RenderPositions
    {
        ObjectVertexData* objectData;
        GeometryExtractionContext context;
        bool fromTemporaryPositions;

        __device__ __inline__ float2 get(Object* object) const
        {
            if (!isInVisibleRect(object->pos, context)) {
                return object->pos;
            }
            if (fromTemporaryPositions && isMovableOnOutline(object)) {
                return object->tempValue2.as_float2;
            }
            auto const& pos = objectData[object->tempValue1.as_uint64].pos;
            return {pos[0], pos[1]};
        }

        __device__ __inline__ void setNext(Object* object, float2 const& value) const
        {
            if (fromTemporaryPositions) {
                auto& pos = objectData[object->tempValue1.as_uint64].pos;
                pos[0] = value.x;
                pos[1] = value.y;
            } else {
                object->tempValue2.as_float2 = value;
            }
        }
    };

    // Two consecutive connections of an object enclose a sector. It belongs to a triangle if the connected objects are connected with each
    // other, or to a quad if both are connected to a common fourth object. The back indices are those of the connections to the object.
    struct Sector
    {
        bool isTriangle;
        Object* quadCorner;

        __device__ __inline__ bool isCovered() const { return isTriangle || quadCorner != nullptr; }
    };

    __device__ __inline__ Sector getSector(Object* object, Object* connectedObject, int backIndex, Object* prevConnectedObject, int prevBackIndex)
    {
        auto candidate = prevConnectedObject->getConnectedObject(prevBackIndex - 1);
        if (candidate == connectedObject) {
            return {true, nullptr};
        }
        auto isQuad = candidate == connectedObject->getConnectedObject(backIndex + 1) && candidate != object && connectedObject != prevConnectedObject;
        return {false, isQuad ? candidate : nullptr};
    }

    struct SolidOutline
    {
        Object* previous;
        Object* next;
        int bend;  // 1: outwards, -1: inwards, 0: straight
    };

    // A solid particle lies on an outline if exactly one sector between its connections is not covered by a polygon
    __device__ __inline__ bool getSolidOutline(SimulationData const& data, Object* object, RenderPositions const& positions, SolidOutline& result)
    {
        if (!isMovableOnOutline(object)) {
            return false;
        }
        int numConnections = object->numConnections;
        auto pos = positions.get(object);
        float2 directions[MAX_OBJECT_CONNECTIONS];
        int backIndices[MAX_OBJECT_CONNECTIONS];
        for (int i = 0; i < numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            directions[i] = data.world.getCorrectedDirection(positions.get(connectedObject) - pos);
            backIndices[i] = connectedObject->getConnectionIndex(object);
        }
        auto uncoveredSector = -1;
        auto coveredAngle = 0.0f;
        for (int i = 0; i < numConnections; ++i) {
            auto prevIndex = (i + numConnections - 1) % numConnections;
            auto sector = getSector(object, object->connections[i].object, backIndices[i], object->connections[prevIndex].object, backIndices[prevIndex]);
            if (!sector.isCovered()) {
                if (uncoveredSector != -1) {
                    return false;
                }
                uncoveredSector = i;
                continue;
            }
            coveredAngle += atan2f(abs(Math::cross(directions[prevIndex], directions[i]).z), Math::dot(directions[prevIndex], directions[i]));
        }
        if (uncoveredSector == -1) {
            return false;
        }
        auto openAngle = 2.0f * Const::PI - coveredAngle;
        result.previous = object->connections[(uncoveredSector + numConnections - 1) % numConnections].object;
        result.next = object->connections[uncoveredSector].object;
        result.bend = openAngle > Const::PI + MinSolidOutlineBend ? 1 : (openAngle < Const::PI - MinSolidOutlineBend ? -1 : 0);
        return true;
    }

    __device__ __inline__ float2 getSmoothedPosition(SimulationData const& data, Object* object, RenderPositions const& positions)
    {
        auto pos = positions.get(object);
        SolidOutline outline;
        if (!getSolidOutline(data, object, positions, outline) || outline.bend == 0) {
            return pos;
        }
        if (outline.previous->type != ObjectType_Solid || outline.next->type != ObjectType_Solid) {
            return pos;
        }
        SolidOutline neighborOutline;
        auto previousBend = getSolidOutline(data, outline.previous, positions, neighborOutline) ? neighborOutline.bend : 0;
        auto nextBend = getSolidOutline(data, outline.next, positions, neighborOutline) ? neighborOutline.bend : 0;
        if (previousBend == outline.bend && nextBend == outline.bend) {
            return pos;
        }
        auto toPrevious = data.world.getCorrectedDirection(positions.get(outline.previous) - pos);
        auto toNext = data.world.getCorrectedDirection(positions.get(outline.next) - pos);
        auto shift = data.world.getCorrectedDirection(pos + (toPrevious + toNext) * (SolidOutlineSmoothing / 2) - object->pos);
        auto shiftLength = Math::length(shift);
        if (shiftLength > MaxSolidOutlineShift) {
            shift = shift * (MaxSolidOutlineShift / shiftLength);
        }
        return object->pos + shift;
    }
}

__global__ void cudaSmoothSolidOutlines(SimulationData data, ObjectVertexData* objectData, GeometryExtractionContext context, bool fromTemporaryPositions)
{
    RenderPositions positions{objectData, context, fromTemporaryPositions};
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (!isMovableOnOutline(object) || !isInVisibleRect(object->pos, context)) {
            continue;
        }
        positions.setNext(object, getSmoothedPosition(data, object, positions));
    }
}

__global__ void cudaExtractLineIndices(SimulationData data, unsigned int* lineIndices, uint64_t* numLineIndices, GeometryExtractionContext context)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (!isInVisibleRect(object->pos, context)) {
            continue;
        }

        // Cell index is just the array index (stored in tempValue)
        uint64_t objectIndex = object->tempValue1.as_uint64;

        // Add line indices for each connection
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            if (!isInVisibleRect(connectedObject->pos, context)) {
                continue;
            }

            // Only add each connection once (from lower index to higher index to avoid duplicates)
            if (object->id < connectedObject->id) {
                if (Math::length(object->pos - connectedObject->pos) <= cudaSimulationParameters.maxBindingDistance.value[object->color]) {
                    uint64_t lineIndex = alienAtomicAdd64(numLineIndices, uint64_t(2));
                    if (lineIndices != nullptr) {
                        uint64_t connectedIndex = connectedObject->tempValue1.as_uint64;
                        lineIndices[lineIndex] = static_cast<unsigned int>(objectIndex);
                        lineIndices[lineIndex + 1] = static_cast<unsigned int>(connectedIndex);
                    }
                }
            }
        }
    }
}

__global__ void cudaExtractTriangleIndices(SimulationData data, unsigned int* triangleIndices, uint64_t* numTriangleIndices, GeometryExtractionContext context)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    auto addTriangle = [&](Object* object, uint64_t objectIndex, Object* connectedObject, Object* prevConnectedObject) {
        if (!isInVisibleRect(connectedObject->pos, context) || !isInVisibleRect(prevConnectedObject->pos, context)) {
            return;
        }
        // Only add triangle once (avoid duplicates by checking ids)
        if (Math::length(object->pos - connectedObject->pos) <= cudaSimulationParameters.maxBindingDistance.value[object->color]
            && Math::length(object->pos - prevConnectedObject->pos) <= cudaSimulationParameters.maxBindingDistance.value[object->color]
            && Math::length(connectedObject->pos - prevConnectedObject->pos) <= cudaSimulationParameters.maxBindingDistance.value[connectedObject->color]) {
            uint64_t connectedIndex1 = connectedObject->tempValue1.as_uint64;
            uint64_t connectedIndex2 = prevConnectedObject->tempValue1.as_uint64;
            uint64_t triangleIndex = alienAtomicAdd64(numTriangleIndices, uint64_t(3));
            if (triangleIndices != nullptr) {
                triangleIndices[triangleIndex] = static_cast<unsigned int>(objectIndex);
                triangleIndices[triangleIndex + 1] = static_cast<unsigned int>(connectedIndex1);
                triangleIndices[triangleIndex + 2] = static_cast<unsigned int>(connectedIndex2);
            }
        }
    };
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (!isInVisibleRect(object->pos, context)) {
            continue;
        }
        if (object->numConnections <= 1) {
            continue;
        }
        int numConnections = object->numConnections;
        int backIndices[MAX_OBJECT_CONNECTIONS];
        for (int i = 0; i < numConnections; ++i) {
            backIndices[i] = object->connections[i].object->getConnectionIndex(object);
        }
        for (int i = 0; i < numConnections; ++i) {
            auto prevIndex = (i + numConnections - 1) % numConnections;
            auto const& connectedObject = object->connections[i].object;
            auto const& prevConnectedObject = object->connections[prevIndex].object;
            auto sector = getSector(object, connectedObject, backIndices[i], prevConnectedObject, backIndices[prevIndex]);

            // Each polygon is added once, by its corner with the lowest id
            if (object->id > connectedObject->id || object->id > prevConnectedObject->id) {
                continue;
            }
            if (sector.isTriangle) {
                addTriangle(object, object->tempValue1.as_uint64, prevConnectedObject, connectedObject);
            }
            if (sector.quadCorner != nullptr && object->id < sector.quadCorner->id) {
                addTriangle(object, object->tempValue1.as_uint64, connectedObject, sector.quadCorner);
                addTriangle(object, object->tempValue1.as_uint64, sector.quadCorner, prevConnectedObject);
            }
        }
    }
}

__global__ void
cudaExtractFluidParticleData(SimulationData data, FluidParticleVertexData* fluidParticleData, uint64_t* numFluidParticles, GeometryExtractionContext context)
{
    // Process energy particles - each particle goes to its index position
    {
        auto const& energyPartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
        for (int index = energyPartition.startIndex; index <= energyPartition.endIndex; index += energyPartition.step) {
            auto const& energy = data.entities.energies.at(index);
            if (!energy) {
                continue;
            }

            auto pos = energy->pos;
            if (!isInVisibleRect(pos, context)) {
                continue;
            }

            float intensity = (energy->energy) / 350.0f + 0.10f;
            if (energy->selected) {
                intensity *= 5.0f;
            }

            auto idx = alienAtomicAdd64(numFluidParticles, uint64_t(1));
            if (fluidParticleData != nullptr) {
                fluidParticleData[idx].pos[0] = pos.x;
                fluidParticleData[idx].pos[1] = pos.y;
                fluidParticleData[idx].pos[2] = 0.0f;
                fluidParticleData[idx].color[0] = intensity * 0.25f;
                fluidParticleData[idx].color[1] = intensity * 0.25f;
                fluidParticleData[idx].color[2] = intensity * 1.0f;
                fluidParticleData[idx].glow = 0.0f;
                fluidParticleData[idx].vel[0] = energy->vel.x;
                fluidParticleData[idx].vel[1] = energy->vel.y;
            }
        }
    }

    // Process fluid objects - render them like energy particles
    {
        auto const& objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
        for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
            auto const& object = data.entities.objects.at(index);
            if (object->type != ObjectType_Fluid) {
                continue;
            }
            if (!isInVisibleRect(object->pos, context)) {
                continue;
            }

            float intensity = (object->typeData.fluid.energy + 5.0f) / 200.0f;
            if (object->selected) {
                intensity *= 2.5f;
            }

            auto idx = alienAtomicAdd64(numFluidParticles, uint64_t(1));
            if (fluidParticleData != nullptr) {
                fluidParticleData[idx].pos[0] = object->pos.x;
                fluidParticleData[idx].pos[1] = object->pos.y;
                fluidParticleData[idx].pos[2] = 1.0f;
                auto color = getObjectColor(object);
                fluidParticleData[idx].color[0] = intensity * toFloat((color >> 16) & 0xff) / 255;
                fluidParticleData[idx].color[1] = intensity * toFloat((color >> 8) & 0xff) / 255;
                fluidParticleData[idx].color[2] = intensity * toFloat(color & 0xff) / 255;
                fluidParticleData[idx].glow = object->typeData.fluid.glow;
                fluidParticleData[idx].vel[0] = object->vel.x;
                fluidParticleData[idx].vel[1] = object->vel.y;
            }
        }
    }
}

__global__ void cudaExtractLocationData(SimulationData data, LocationVertexData* locationData, uint64_t* numLocations, float2 visibleTopLeft)
{
    // Extract location data for layers and sources
    // Each location is rendered 5 times (normal + 4 world offsets for periodic boundaries)
    auto worldSize = data.world.getSize();

    // Process layers
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        auto const& pos_asRealVector2D = cudaSimulationParameters.layerPosition.layerValues[i];
        float2 pos{pos_asRealVector2D.x, pos_asRealVector2D.y};
        correctPositionForRendering(pos, visibleTopLeft, data.world.getSize());
        auto color = cudaSimulationParameters.backgroundColor.layerValues[i].value;
        auto shapeType = cudaSimulationParameters.layerShape.layerValues[i];
        auto radius = cudaSimulationParameters.layerCoreRadius.layerValues[i];
        auto rect = cudaSimulationParameters.layerCoreRect.layerValues[i];
        auto fadeoutRadius = cudaSimulationParameters.layerFadeoutRadius.layerValues[i];
        auto opacity = cudaSimulationParameters.layerOpacity.layerValues[i];

        // Vary the background brightness along the height map of the force field
        auto const& forceFieldType = cudaSimulationParameters.layerForceFieldType.layerValues[i];
        ForceField fieldType = ForceField_None;
        auto fieldParam1 = 0.0f;
        auto fieldParam2 = 0.0f;
        if (forceFieldType.enabled && cudaSimulationParameters.layerShowForceField.layerValues[i]) {
            fieldType = forceFieldType.value;
            switch (fieldType) {
            case ForceField_Radial:
                fieldParam1 = Orientation_Clockwise == cudaSimulationParameters.layerRadialForceFieldOrientation.layerValues[i] ? 1.0f : -1.0f;
                break;
            case ForceField_Linear:
                fieldParam1 = cudaSimulationParameters.layerLinearForceFieldAngle.layerValues[i];
                break;
            case ForceField_PerlinNoise: {
                fieldParam1 = cudaSimulationParameters.layerPerlinNoiseForceFieldSpatialSize.layerValues[i];
                auto temporalSize = cudaSimulationParameters.layerPerlinNoiseForceFieldTemporalSize.layerValues[i];
                fieldParam2 = toFloat(*data.timestep) / max(temporalSize, 1.0f);
            } break;
            }
        }
        auto colored = cudaSimulationParameters.backgroundColor.layerValues[i].enabled;
        if (!colored && fieldType == ForceField_None) {
            continue;
        }

        // Render at 9 positions for periodic boundaries
        auto worldSizeX = toFloat(worldSize.x);
        auto worldSizeY = toFloat(worldSize.y);
        float offsets[9][2] = {
            {-worldSizeX, -worldSizeY},
            {0.0f, -worldSizeY},
            {worldSizeX, -worldSizeY},
            {-worldSizeX, 0.0f},
            {0.0f, 0.0f},
            {worldSizeX, 0.0f},
            {-worldSizeX, worldSizeY},
            {0.0f, worldSizeY},
            {worldSizeX, worldSizeY},
        };

        for (int j = 0; j < 9; ++j) {
            if (locationData != nullptr) {
                locationData[*numLocations].pos[0] = pos.x + offsets[j][0];
                locationData[*numLocations].pos[1] = pos.y + offsets[j][1];
                locationData[*numLocations].color[0] = color.r;
                locationData[*numLocations].color[1] = color.g;
                locationData[*numLocations].color[2] = color.b;
                locationData[*numLocations].shapeType = shapeType;  // 0 = circular, 1 = rectangular
                if (shapeType == 0) {                               // Circular
                    locationData[*numLocations].dimension1 = radius;
                    locationData[*numLocations].dimension2 = 0.0f;
                } else {  // Rectangular
                    locationData[*numLocations].dimension1 = rect.x;
                    locationData[*numLocations].dimension2 = rect.y;
                }
                locationData[*numLocations].fadeoutRadius = fadeoutRadius;
                locationData[*numLocations].opacity = opacity;
                locationData[*numLocations].fieldType = fieldType;
                locationData[*numLocations].fieldParam1 = fieldParam1;
                locationData[*numLocations].fieldParam2 = fieldParam2;
                locationData[*numLocations].colored = colored ? 1 : 0;
            }
            ++(*numLocations);
        }
    }

    // Process sources
    for (int i = 0; i < cudaSimulationParameters.numSources; ++i) {
        if (!cudaSimulationParameters.sourceShowRadiationCenter.sourceValues[i]) {
            continue;
        }
        auto const& pos_asRealVector2D = cudaSimulationParameters.sourcePosition.sourceValues[i];
        float2 pos{pos_asRealVector2D.x, pos_asRealVector2D.y};
        correctPositionForRendering(pos, visibleTopLeft, data.world.getSize());

        auto const& lightColor = cudaSimulationParameters.sourceLightColor.sourceValues[i];
        float3 color = {lightColor.r * RadiationCenterBrightness, lightColor.g * RadiationCenterBrightness, lightColor.b * RadiationCenterBrightness};
        auto shapeType = cudaSimulationParameters.sourceShapeType.sourceValues[i];
        auto radius = cudaSimulationParameters.sourceCircularRadius.sourceValues[i];
        auto rect = cudaSimulationParameters.sourceRectangularRect.sourceValues[i];

        // Render at 5 positions for periodic boundaries
        float offsets[5][2] = {
            {0.0f, 0.0f},
            {static_cast<float>(worldSize.x), 0.0f},
            {-static_cast<float>(worldSize.x), 0.0f},
            {0.0f, static_cast<float>(worldSize.y)},
            {0.0f, -static_cast<float>(worldSize.y)}};

        for (int j = 0; j < 5; ++j) {
            if (locationData != nullptr) {
                locationData[*numLocations].pos[0] = pos.x + offsets[j][0];
                locationData[*numLocations].pos[1] = pos.y + offsets[j][1];
                locationData[*numLocations].color[0] = color.x;
                locationData[*numLocations].color[1] = color.y;
                locationData[*numLocations].color[2] = color.z;
                locationData[*numLocations].shapeType = shapeType;  // 0 = circular, 1 = rectangular
                if (shapeType == 0) {                               // Circular
                    locationData[*numLocations].dimension1 = radius;
                    locationData[*numLocations].dimension2 = 0.0f;
                    locationData[*numLocations].fadeoutRadius = radius / 5.0f;
                } else {  // Rectangular
                    locationData[*numLocations].dimension1 = rect.x;
                    locationData[*numLocations].dimension2 = rect.y;
                    locationData[*numLocations].fadeoutRadius = 0.0f;
                    locationData[*numLocations].fadeoutRadius = radius / min(rect.x, rect.y) / 5.0f;
                }
                locationData[*numLocations].opacity = 1.0f;  // Sources are fully opaque
                locationData[*numLocations].fieldType = ForceField_None;
                locationData[*numLocations].fieldParam1 = 0.0f;
                locationData[*numLocations].fieldParam2 = 0.0f;
                locationData[*numLocations].colored = 1;
            }
            ++(*numLocations);
        }
    }
}

__global__ void cudaExtractSelectedObjectData(
    SimulationData data,
    SelectedObjectVertexData* selectedObjectData,
    uint64_t* numSelectedObjects,
    GeometryExtractionContext context)
{
    // Process selected cells
    auto const& objects = data.entities.objects;
    auto numObjects = objects.getNumEntries();

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < numObjects; index += blockDim.x * gridDim.x) {
        auto const& object = objects.at(index);
        if (object->selected == 1 && isInVisibleRect(object->pos, context)) {
            auto outputIndex = alienAtomicAdd64(numSelectedObjects, static_cast<uint64_t>(1));
            if (selectedObjectData != nullptr) {
                selectedObjectData[outputIndex].pos[0] = object->pos.x;
                selectedObjectData[outputIndex].pos[1] = object->pos.y;
            }
        }
    }

    // Process selected energy particles
    auto const& energies = data.entities.energies;
    auto numEnergies = energies.getNumEntries();

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < numEnergies; index += blockDim.x * gridDim.x) {
        auto const& energy = energies.at(index);
        if (energy->selected == 1 && isInVisibleRect(energy->pos, context)) {
            auto outputIndex = alienAtomicAdd64(numSelectedObjects, static_cast<uint64_t>(1));
            if (selectedObjectData != nullptr) {
                selectedObjectData[outputIndex].pos[0] = energy->pos.x;
                selectedObjectData[outputIndex].pos[1] = energy->pos.y;
            }
        }
    }
}

__global__ void cudaExtractSelectedConnectionData(
    SimulationData data,
    ConnectionArrowVertexData* connectionArrowData,
    uint64_t* numConnectionArrowVertices,
    GeometryExtractionContext context)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (object->selected == 0) {
            continue;
        }
        if (!isInVisibleRect(object->pos, context)) {
            continue;
        }

        // Process each connection from this cell
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;
            if (!isInVisibleRect(connectedObject->pos, context)) {
                continue;
            }

            // Only add each connection once (from lower id to higher id to avoid duplicates)
            if (object->id >= connectedObject->id) {
                continue;
            }

            // Check if this connection should be drawn
            if (Math::length(object->pos - connectedObject->pos) > cudaSimulationParameters.maxBindingDistance.value[object->color]) {
                continue;
            }

            float connectionWeightToObject1 = 0.0f;
            float connectionWeightToObject2 = 0.0f;
            if (object->type == ObjectType_Cell) {
                // connectionWeightToObject1: weight on object for this connection (signal flows from connectedObject to object)
                auto* nn = object->typeData.cell.neuralNetwork;
                connectionWeightToObject1 = nn->connectionWeights[i];
            }
            if (connectedObject->type == ObjectType_Cell) {
                // connectionWeightToObject2: weight on connectedObject for reverse connection (signal flows from object to connectedObject)
                auto* nn = connectedObject->typeData.cell.neuralNetwork;
                auto backIndex = connectedObject->getConnectionIndex(object);
                connectionWeightToObject2 = nn->connectionWeights[backIndex];
            }

            // Get cell colors
            auto cellColor = getObjectColor(object);
            auto connectedObjectColor = getObjectColor(connectedObject);

            // Add connection arrow data (2 vertices for the line)
            uint64_t vertexIndex = alienAtomicAdd64(numConnectionArrowVertices, uint64_t(2));
            if (connectionArrowData != nullptr) {

                // First vertex (object1)
                connectionArrowData[vertexIndex].pos[0] = object->pos.x;
                connectionArrowData[vertexIndex].pos[1] = object->pos.y;
                connectionArrowData[vertexIndex].color[0] = toFloat((cellColor >> 16) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex].color[1] = toFloat((cellColor >> 8) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex].color[2] = toFloat((cellColor >> 0) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex].connectionWeightToObject1 = connectionWeightToObject1;
                connectionArrowData[vertexIndex].connectionWeightToObject2 = connectionWeightToObject2;

                // Second vertex (object2)
                connectionArrowData[vertexIndex + 1].pos[0] = connectedObject->pos.x;
                connectionArrowData[vertexIndex + 1].pos[1] = connectedObject->pos.y;
                connectionArrowData[vertexIndex + 1].color[0] = toFloat((connectedObjectColor >> 16) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex + 1].color[1] = toFloat((connectedObjectColor >> 8) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex + 1].color[2] = toFloat((connectedObjectColor >> 0) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex + 1].connectionWeightToObject1 = connectionWeightToObject1;
                connectionArrowData[vertexIndex + 1].connectionWeightToObject2 = connectionWeightToObject2;
            }
        }
    }
}
__global__ void
cudaExtractAttackEventData(SimulationData data, AttackEventVertexData* attackEventData, uint64_t* numAttackEventVertices, GeometryExtractionContext context)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (!isInVisibleRect(object->pos, context)) {
            continue;
        }

        CellEvent event;
        uint8_t eventCounter;
        float2 eventPos;

        if (object->type == ObjectType_Cell) {
            event = object->typeData.cell.event;
            eventCounter = object->typeData.cell.eventCounter;
            eventPos = object->typeData.cell.eventPos;
        } else if (object->type == ObjectType_FreeCell) {
            event = object->typeData.freeCell.event;
            eventCounter = object->typeData.freeCell.eventCounter;
            eventPos = object->typeData.freeCell.eventPos;
        } else {
            continue;
        }
        // Only process cells that have been attacked and have attackVisualization enabled
        if (eventCounter > 0 && event == CellEvent_Attacked) {

            // The attacker position follows the attacked object only when the cell functions are executed, in between it is extrapolated
            eventPos += object->vel * (cudaSimulationParameters.timestepSize.value * toFloat(*data.timestep % TIMESTEPS_PER_CELL_FUNCTION));

            if (!isInVisibleRect(eventPos, context)) {
                continue;
            }

            // Check if the attacker position is close enough to draw
            if (Math::length(eventPos - object->pos) < 10.0f) {
                // Add attack event line data (2 vertices for the line: from attacker to attacked)
                uint64_t vertexIndex = alienAtomicAdd64(numAttackEventVertices, uint64_t(2));
                if (attackEventData != nullptr) {
                    // Red color for attacked event
                    float redColor[3] = {0.5f, 0.0f, 0.0f};

                    // First vertex (attacker position - from eventPos)
                    attackEventData[vertexIndex].pos[0] = eventPos.x;
                    attackEventData[vertexIndex].pos[1] = eventPos.y;
                    attackEventData[vertexIndex].color[0] = redColor[0];
                    attackEventData[vertexIndex].color[1] = redColor[1];
                    attackEventData[vertexIndex].color[2] = redColor[2];

                    // Second vertex (attacked cell position)
                    attackEventData[vertexIndex + 1].pos[0] = object->pos.x;
                    attackEventData[vertexIndex + 1].pos[1] = object->pos.y;
                    attackEventData[vertexIndex + 1].color[0] = redColor[0];
                    attackEventData[vertexIndex + 1].color[1] = redColor[1];
                    attackEventData[vertexIndex + 1].color[2] = redColor[2];
                }
            }
        }
    }
}
__global__ void cudaExtractDetonationEventData(
    SimulationData data,
    DetonationEventVertexData* detonationEventData,
    uint64_t* numDetonationEventVertices,
    GeometryExtractionContext context)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        if (!isInVisibleRect(object->pos, context)) {
            continue;
        }

        if (object->typeData.cell.eventCounter > 0 && object->typeData.cell.event == CellEvent_Detonation) {
            uint64_t vertexIndex = alienAtomicAdd64(numDetonationEventVertices, uint64_t(1));
            if (detonationEventData != nullptr) {
                auto& vertex = detonationEventData[vertexIndex];
                vertex.objectId = object->id;
                vertex.pos[0] = object->pos.x;
                vertex.pos[1] = object->pos.y;
                vertex.radius = cudaSimulationParameters.detonatorRadius.value[object->color];
            }
        }
    }
}
