#include "GeometryKernels.cuh"
#include "ParameterCalculator.cuh"

namespace
{
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
            correctPositionForRendering(object->pos, visibleTopLeft, data.worldSize);
        }
    }
    {
        auto const& partition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& particle = data.entities.energies.at(index);
            correctPositionForRendering(particle->pos, visibleTopLeft, data.worldSize);
        }
    }
}


namespace
{
    __device__ __inline__ uint32_t hsvToRgb(float h, float s, float v)
    {
        float c = v * s;
        float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
        float m = v - c;
        float r, g, b;
        int hi = static_cast<int>(h * 6.0f) % 6;
        switch (hi) {
        case 0:
            r = c;
            g = x;
            b = 0;
            break;
        case 1:
            r = x;
            g = c;
            b = 0;
            break;
        case 2:
            r = 0;
            g = c;
            b = x;
            break;
        case 3:
            r = 0;
            g = x;
            b = c;
            break;
        case 4:
            r = x;
            g = 0;
            b = c;
            break;
        default:
            r = c;
            g = 0;
            b = x;
            break;
        }
        auto ri = min(255u, static_cast<uint32_t>((r + m) * 255.0f + 0.5f));
        auto gi = min(255u, static_cast<uint32_t>((g + m) * 255.0f + 0.5f));
        auto bi = min(255u, static_cast<uint32_t>((b + m) * 255.0f + 0.5f));
        return (ri << 16) | (gi << 8) | bi;
    }

    __device__ __inline__ uint32_t getCellColorByCode(int colorCode)
    {
        uint32_t result;
        switch (calcMod(colorCode, MAX_COLORS)) {
        case 0: {
            result = Const::IndividualObjectColor1;
            break;
        }
        case 1: {
            result = Const::IndividualObjectColor2;
            break;
        }
        case 2: {
            result = Const::IndividualObjectColor3;
            break;
        }
        case 3: {
            result = Const::IndividualObjectColor4;
            break;
        }
        case 4: {
            result = Const::IndividualObjectColor5;
            break;
        }
        case 5: {
            result = Const::IndividualObjectColor6;
            break;
        }
        case 6: {
            result = Const::IndividualObjectColor7;
            break;
        }
        }
        return result;
    }

    __device__ __inline__ uint32_t getCellColor(Object* object)
    {
        auto coloring = cudaSimulationParameters.cellColoring.value;
        if (coloring == CellColoring_None) {
            return 0xffffff;
        }
        if (coloring == CellColoring_LineageId) {
            if (object->type != ObjectType_Cell) {
                return 0xffffff;
            }
            auto lineageId = object->typeData.cell.creature->genome->lineageId;
            uint32_t hash1 = lineageId * 2654435761u;
            uint32_t hash2 = lineageId * 2246822519u;
            float h = static_cast<float>(hash1 & 0xFFFFu) / 65535.0f;
            float s = 0.5f + 0.5f * (static_cast<float>(hash2 & 0xFFFFu) / 65535.0f);
            return hsvToRgb(h, s, 1.0f);
        }
        return getCellColorByCode(object->color);
    }
}

__global__ void cudaExtractCellData(SimulationData data, ObjectVertexData* objectData)
{
    // Process cells - each cell goes to its index position
    auto const& objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
        auto const& object = data.entities.objects.at(index);

        int isInTriangleOrQuad = 0;
        if (object->numConnections > 1) {
            bool first = true;
            int backIndices[MAX_OBJECT_CONNECTIONS];
            for (int i = 0, numConnections = object->numConnections; i < numConnections + 1; ++i) {
                auto connectionIndex = i % numConnections;
                auto const& connectedObject = object->connections[connectionIndex].object;
                auto backIndex = connectedObject->getConnectionIndex(object);
                backIndices[connectionIndex] = backIndex;
                if (first) {
                    first = false;
                    continue;
                }
                auto prevIndex = (connectionIndex + numConnections - 1) % numConnections;
                auto const& prevConnectedObject = object->connections[prevIndex].object;
                auto prevBackIndex = backIndices[prevIndex];

                // Triangle?
                if (prevConnectedObject->getConnectedObject(prevBackIndex - 1) == connectedObject) {
                    isInTriangleOrQuad = 1;
                    break;
                }

                // Rectangle?
                auto fourthCellCandidate1 = connectedObject->getConnectedObject(backIndex + 1);
                auto fourthCellCandidate2 = prevConnectedObject->getConnectedObject(prevBackIndex - 1);
                if (fourthCellCandidate2 == fourthCellCandidate1 && fourthCellCandidate1 != object && fourthCellCandidate2 != object
                    && connectedObject != prevConnectedObject) {
                    isInTriangleOrQuad = 1;
                    break;
                }
            }
        }


        auto const& pos = object->pos;

        auto const& cellColor = getCellColor(object);

        float luminance;
        float zOffset = 0.0f;
        int cellType = 0;

        if (object->type == ObjectType_Cell) {
            luminance = object->typeData.cell.getEnergy() / 300.0f;
            zOffset = toFloat(object->typeData.cell.creature->id % 1000) / 2000;
            cellType = object->typeData.cell.cellType;
        } else if (object->type == ObjectType_FreeCell) {
            luminance = object->typeData.freeCell.energy / 300.0f;
        } else {
            // Structure - use energy for luminance if available
            luminance = object->typeData.structure.energy / 300.0f;
        }

        auto white = luminance / 10.0f;
        if (object->selected == 1) {
            luminance = (luminance + 0.1f) * 1.7f;
            white = 0.5f;
        }
        if (object->selected == 2) {
            luminance = luminance * 1.3f;
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

        // Isolated structures (no connections) will be rendered as blurry particles instead
        bool isIsolatedStructure = (object->type == ObjectType_Structure && object->numConnections == 0);

        // Write cell data at cell index position
        objectData[index].pos[0] = pos.x;
        objectData[index].pos[1] = pos.y;
        objectData[index].pos[2] = zPos + zOffset;
        if (isIsolatedStructure) {
            objectData[index].color[0] = 0.0f;
            objectData[index].color[1] = 0.0f;
            objectData[index].color[2] = 0.0f;
        } else {
            objectData[index].color[0] = toFloat((cellColor >> 16) & 0xff) / 255.0f * luminance + white;
            objectData[index].color[1] = toFloat((cellColor >> 8) & 0xff) / 255.0f * luminance + white;
            objectData[index].color[2] = toFloat(cellColor & 0xff) / 255.0f * luminance + white;
        }

        // Compute signal changes from cell
        float signalChanges = 0.0f;
        if (object->type == ObjectType_Cell) {
            signalChanges = toFloat(object->typeData.cell.signalChanges) / 255.0f;
        }

        // Pack cellType (bits 0-7), objectType (bits 8-15), and isInTriangleOrQuad (bit 16) into state field
        objectData[index].state = cellType | (object->type << 8) | (isInTriangleOrQuad << 16);
        objectData[index].signalChanges = signalChanges;

        // Store cell index for line extraction (just use the index directly)
        object->tempValue.as_uint64 = index;
    }
}

__global__ void cudaExtractLineIndices(SimulationData data, unsigned int* lineIndices, uint64_t* numLineIndices)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);

        // Cell index is just the array index (stored in tempValue)
        uint64_t objectIndex = object->tempValue.as_uint64;

        // Add line indices for each connection
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;

            // Only add each connection once (from lower index to higher index to avoid duplicates)
            if (object->id < connectedObject->id) {
                if (Math::length(object->pos - connectedObject->pos) <= cudaSimulationParameters.maxBindingDistance.value[object->color]) {
                    uint64_t lineIndex = alienAtomicAdd64(numLineIndices, uint64_t(2));
                    if (lineIndices != nullptr) {
                        uint64_t connectedIndex = connectedObject->tempValue.as_uint64;
                        lineIndices[lineIndex] = static_cast<unsigned int>(objectIndex);
                        lineIndices[lineIndex + 1] = static_cast<unsigned int>(connectedIndex);
                    }
                }
            }
        }
    }
}

__global__ void cudaExtractTriangleIndices(SimulationData data, unsigned int* triangleIndices, uint64_t* numTriangleIndices)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    auto addTriangle = [&](Object* object, uint64_t objectIndex, Object* connectedObject, Object* prevConnectedObject) {
        // Only add triangle once (avoid duplicates by checking ids)
        if (Math::length(object->pos - connectedObject->pos) <= cudaSimulationParameters.maxBindingDistance.value[object->color]
            && Math::length(object->pos - prevConnectedObject->pos) <= cudaSimulationParameters.maxBindingDistance.value[object->color]
            && Math::length(connectedObject->pos - prevConnectedObject->pos) <= cudaSimulationParameters.maxBindingDistance.value[connectedObject->color]) {
            uint64_t connectedIndex1 = connectedObject->tempValue.as_uint64;
            uint64_t connectedIndex2 = prevConnectedObject->tempValue.as_uint64;
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
        if (object->numConnections <= 1) {
            continue;
        }
        bool first = true;
        int backIndices[MAX_OBJECT_CONNECTIONS];
        for (int i = 0, numConnections = object->numConnections; i < numConnections + 1; ++i) {
            auto connectionIndex = i % numConnections;
            auto const& connectedObject = object->connections[connectionIndex].object;
            auto backIndex = connectedObject->getConnectionIndex(object);
            backIndices[connectionIndex] = backIndex;
            if (first) {
                first = false;
                continue;
            }
            auto prevIndex = (connectionIndex + numConnections - 1) % numConnections;
            auto const& prevConnectedObject = object->connections[prevIndex].object;
            auto prevBackIndex = backIndices[prevIndex];

            // Triangle?
            if (prevConnectedObject->getConnectedObject(prevBackIndex - 1) == connectedObject) {
                if (object->id < connectedObject->id && object->id < prevConnectedObject->id) {
                    addTriangle(object, index, prevConnectedObject, connectedObject);
                }
            }

            // Rectangle?
            auto fourthCellCandidate1 = connectedObject->getConnectedObject(backIndex + 1);
            auto fourthCellCandidate2 = prevConnectedObject->getConnectedObject(prevBackIndex - 1);
            if (fourthCellCandidate2 == fourthCellCandidate1 && fourthCellCandidate1 != object && fourthCellCandidate2 != object
                && connectedObject != prevConnectedObject) {
                if (object->id < connectedObject->id && object->id < prevConnectedObject->id && object->id < fourthCellCandidate2->id) {
                    addTriangle(object, index, connectedObject, fourthCellCandidate1);
                    addTriangle(object, index, fourthCellCandidate1, prevConnectedObject);
                }
            }
        }
    }
}

__global__ void cudaExtractBlurryParticleData(SimulationData data, BlurryParticleVertexData* blurryParticleData, uint64_t* numBlurryParticles)
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

            float intensity = (energy->energy + 5.0f) / 200.0f;
            if (energy->selected) {
                intensity *= 2.5f;
            }

            auto idx = alienAtomicAdd64(numBlurryParticles, uint64_t(1));
            if (blurryParticleData != nullptr) {
                blurryParticleData[idx].pos[0] = pos.x;
                blurryParticleData[idx].pos[1] = pos.y;
                blurryParticleData[idx].pos[2] = 0.0f;
                blurryParticleData[idx].color[0] = intensity * 0.25f;
                blurryParticleData[idx].color[1] = intensity * 0.25f;
                blurryParticleData[idx].color[2] = intensity * 1.0f;
            }
        }
    }

    // Process isolated structure objects (no connections) - render them like energy particles
    {
        auto const& objectPartition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
        for (int index = objectPartition.startIndex; index <= objectPartition.endIndex; index += objectPartition.step) {
            auto const& object = data.entities.objects.at(index);
            if (object->type != ObjectType_Structure || object->numConnections != 0) {
                continue;
            }

            float intensity = (object->typeData.structure.energy + 5.0f) / 200.0f;
            if (object->selected) {
                intensity *= 2.5f;
            }

            auto idx = alienAtomicAdd64(numBlurryParticles, uint64_t(1));
            if (blurryParticleData != nullptr) {
                blurryParticleData[idx].pos[0] = object->pos.x;
                blurryParticleData[idx].pos[1] = object->pos.y;
                blurryParticleData[idx].pos[2] = 0.0f;
                blurryParticleData[idx].color[0] = intensity * 0.25f;
                blurryParticleData[idx].color[1] = intensity * 0.25f;
                blurryParticleData[idx].color[2] = intensity * 1.0f;
            }
        }
    }
}

__global__ void cudaExtractLocationData(SimulationData data, LocationVertexData* locationData, uint64_t* numLocations, float2 visibleTopLeft)
{
    // Extract location data for layers and sources
    // Each location is rendered 5 times (normal + 4 world offsets for periodic boundaries)
    auto worldSize = data.worldSize;

    // Process layers
    for (int i = 0; i < cudaSimulationParameters.numLayers; ++i) {
        auto const& pos_asRealVector2D = cudaSimulationParameters.layerPosition.layerValues[i];
        float2 pos{pos_asRealVector2D.x, pos_asRealVector2D.y};
        correctPositionForRendering(pos, visibleTopLeft, data.worldSize);
        auto color = cudaSimulationParameters.backgroundColor.layerValues[i].value;
        auto shapeType = cudaSimulationParameters.layerShape.layerValues[i];
        auto radius = cudaSimulationParameters.layerCoreRadius.layerValues[i];
        auto rect = cudaSimulationParameters.layerCoreRect.layerValues[i];
        auto fadeoutRadius = cudaSimulationParameters.layerFadeoutRadius.layerValues[i];
        auto opacity = cudaSimulationParameters.layerOpacity.layerValues[i];

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
        correctPositionForRendering(pos, visibleTopLeft, data.worldSize);

        // Use a distinct color for sources (e.g., yellow)
        float3 color = {0.05f, 0.05f, 0.15f};
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
            }
            ++(*numLocations);
        }
    }
}

__global__ void cudaExtractSelectedObjectData(SimulationData data, SelectedObjectVertexData* selectedObjectData, uint64_t* numSelectedObjects)
{
    // Process selected cells
    auto const& objects = data.entities.objects;
    auto numObjects = objects.getNumEntries();

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < numObjects; index += blockDim.x * gridDim.x) {
        auto const& object = objects.at(index);
        if (object->selected == 1) {
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
        if (energy->selected == 1) {
            auto outputIndex = alienAtomicAdd64(numSelectedObjects, static_cast<uint64_t>(1));
            if (selectedObjectData != nullptr) {
                selectedObjectData[outputIndex].pos[0] = energy->pos.x;
                selectedObjectData[outputIndex].pos[1] = energy->pos.y;
            }
        }
    }
}

__global__ void cudaExtractSelectedConnectionData(SimulationData data, ConnectionArrowVertexData* connectionArrowData, uint64_t* numConnectionArrowVertices)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (object->selected == 0) {
            continue;
        }

        // Process each connection from this cell
        for (int i = 0; i < object->numConnections; ++i) {
            auto connectedObject = object->connections[i].object;

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
            auto cellColor = getCellColor(object);
            auto connectedObjectColor = getCellColor(connectedObject);

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
__global__ void cudaExtractAttackEventData(SimulationData data, AttackEventVertexData* attackEventData, uint64_t* numAttackEventVertices)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);

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
__global__ void cudaExtractDetonationEventData(SimulationData data, DetonationEventVertexData* detonationEventData, uint64_t* numDetonationEventVertices)
{
    auto const& partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& object = data.entities.objects.at(index);
        if (object->type != ObjectType_Cell) {
            continue;
        }

        // Only process cells that have detonation event
        if (object->typeData.cell.eventCounter > 0 && object->typeData.cell.event == CellEvent_Detonation) {

            // Add detonation event point data (1 vertex for the circle center)
            uint64_t vertexIndex = alienAtomicAdd64(numDetonationEventVertices, uint64_t(1));
            if (detonationEventData != nullptr) {
                // Position of the detonation
                detonationEventData[vertexIndex].pos[0] = object->pos.x;
                detonationEventData[vertexIndex].pos[1] = object->pos.y;

                // Radius proportional to eventCounter
                // Scale the radius based on eventCounter (make it visible)
                detonationEventData[vertexIndex].radius = toFloat(object->typeData.cell.eventCounter * object->typeData.cell.eventCounter) / 3.0f;
            }
        }
    }
}
