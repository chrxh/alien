#include "GeometryKernels.cuh"
#include "ParameterCalculator.cuh"
#include "SignalProcessor.cuh"

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
    __device__ __inline__ uint32_t getCellColor(int colorCode)
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

        auto const& cellColor = getCellColor(object->color);

        float luminance;
        float zOffset = 0.0f;
        int cellType = 0;
        int signalState = 0;

        if (object->type == ObjectType_Cell) {
            luminance = object->typeData.cell.getEnergy() / 300.0f;
            zOffset = toFloat(object->typeData.cell.creature->id % 1000) / 2000;
            cellType = object->typeData.cell.cellType;
            signalState = object->typeData.cell.signalState;
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

        // Calculate deterministic z-position based on cell id for lighting
        // Use a simple hash function to get a pseudo-random value in range [0, 1]
        uint64_t hash = object->id * 2654435761u;  // Knuth's multiplicative hash
        hash = (hash ^ (hash >> 16)) * 0x85ebca6b;
        hash = (hash ^ (hash >> 13)) * 0xc2b2ae35;
        hash = hash ^ (hash >> 16);
        float normalizedHash = toFloat(hash & 0xFFFFFF) / toFloat(0xFFFFFF);
        float zPos = normalizedHash * 0.05f;

        // Write cell data at cell index position
        objectData[index].pos[0] = pos.x;
        objectData[index].pos[1] = pos.y;
        objectData[index].pos[2] = zPos + zOffset;
        objectData[index].color[0] = toFloat((cellColor >> 16) & 0xff) / 255.0f * luminance + white;
        objectData[index].color[1] = toFloat((cellColor >> 8) & 0xff) / 255.0f * luminance + white;
        objectData[index].color[2] = toFloat(cellColor & 0xff) / 255.0f * luminance + white;

        // Pack cellType (bits 0-7), objectType (bits 8-15), signalState (bits 16-23), and isInTriangleOrQuad (bit 24) into state field
        objectData[index].state = cellType | (object->type << 8) | (signalState << 16) | (isInTriangleOrQuad << 24);

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

__global__ void cudaExtractEnergyData(SimulationData data, EnergyVertexData* energyParticleData)
{
    // Process energy particles - each particle goes to its index position
    auto const& energyPartition = calcSystemThreadPartition(data.entities.energies.getNumEntries());
    for (int index = energyPartition.startIndex; index <= energyPartition.endIndex; index += energyPartition.step) {
        auto const& energy = data.entities.energies.at(index);
        if (!energy) {
            continue;
        }

        auto pos = energy->pos;

        // Light yellow color for energy particles
        float intensity = (energy->energy + 5.0f) / 200.0f;
        //max(min((particle->energy + 10.0f) * 5, 450.0f), 20.0f) / 1000.0f;
        if (energy->selected) {
            intensity *= 2.5f;
        }

        // Write energy particle data
        energyParticleData[index].pos[0] = pos.x;
        energyParticleData[index].pos[1] = pos.y;
        energyParticleData[index].pos[2] = 0.0f;                 // Energy particles don't need z-position for lighting
        energyParticleData[index].color[0] = intensity * 0.25f;  // Red component
        energyParticleData[index].color[1] = intensity * 0.25f;  // Green component
        energyParticleData[index].color[2] = intensity * 1.0f;   // Blue component (reduced for yellow tint)
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

                // Calculate signal angle restrictions for this cell
                // The 180° offset converts from connection-relative to absolute angles in world space
                // Render as active if mode is Active or Conditional
                bool hasRestriction = object->type == ObjectType_Cell
                    && (object->typeData.cell.signalRestriction.mode == SignalRestrictionMode_Active
                        || object->typeData.cell.signalRestriction.mode == SignalRestrictionMode_Conditional)
                    && object->numConnections > 0;
                if (hasRestriction) {
                    auto const& connectedObject = object->connections[0].object;
                    auto connectionAngle = Math::angleOfVector(connectedObject->pos - object->pos);

                    auto signalAngleRestrictionStart =
                        connectionAngle + 180.0f + object->typeData.cell.signalRestriction.baseAngle - object->typeData.cell.signalRestriction.openingAngle / 2;
                    auto signalAngleRestrictionEnd =
                        connectionAngle + 180.0f + object->typeData.cell.signalRestriction.baseAngle + object->typeData.cell.signalRestriction.openingAngle / 2;
                    signalAngleRestrictionStart = Math::getNormalizedAngle(signalAngleRestrictionStart, 0.0f);
                    signalAngleRestrictionEnd = Math::getNormalizedAngle(signalAngleRestrictionEnd, 0.0f);

                    selectedObjectData[outputIndex].hasSignalRestriction = 1;
                    selectedObjectData[outputIndex].startAngle = signalAngleRestrictionStart;
                    selectedObjectData[outputIndex].endAngle = signalAngleRestrictionEnd;
                } else {
                    selectedObjectData[outputIndex].hasSignalRestriction = 0;
                    selectedObjectData[outputIndex].startAngle = 0.0f;
                    selectedObjectData[outputIndex].endAngle = 0.0f;
                }
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
                selectedObjectData[outputIndex].hasSignalRestriction = 0;
                selectedObjectData[outputIndex].startAngle = 0.0f;
                selectedObjectData[outputIndex].endAngle = 0.0f;
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

        // Calculate signal angle restrictions for this cell
        float signalAngleRestrictionStart = 0, signalAngleRestrictionEnd = 0;
        if (object->type == ObjectType_Cell) {
            signalAngleRestrictionStart = 180.0f + object->typeData.cell.signalRestriction.baseAngle - object->typeData.cell.signalRestriction.openingAngle / 2;
            signalAngleRestrictionEnd = 180.0f + object->typeData.cell.signalRestriction.baseAngle + object->typeData.cell.signalRestriction.openingAngle / 2;
            signalAngleRestrictionStart = Math::getNormalizedAngle(signalAngleRestrictionStart, 0.0f);
            signalAngleRestrictionEnd = Math::getNormalizedAngle(signalAngleRestrictionEnd, 0.0f);
        }

        auto summedAngle = 0.0f;

        // Process each connection from this cell
        for (int i = 0; i < object->numConnections; ++i) {
            if (i > 0) {
                summedAngle += object->connections[i].angleFromPrevious;
            }

            auto connectedObject = object->connections[i].object;

            // Only add each connection once (from lower id to higher id to avoid duplicates)
            if (object->id >= connectedObject->id) {
                continue;
            }

            // Check if this connection should be drawn
            if (Math::length(object->pos - connectedObject->pos) > cudaSimulationParameters.maxBindingDistance.value[object->color]) {
                continue;
            }

            bool arrowToCell1 = false;
            bool arrowToCell2 = false;
            if (object->type == ObjectType_Cell) {
                // Determine if signal can flow from object1 to object2
                // For rendering, Active and Conditional modes are treated as having restriction
                bool hasRestriction1 =
                    (object->typeData.cell.signalRestriction.mode == SignalRestrictionMode_Active
                     || object->typeData.cell.signalRestriction.mode == SignalRestrictionMode_Conditional);
                arrowToCell2 = !hasRestriction1 || Math::isAngleStrictInBetween(signalAngleRestrictionStart, signalAngleRestrictionEnd, summedAngle);

                // Determine if signal can flow from object2 to object1
                // Need to calculate the reverse angle from connectedObject's perspective
                auto signalAngleRestrictionStart2 =
                    180.0f + connectedObject->typeData.cell.signalRestriction.baseAngle - connectedObject->typeData.cell.signalRestriction.openingAngle / 2;
                auto signalAngleRestrictionEnd2 =
                    180.0f + connectedObject->typeData.cell.signalRestriction.baseAngle + connectedObject->typeData.cell.signalRestriction.openingAngle / 2;
                signalAngleRestrictionStart2 = Math::getNormalizedAngle(signalAngleRestrictionStart2, 0.0f);
                signalAngleRestrictionEnd2 = Math::getNormalizedAngle(signalAngleRestrictionEnd2, 0.0f);

                // Find the angle of this connection from connectedObject's perspective
                auto summedAngle2 = 0.0f;
                bool hasRestriction2 =
                    (connectedObject->typeData.cell.signalRestriction.mode == SignalRestrictionMode_Active
                     || connectedObject->typeData.cell.signalRestriction.mode == SignalRestrictionMode_Conditional);
                for (int j = 0; j < connectedObject->numConnections; ++j) {
                    if (j > 0) {
                        summedAngle2 += connectedObject->connections[j].angleFromPrevious;
                    }
                    if (connectedObject->connections[j].object->id == object->id) {
                        arrowToCell1 = !hasRestriction2 || Math::isAngleStrictInBetween(signalAngleRestrictionStart2, signalAngleRestrictionEnd2, summedAngle2);
                        break;
                    }
                }
            }

            // Get cell colors
            auto cellColor = getCellColor(object->color);
            auto connectedObjectColor = getCellColor(connectedObject->color);

            // Encode arrow direction in flags: bit 0 = arrow to object1, bit 1 = arrow to object2
            int arrowFlags = (arrowToCell1 ? 1 : 0) | (arrowToCell2 ? 2 : 0);

            // Add connection arrow data (2 vertices for the line)
            uint64_t vertexIndex = alienAtomicAdd64(numConnectionArrowVertices, uint64_t(2));
            if (connectionArrowData != nullptr) {

                // First vertex (object1)
                connectionArrowData[vertexIndex].pos[0] = object->pos.x;
                connectionArrowData[vertexIndex].pos[1] = object->pos.y;
                connectionArrowData[vertexIndex].color[0] = toFloat((cellColor >> 16) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex].color[1] = toFloat((cellColor >> 8) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex].color[2] = toFloat((cellColor >> 0) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex].arrowFlags = arrowFlags;

                // Second vertex (object2)
                connectionArrowData[vertexIndex + 1].pos[0] = connectedObject->pos.x;
                connectionArrowData[vertexIndex + 1].pos[1] = connectedObject->pos.y;
                connectionArrowData[vertexIndex + 1].color[0] = toFloat((connectedObjectColor >> 16) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex + 1].color[1] = toFloat((connectedObjectColor >> 8) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex + 1].color[2] = toFloat((connectedObjectColor >> 0) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex + 1].arrowFlags = arrowFlags;
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
