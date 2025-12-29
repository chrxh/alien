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
        auto const& partition = calcSystemThreadPartitionNew(data.objects.cells.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& cell = data.objects.cells.at(index);
            correctPositionForRendering(cell->pos, visibleTopLeft, data.worldSize);
        }
    }
    {
        auto const& partition = calcSystemThreadPartitionNew(data.objects.particles.getNumEntries());
        for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
            auto const& particle = data.objects.particles.at(index);
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
        return result;
    }
}

__global__ void cudaExtractCellData(SimulationData data, CellVertexData* objectData)
{
    // Process cells - each cell goes to its index position
    auto const& cellPartition = calcSystemThreadPartitionNew(data.objects.cells.getNumEntries());
    for (int index = cellPartition.startIndex; index <= cellPartition.endIndex; index += cellPartition.step) {
        auto const& cell = data.objects.cells.at(index);

        int isInTriangleOrQuad = 0;
        if (cell->numConnections > 1) {
            bool first = true;
            int backIndices[MAX_CELL_BONDS];
            for (int i = 0, numConnections = cell->numConnections; i < numConnections + 1; ++i) {
                auto connectionIndex = i % numConnections;
                auto const& connectedCell = cell->connections[connectionIndex].cell;
                auto backIndex = connectedCell->getConnectionIndex(cell);
                backIndices[connectionIndex] = backIndex;
                if (first) {
                    first = false;
                    continue;
                }
                auto prevIndex = (connectionIndex + numConnections - 1) % numConnections;
                auto const& prevConnectedCell = cell->connections[prevIndex].cell;
                auto prevBackIndex = backIndices[prevIndex];

                // Triangle?
                if (prevConnectedCell->getConnectedCell(prevBackIndex - 1) == connectedCell) {
                    isInTriangleOrQuad = 1;
                    break;
                }

                // Rectangle?
                auto fourthCellCandidate1 = connectedCell->getConnectedCell(backIndex + 1);
                auto fourthCellCandidate2 = prevConnectedCell->getConnectedCell(prevBackIndex - 1);
                if (fourthCellCandidate2 == fourthCellCandidate1 && fourthCellCandidate1 != cell && fourthCellCandidate2 != cell
                    && connectedCell != prevConnectedCell) {
                    isInTriangleOrQuad = 1;
                    break;
                }
            }
        }


        auto const& pos = cell->pos;

        auto const& cellColor = getCellColor(cell->color);

        auto luminance = cell->getEnergy() / 300.0f;  //1.0f - 50000.0f / (cell->energy * cell->energy + 50000.0f);
        auto white = luminance / 10.0f;
        if (cell->selected == 1) {
            luminance = (luminance + 0.1f) * 1.7f;
            white = 0.5f;
        }
        if (cell->selected == 2) {
            luminance = luminance * 1.3f;
        }

        // Calculate deterministic z-position based on cell id for lighting
        // Use a simple hash function to get a pseudo-random value in range [0, 1]
        uint64_t hash = cell->id * 2654435761u;  // Knuth's multiplicative hash
        hash = (hash ^ (hash >> 16)) * 0x85ebca6b;
        hash = (hash ^ (hash >> 13)) * 0xc2b2ae35;
        hash = hash ^ (hash >> 16);
        float normalizedHash = toFloat(hash & 0xFFFFFF) / toFloat(0xFFFFFF);
        float zPos = normalizedHash * 0.05f;

        auto zOffset = cell->creature != nullptr ? toFloat(cell->creature->id % 1000) / 2000 : 0.0f;

        // Write cell data at cell index position
        objectData[index].pos[0] = pos.x;
        objectData[index].pos[1] = pos.y;
        objectData[index].pos[2] = zPos + zOffset;
        objectData[index].color[0] = toFloat((cellColor >> 16) & 0xff) / 255.0f * luminance + white;
        objectData[index].color[1] = toFloat((cellColor >> 8) & 0xff) / 255.0f * luminance + white;
        objectData[index].color[2] = toFloat(cellColor & 0xff) / 255.0f * luminance + white;

        // Pack both cellType (lower 8 bits) and signalState (upper 8 bits) into state field
        objectData[index].state = cell->cellType | (cell->signalState << 8) | (isInTriangleOrQuad << 16);

        // Store cell index for line extraction (just use the index directly)
        cell->tempValue.as_uint64 = index;
    }
}

__global__ void cudaExtractLineIndices(SimulationData data, unsigned int* lineIndices, uint64_t* numLineIndices)
{
    auto const& partition = calcSystemThreadPartitionNew(data.objects.cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& cell = data.objects.cells.at(index);

        // Cell index is just the array index (stored in tempValue)
        uint64_t cellIndex = cell->tempValue.as_uint64;

        // Add line indices for each connection
        for (int i = 0; i < cell->numConnections; ++i) {
            auto connectedCell = cell->connections[i].cell;

            // Only add each connection once (from lower index to higher index to avoid duplicates)
            if (cell->id < connectedCell->id) {
                if (Math::length(cell->pos - connectedCell->pos) <= cudaSimulationParameters.maxBindingDistance.value[cell->color]) {
                    uint64_t lineIndex = alienAtomicAdd64(numLineIndices, uint64_t(2));
                    if (lineIndices != nullptr) {
                        uint64_t connectedIndex = connectedCell->tempValue.as_uint64;
                        lineIndices[lineIndex] = static_cast<unsigned int>(cellIndex);
                        lineIndices[lineIndex + 1] = static_cast<unsigned int>(connectedIndex);
                    }
                }
            }
        }
    }
}

__global__ void cudaExtractTriangleIndices(SimulationData data, unsigned int* triangleIndices, uint64_t* numTriangleIndices)
{
    auto const& partition = calcSystemThreadPartitionNew(data.objects.cells.getNumEntries());

    auto addTriangle = [&](Cell* cell, uint64_t cellIndex, Cell* connectedCell, Cell* prevConnectedCell) {
        // Only add triangle once (avoid duplicates by checking ids)
        if (Math::length(cell->pos - connectedCell->pos) <= cudaSimulationParameters.maxBindingDistance.value[cell->color]
            && Math::length(cell->pos - prevConnectedCell->pos) <= cudaSimulationParameters.maxBindingDistance.value[cell->color]
            && Math::length(connectedCell->pos - prevConnectedCell->pos) <= cudaSimulationParameters.maxBindingDistance.value[connectedCell->color]) {
            uint64_t connectedIndex1 = connectedCell->tempValue.as_uint64;
            uint64_t connectedIndex2 = prevConnectedCell->tempValue.as_uint64;
            uint64_t triangleIndex = alienAtomicAdd64(numTriangleIndices, uint64_t(3));
            if (triangleIndices != nullptr) {
                triangleIndices[triangleIndex] = static_cast<unsigned int>(cellIndex);
                triangleIndices[triangleIndex + 1] = static_cast<unsigned int>(connectedIndex1);
                triangleIndices[triangleIndex + 2] = static_cast<unsigned int>(connectedIndex2);
            }
        }
    };
    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& cell = data.objects.cells.at(index);
        if (cell->numConnections <= 1) {
            continue;
        }
        bool first = true;
        int backIndices[MAX_CELL_BONDS];
        for (int i = 0, numConnections = cell->numConnections; i < numConnections + 1; ++i) {
            auto connectionIndex = i % numConnections;
            auto const& connectedCell = cell->connections[connectionIndex].cell;
            auto backIndex = connectedCell->getConnectionIndex(cell);
            backIndices[connectionIndex] = backIndex;
            if (first) {
                first = false;
                continue;
            }
            auto prevIndex = (connectionIndex + numConnections - 1) % numConnections;
            auto const& prevConnectedCell = cell->connections[prevIndex].cell;
            auto prevBackIndex = backIndices[prevIndex];

            // Triangle?
            if (prevConnectedCell->getConnectedCell(prevBackIndex - 1) == connectedCell) {
                if (cell->id < connectedCell->id && cell->id < prevConnectedCell->id) {
                    addTriangle(cell, index, prevConnectedCell, connectedCell);
                }
            }

            // Rectangle?
            auto fourthCellCandidate1 = connectedCell->getConnectedCell(backIndex + 1);
            auto fourthCellCandidate2 = prevConnectedCell->getConnectedCell(prevBackIndex - 1);
            if (fourthCellCandidate2 == fourthCellCandidate1 && fourthCellCandidate1 != cell && fourthCellCandidate2 != cell
                && connectedCell != prevConnectedCell) {
                if (cell->id < connectedCell->id && cell->id < prevConnectedCell->id && cell->id < fourthCellCandidate2->id) {
                    addTriangle(cell, index, connectedCell, fourthCellCandidate1);
                    addTriangle(cell, index, fourthCellCandidate1, prevConnectedCell);
                }
            }
        }
    }
}

__global__ void cudaExtractEnergyParticleData(SimulationData data, EnergyParticleVertexData* energyParticleData)
{
    // Process energy particles - each particle goes to its index position
    auto const& particlePartition = calcSystemThreadPartitionNew(data.objects.particles.getNumEntries());
    for (int index = particlePartition.startIndex; index <= particlePartition.endIndex; index += particlePartition.step) {
        auto const& particle = data.objects.particles.at(index);
        if (!particle) {
            continue;
        }

        auto pos = particle->pos;

        // Light yellow color for energy particles
        float intensity = (particle->energy + 5.0f) / 200.0f;
        //max(min((particle->energy + 10.0f) * 5, 450.0f), 20.0f) / 1000.0f;
        if (particle->selected) {
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
    auto const& cells = data.objects.cells;
    auto numCells = cells.getNumEntries();

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < numCells; index += blockDim.x * gridDim.x) {
        auto const& cell = cells.at(index);
        if (cell->selected == 1) {
            auto outputIndex = alienAtomicAdd64(numSelectedObjects, static_cast<uint64_t>(1));
            if (selectedObjectData != nullptr) {
                selectedObjectData[outputIndex].pos[0] = cell->pos.x;
                selectedObjectData[outputIndex].pos[1] = cell->pos.y;

                // Calculate signal angle restrictions for this cell
                // The 180° offset converts from connection-relative to absolute angles in world space
                if (cell->signalRestriction.active && cell->numConnections > 0) {
                    auto const& connectedCell = cell->connections[0].cell;
                    auto connectionAngle = Math::angleOfVector(connectedCell->pos - cell->pos);

                    auto signalAngleRestrictionStart = connectionAngle + 180.0f + cell->signalRestriction.baseAngle - cell->signalRestriction.openingAngle / 2;
                    auto signalAngleRestrictionEnd = connectionAngle + 180.0f + cell->signalRestriction.baseAngle + cell->signalRestriction.openingAngle / 2;
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
    auto const& particles = data.objects.particles;
    auto numParticles = particles.getNumEntries();

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < numParticles; index += blockDim.x * gridDim.x) {
        auto const& particle = particles.at(index);
        if (particle->selected == 1) {
            auto outputIndex = alienAtomicAdd64(numSelectedObjects, static_cast<uint64_t>(1));
            if (selectedObjectData != nullptr) {
                selectedObjectData[outputIndex].pos[0] = particle->pos.x;
                selectedObjectData[outputIndex].pos[1] = particle->pos.y;
                selectedObjectData[outputIndex].hasSignalRestriction = 0;
                selectedObjectData[outputIndex].startAngle = 0.0f;
                selectedObjectData[outputIndex].endAngle = 0.0f;
            }
        }
    }
}

__global__ void cudaExtractSelectedConnectionData(SimulationData data, ConnectionArrowVertexData* connectionArrowData, uint64_t* numConnectionArrowVertices)
{
    auto const& partition = calcSystemThreadPartitionNew(data.objects.cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& cell = data.objects.cells.at(index);
        if (cell->selected == 0) {
            continue;
        }

        // Calculate signal angle restrictions for this cell
        auto signalAngleRestrictionStart = 180.0f + cell->signalRestriction.baseAngle - cell->signalRestriction.openingAngle / 2;
        auto signalAngleRestrictionEnd = 180.0f + cell->signalRestriction.baseAngle + cell->signalRestriction.openingAngle / 2;
        signalAngleRestrictionStart = Math::getNormalizedAngle(signalAngleRestrictionStart, 0.0f);
        signalAngleRestrictionEnd = Math::getNormalizedAngle(signalAngleRestrictionEnd, 0.0f);

        auto summedAngle = 0.0f;

        // Process each connection from this cell
        for (int i = 0; i < cell->numConnections; ++i) {
            if (i > 0) {
                summedAngle += cell->connections[i].angleFromPrevious;
            }

            auto connectedCell = cell->connections[i].cell;

            // Only add each connection once (from lower id to higher id to avoid duplicates)
            if (cell->id >= connectedCell->id) {
                continue;
            }

            // Check if this connection should be drawn
            if (Math::length(cell->pos - connectedCell->pos) > cudaSimulationParameters.maxBindingDistance.value[cell->color]) {
                continue;
            }

            // Determine if signal can flow from cell1 to cell2
            bool arrowToCell2 =
                !cell->signalRestriction.active || Math::isAngleStrictInBetween(signalAngleRestrictionStart, signalAngleRestrictionEnd, summedAngle);

            // Determine if signal can flow from cell2 to cell1
            // Need to calculate the reverse angle from connectedCell's perspective
            auto signalAngleRestrictionStart2 = 180.0f + connectedCell->signalRestriction.baseAngle - connectedCell->signalRestriction.openingAngle / 2;
            auto signalAngleRestrictionEnd2 = 180.0f + connectedCell->signalRestriction.baseAngle + connectedCell->signalRestriction.openingAngle / 2;
            signalAngleRestrictionStart2 = Math::getNormalizedAngle(signalAngleRestrictionStart2, 0.0f);
            signalAngleRestrictionEnd2 = Math::getNormalizedAngle(signalAngleRestrictionEnd2, 0.0f);

            // Find the angle of this connection from connectedCell's perspective
            auto summedAngle2 = 0.0f;
            bool arrowToCell1 = false;
            for (int j = 0; j < connectedCell->numConnections; ++j) {
                if (j > 0) {
                    summedAngle2 += connectedCell->connections[j].angleFromPrevious;
                }
                if (connectedCell->connections[j].cell->id == cell->id) {
                    arrowToCell1 = !connectedCell->signalRestriction.active
                        || Math::isAngleStrictInBetween(signalAngleRestrictionStart2, signalAngleRestrictionEnd2, summedAngle2);
                    break;
                }
            }

            // Get cell colors
            auto cellColor = getCellColor(cell->color);
            auto connectedCellColor = getCellColor(connectedCell->color);

            // Encode arrow direction in flags: bit 0 = arrow to cell1, bit 1 = arrow to cell2
            int arrowFlags = (arrowToCell1 ? 1 : 0) | (arrowToCell2 ? 2 : 0);

            // Add connection arrow data (2 vertices for the line)
            uint64_t vertexIndex = alienAtomicAdd64(numConnectionArrowVertices, uint64_t(2));
            if (connectionArrowData != nullptr) {

                // First vertex (cell1)
                connectionArrowData[vertexIndex].pos[0] = cell->pos.x;
                connectionArrowData[vertexIndex].pos[1] = cell->pos.y;
                connectionArrowData[vertexIndex].color[0] = toFloat((cellColor >> 16) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex].color[1] = toFloat((cellColor >> 8) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex].color[2] = toFloat((cellColor >> 0) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex].arrowFlags = arrowFlags;

                // Second vertex (cell2)
                connectionArrowData[vertexIndex + 1].pos[0] = connectedCell->pos.x;
                connectionArrowData[vertexIndex + 1].pos[1] = connectedCell->pos.y;
                connectionArrowData[vertexIndex + 1].color[0] = toFloat((connectedCellColor >> 16) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex + 1].color[1] = toFloat((connectedCellColor >> 8) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex + 1].color[2] = toFloat((connectedCellColor >> 0) & 0xff) / 255.0f;
                connectionArrowData[vertexIndex + 1].arrowFlags = arrowFlags;
            }
        }
    }
}
__global__ void cudaExtractAttackEventData(SimulationData data, AttackEventVertexData* attackEventData, uint64_t* numAttackEventVertices)
{
    auto const& partition = calcSystemThreadPartitionNew(data.objects.cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& cell = data.objects.cells.at(index);

        // Only process cells that have been attacked and have attackVisualization enabled
        if (cell->eventCounter > 0 && cell->event == CellEvent_Attacked) {

            // Check if the attacker position is close enough to draw
            if (Math::length(cell->eventPos - cell->pos) < 10.0f) {
                // Add attack event line data (2 vertices for the line: from attacker to attacked)
                uint64_t vertexIndex = alienAtomicAdd64(numAttackEventVertices, uint64_t(2));
                if (attackEventData != nullptr) {
                    // Red color for attacked event
                    float redColor[3] = {0.5f, 0.0f, 0.0f};

                    // First vertex (attacker position - from eventPos)
                    attackEventData[vertexIndex].pos[0] = cell->eventPos.x;
                    attackEventData[vertexIndex].pos[1] = cell->eventPos.y;
                    attackEventData[vertexIndex].color[0] = redColor[0];
                    attackEventData[vertexIndex].color[1] = redColor[1];
                    attackEventData[vertexIndex].color[2] = redColor[2];

                    // Second vertex (attacked cell position)
                    attackEventData[vertexIndex + 1].pos[0] = cell->pos.x;
                    attackEventData[vertexIndex + 1].pos[1] = cell->pos.y;
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
    auto const& partition = calcSystemThreadPartitionNew(data.objects.cells.getNumEntries());

    for (int index = partition.startIndex; index <= partition.endIndex; index += partition.step) {
        auto const& cell = data.objects.cells.at(index);

        // Only process cells that have detonation event
        if (cell->eventCounter > 0 && cell->event == CellEvent_Detonation) {

            // Add detonation event point data (1 vertex for the circle center)
            uint64_t vertexIndex = alienAtomicAdd64(numDetonationEventVertices, uint64_t(1));
            if (detonationEventData != nullptr) {
                // Position of the detonation
                detonationEventData[vertexIndex].pos[0] = cell->pos.x;
                detonationEventData[vertexIndex].pos[1] = cell->pos.y;

                // Radius proportional to eventCounter
                // Scale the radius based on eventCounter (make it visible)
                detonationEventData[vertexIndex].radius = toFloat(cell->eventCounter * cell->eventCounter) / 3.0f;
            }
        }
    }
}
