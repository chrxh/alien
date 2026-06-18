#pragma once

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/ShapeGenerator.h>

#include "CellProcessor.cuh"

class ConstructorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics, bool isPreview);
    __inline__ __device__ static void countConstructorsNeedingEnergy(SimulationData& data);
    __inline__ __device__ static void provideExternalEnergy(SimulationData& data);

private:
    struct ConstructionData
    {
        // Creature and genome data
        Creature* creature;
        Gene* gene;
        Node* node;
        bool isSeparation;

        // Construction position
        uint16_t currentNodeIndex;
        uint32_t currentConcatenation;
        uint8_t currentBranch;
        bool isFirstNode;
        bool isFirstNodeOfFirstConcatenation;
        bool isLastNode;
        bool isLastNodeOfLastConcatenation;
        bool hasInfiniteConcatenations;

        // Construction data
        Object* lastConstructionObject;
        ShapeGeneratorResult shapeResult;
        float neededUsableEnergy;
        float neededReservedEnergy;
        float neededDepotEnergy;
    };
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object, bool isPreview);
    __inline__ __device__ static Creature* findOrCreateNewCreature(SimulationData& data, Object* object);
    __inline__ __device__ static ConstructionData createConstructionData(Object* object);

    __inline__ __device__ static Object*
    tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Object* hostObject, ConstructionData const& constructionData);

    __inline__ __device__ static Object*
    startConstructionOnNewBranch(SimulationData& data, SimulationStatistics& statistics, Object* hostObject, ConstructionData const& constructionData);
    __inline__ __device__ static Object*
    continueConstructionOnBranch(SimulationData& data, SimulationStatistics& statistics, Object* hostObject, ConstructionData const& constructionData);

    __inline__ __device__ static void getObjectsToConnect(
        Object* result[],
        int& numResultCells,
        SimulationData& data,
        Object* hostObject,
        float2 const& newObjectPos,
        ConstructionData const& constructionData);

    __inline__ __device__ static Object* constructCellIntern(
        SimulationData& data,
        SimulationStatistics& statistics,
        uint64_t& objectIndex,
        Object* hostObject,
        float2 newObjectPos,
        ConstructionData const& constructionData);

    __inline__ __device__ static bool checkAndReduceHostEnergy(SimulationData& data, Object* hostObject, ConstructionData const& constructionData);
    __inline__ __device__ static bool isExternalEnergyInflowAllowed(Object const* hostObject);
    __inline__ __device__ static void activateNewObjectOnLastNode(Object* newObject, Object* hostObject, ConstructionData const& constructionData);
    __inline__ __device__ static void setHeadCellOnFirstNode(Object* newObject, Object* hostObject, ConstructionData const& constructionData);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ConstructorProcessor::process(SimulationData& data, SimulationStatistics& statistics, bool isPreview)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumOrigEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        auto object = data.entities.objects.at(i);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        if (!object->typeData.cell.constructorAvailable) {
            continue;
        }
        object->typeData.cell.constructor.energyNeeded = false;
        if (!CellProcessor::isCellReady(data, object)) {
            continue;
        }
        processCell(data, statistics, object, isPreview);
    }
}

__inline__ __device__ void ConstructorProcessor::countConstructorsNeedingEnergy(SimulationData& data)
{
    __shared__ uint32_t sharedNumConstructorsNeedingEnergyByColor[MAX_COLORS];
    for (int color = threadIdx.x; color < MAX_COLORS; color += blockDim.x) {
        sharedNumConstructorsNeedingEnergyByColor[color] = 0;
    }
    __syncthreads();

    uint32_t numConstructorsNeedingEnergyByColor[MAX_COLORS];
    for (int color = 0; color < MAX_COLORS; ++color) {
        numConstructorsNeedingEnergyByColor[color] = 0;
    }

    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumOrigEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        auto object = data.entities.objects.at(i);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto const& cell = object->typeData.cell;
        if (cell.constructorAvailable && cell.constructor.energyNeeded) {
            ++numConstructorsNeedingEnergyByColor[object->color];
        }
    }

    for (int color = 0; color < MAX_COLORS; ++color) {
        if (numConstructorsNeedingEnergyByColor[color] > 0) {
            atomicAdd_block(&sharedNumConstructorsNeedingEnergyByColor[color], numConstructorsNeedingEnergyByColor[color]);
        }
    }
    __syncthreads();

    for (int color = threadIdx.x; color < MAX_COLORS; color += blockDim.x) {
        if (sharedNumConstructorsNeedingEnergyByColor[color] > 0) {
            atomicAdd(&data.numConstructorsNeedingEnergyByColor[color], sharedNumConstructorsNeedingEnergyByColor[color]);
        }
    }
}

__inline__ __device__ void ConstructorProcessor::provideExternalEnergy(SimulationData& data)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumOrigEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        auto object = data.entities.objects.at(i);
        if (object->type != ObjectType_Cell) {
            continue;
        }
        auto& cell = object->typeData.cell;
        if (cell.constructorAvailable && cell.constructor.energyNeeded) {
            cell.usableEnergy += data.externalEnergyInflowPerConstructorByColor[object->color];
        }
    }
}

__inline__ __device__ void ConstructorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object, bool isPreview)
{
    auto& constructor = object->typeData.cell.constructor;
    if (NeuronProcessor::isAutoOrManuallyTriggered(data, object, constructor.autoTriggerInterval, isPreview)) {

        constructor.offspring = findOrCreateNewCreature(data, object);

        if (ConstructorHelper::isFinished(object, *constructor.offspring->genome)) {
            return;
        }

        auto constructionData = createConstructionData(object);
        if (tryConstructCell(data, statistics, object, constructionData)) {
            object->typeData.cell.signal.channels[Channels::ConstructorSuccess] = 1;  // Successful

            alienAtomicAdd32(&constructionData.creature->numCells, static_cast<uint32_t>(1));
            if (constructionData.isLastNodeOfLastConcatenation) {
                if (constructionData.isSeparation) {
                    ++constructor.currentOffspring;
                    if (constructor.provideEnergy == ProvideEnergy_Free) {
                        constructor.provideEnergy = ProvideEnergy_ReduceCellEnergy;
                    }
                    constructor.offspring = nullptr;

                    // HACK for preview mode: Do not construct more than one offspring + move seed away
                    if (isPreview) {
                        object->pos.y += toFloat(PREVIEW_HEIGHT / 3);
                    }
                }
            }
        } else {
            object->typeData.cell.signal.channels[Channels::ConstructorSuccess] = 0;  // Failed
        }
    }
}

__inline__ __device__ Creature* ConstructorProcessor::findOrCreateNewCreature(SimulationData& data, Object* object)
{
    auto& constructor = object->typeData.cell.constructor;

    if (constructor.offspring != nullptr) {
        return constructor.offspring;
    }

    // No separation for non-root genes => same creature
    auto& genome = object->typeData.cell.creature->genome;
    if (constructor.geneIndex < genome->numGenes) {
        if (!constructor.separation && constructor.geneIndex != 0) {
            return object->typeData.cell.creature;
        }
    }

    // Current branch under construction => use creature reference from there
    auto lastConstructionCell = ConstructorHelper::getLastConstructedCell(object);
    if (lastConstructionCell) {
        return lastConstructionCell->typeData.cell.creature;
    }

    // Nothing found => clone creature
    EntityFactory factory;
    factory.init(&data);
    auto result = factory.cloneCreature(object->typeData.cell.creature);
    result->numCells = 0;

    return result;
}

__inline__ __device__ ConstructorProcessor::ConstructionData ConstructorProcessor::createConstructionData(Object* object)
{
    auto& constructor = object->typeData.cell.constructor;
    auto& genome = constructor.offspring->genome;

    ConstructionData result;
    ConstructorHelper::getConstructorIndices(result.currentNodeIndex, result.currentConcatenation, result.currentBranch, object, *genome);
    result.creature = constructor.offspring;
    result.gene = ConstructorHelper::getCurrentGene(constructor, *genome);
    result.node = &result.gene->nodes[result.currentNodeIndex];
    result.isSeparation = constructor.separation;
    result.isFirstNode = result.currentNodeIndex == 0;
    auto isFirstConcatenation = result.currentConcatenation == 0;
    result.isFirstNodeOfFirstConcatenation = result.isFirstNode && isFirstConcatenation;
    result.isLastNode = result.currentNodeIndex == result.gene->numNodes - 1;
    result.isLastNodeOfLastConcatenation = result.isLastNode && result.currentConcatenation == constructor.numConcatenations - 1;
    result.hasInfiniteConcatenations = ConstructorHelper::hasInfiniteConcatenations(constructor);
    result.lastConstructionObject = ConstructorHelper::getLastConstructedCell(object);
    result.neededUsableEnergy = cudaSimulationParameters.normalCellEnergy.value[object->color];
    result.neededReservedEnergy = result.node->constructorAvailable ? result.node->constructor.reservedEnergy : 0.0f;
    auto cellTypeNode = result.gene->homogeneousCellType ? &result.gene->nodes[0] : result.node;
    result.neededDepotEnergy = cellTypeNode->cellType == CellType_Depot ? cellTypeNode->cellTypeData.depot.initialStoredUsableEnergy : 0.0f;

    ShapeGenerator shapeGenerator;
    auto shape = result.gene->shape;
    for (int i = 0; i <= result.currentNodeIndex; ++i) {
        auto generationResult = shapeGenerator.generateNextConstructionData(shape);
        if (i == result.currentNodeIndex) {
            result.shapeResult = generationResult;
        }
    }
    if (result.isFirstNode || result.isLastNode) {
        result.shapeResult.angle = result.node->referenceAngle;
    }

    if (result.gene->numNodes == 1) {
        result.shapeResult.numAdditionalConnections = 0;
    }

    if (result.isFirstNode) {
        if (result.isFirstNodeOfFirstConcatenation && result.currentBranch == 0) {
            result.shapeResult.angle = constructor.constructionAngle;
        } else if (isFirstConcatenation) {
            result.shapeResult.angle = 0;
        } else {
            result.shapeResult.angle = result.node->referenceAngle;
        }
    } else if (result.isLastNode) {
        result.shapeResult.angle = result.node->referenceAngle;
    }

    return result;
}

__inline__ __device__ Object*
ConstructorProcessor::tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Object* hostObject, ConstructionData const& constructionData)
{
    if (!hostObject->tryLock()) {
        return nullptr;
    }
    if (constructionData.isFirstNodeOfFirstConcatenation) {
        auto newObject = startConstructionOnNewBranch(data, statistics, hostObject, constructionData);

        hostObject->releaseLock();
        return newObject;
    } else {
        if (!constructionData.lastConstructionObject->tryLock()) {
            hostObject->releaseLock();
            return nullptr;
        }
        auto newObject = continueConstructionOnBranch(data, statistics, hostObject, constructionData);

        constructionData.lastConstructionObject->releaseLock();
        hostObject->releaseLock();
        return newObject;
    }
}

__inline__ __device__ Object* ConstructorProcessor::startConstructionOnNewBranch(
    SimulationData& data,
    SimulationStatistics& statistics,
    Object* hostObject,
    ConstructionData const& constructionData)
{
    if (hostObject->numConnections == MAX_OBJECT_CONNECTIONS) {
        return nullptr;
    }
    auto anglesForNewConnection = ObjectConnectionProcessor::calcLargestGapReferenceAndActualAngle(data, hostObject, constructionData.shapeResult.angle);

    auto newObjectDirection = Math::unitVectorOfAngle(anglesForNewConnection.actualAngle);
    float2 newObjectPos = hostObject->pos + newObjectDirection / 2;

    if (ObjectConnectionProcessor::existCrossingConnections(
            data, hostObject->pos, newObjectPos, cudaSimulationParameters.constructorConnectingCellDistance.value[hostObject->color], hostObject->detached())) {
        return nullptr;
    }

    if (!checkAndReduceHostEnergy(data, hostObject, constructionData)) {
        return nullptr;
    }

    // For bending muscle cells: Reset front angle and restore initial angle
    for (int i = 0; i < hostObject->numConnections; ++i) {
        auto const& connectedObject = hostObject->connections[i].object;
        if (connectedObject->type != ObjectType_Cell) {
            continue;
        }
        if (connectedObject->typeData.cell.cellType == CellType_Muscle && connectedObject->typeData.cell.cellTypeData.muscle.isBendingMuscle()) {
            connectedObject->typeData.cell.frontAngle = VALUE_NOT_SET_FLOAT;
            MuscleProcessor::restoreInitialAngleFromPrevious(connectedObject, hostObject);

            // Update newObject position and direction for corrected angle
            anglesForNewConnection = ObjectConnectionProcessor::calcLargestGapReferenceAndActualAngle(data, hostObject, constructionData.shapeResult.angle);
            newObjectDirection = Math::unitVectorOfAngle(anglesForNewConnection.actualAngle);
            newObjectPos = hostObject->pos + newObjectDirection / 2;
        }
    }

    uint64_t cellPointerIndex;
    Object* newObject = constructCellIntern(data, statistics, cellPointerIndex, hostObject, newObjectPos, constructionData);

    if (!newObject->tryLock()) {
        return nullptr;
    }

    if (!constructionData.isLastNodeOfLastConcatenation || !constructionData.isSeparation) {
        auto distance = constructionData.gene->connectionDistance;
        if (!ObjectConnectionProcessor::tryAddConnectionWithRelAngle(data, hostObject, newObject, distance, anglesForNewConnection.referenceAngle)) {
            ObjectConnectionProcessor::scheduleDeleteObject(data, cellPointerIndex);
        }
    }

    setHeadCellOnFirstNode(newObject, hostObject, constructionData);
    activateNewObjectOnLastNode(newObject, hostObject, constructionData);

    newObject->releaseLock();
    return newObject;
}

__inline__ __device__ Object* ConstructorProcessor::continueConstructionOnBranch(
    SimulationData& data,
    SimulationStatistics& statistics,
    Object* hostObject,
    ConstructionData const& constructionData)
{
    auto const& lastObject = constructionData.lastConstructionObject;
    auto posDelta = data.objectMap.getCorrectedDirection(lastObject->pos - hostObject->pos) / 2;

    auto desiredDistance = constructionData.gene->connectionDistance;
    //if (Math::length(posDelta) <= cudaSimulationParameters.minObjectDistance.value
    //    || desiredDistance < cudaSimulationParameters.minObjectDistance.value) {
    //    return nullptr;
    //}

    auto newObjectPos = hostObject->pos + posDelta;

    Object* objectsToConnect[3] = {};
    int numObjectsToConnect;
    getObjectsToConnect(objectsToConnect, numObjectsToConnect, data, hostObject, newObjectPos, constructionData);

    if (numObjectsToConnect < constructionData.shapeResult.numAdditionalConnections) {
        return nullptr;
    }

    if (!checkAndReduceHostEnergy(data, hostObject, constructionData)) {
        return nullptr;
    }

    // For bending muscle cells: Reset front angle and restore initial angle
    if (lastObject->typeData.cell.cellType == CellType_Muscle && lastObject->typeData.cell.cellTypeData.muscle.isBendingMuscle()) {
        lastObject->typeData.cell.frontAngle = VALUE_NOT_SET_FLOAT;
        MuscleProcessor::restoreInitialAngleFromPrevious(lastObject, hostObject);
    }
    if (hostObject->typeData.cell.cellType == CellType_Muscle && hostObject->typeData.cell.cellTypeData.muscle.isBendingMuscle()) {
        hostObject->typeData.cell.frontAngle = VALUE_NOT_SET_FLOAT;
        // If lastObject is also pivot object of hostObject => also restore initial angle on lastObject
        if (hostObject->connections[0].object == lastObject) {
            MuscleProcessor::restoreInitialAngleFromPrevious(hostObject, lastObject);
        }
    }

    uint64_t cellPointerIndex;
    Object* newObject = constructCellIntern(data, statistics, cellPointerIndex, hostObject, newObjectPos, constructionData);

    if (!newObject->tryLock()) {
        return nullptr;
    }
    if (constructionData.lastConstructionObject->typeData.cell.cellState == CellState_Dying) {
        newObject->typeData.cell.cellState = CellState_Dying;
    }

    auto lastToHostConnectionIndex = constructionData.lastConstructionObject->getConnectionIndex(
        hostObject);  // lastToHostConnectionIndex is usually 0 if construction has proceeded as planned so far
    auto origAngleFromPreviousOnLastConstructedCell = constructionData.lastConstructionObject->connections[lastToHostConnectionIndex].angleFromPrevious;

    // Move connection between lastConstructionCell and hostObject to a connection between lastConstructionCell and newObject
    auto separation = constructionData.isSeparation && constructionData.isLastNodeOfLastConcatenation;
    if (!separation) {
        newObject->numConnections = 2;

        // Connection between lastObject and newObject
        {
            auto& connection = lastObject->connections[lastToHostConnectionIndex];
            connection.object = newObject;
            connection.distance = desiredDistance;
        }
        {
            auto& connection = newObject->connections[1];
            connection.object = lastObject;
            connection.distance = desiredDistance;
            connection.angleFromPrevious = 180.0f - constructionData.shapeResult.angle;
        }

        // Connection between newObject and hostObject
        {
            auto& connection = newObject->connections[0];
            connection.object = hostObject;
            connection.distance = min(1.0f, desiredDistance);
            connection.angleFromPrevious = 180.0f + constructionData.shapeResult.angle;
        }
        {
            auto index = hostObject->getConnectionIndex(lastObject);
            auto& connection = hostObject->connections[index];
            connection.object = newObject;
            connection.distance = min(1.0f, desiredDistance);
        }
    } else {
        newObject->numConnections = 1;

        // Connection between lastObject and newObject
        {
            auto& connection = lastObject->connections[lastToHostConnectionIndex];
            connection.object = newObject;
            connection.distance = desiredDistance;
            connection.angleFromPrevious = origAngleFromPreviousOnLastConstructedCell;
        }
        {
            auto& connection = newObject->connections[0];
            connection.object = lastObject;
            connection.distance = desiredDistance;
            connection.angleFromPrevious = 360.0f;
            ObjectConnectionProcessor::deleteConnectionOneWay(hostObject, lastObject);
        }
    }

    // Connect to surrounding cells if possible
    int numConnectedObjects = 0;
    for (int i = 0; i < numObjectsToConnect; ++i) {
        Object* otherObject = objectsToConnect[i];

        if (otherObject->tryLock()) {
            if (newObject->numConnections < MAX_OBJECT_CONNECTIONS && otherObject->numConnections < MAX_OBJECT_CONNECTIONS) {
                auto requiredAngle1 = constructionData.shapeResult.requiredNodeAngle1[i] - constructionData.shapeResult.angle;
                // requiredAngle is given from connection to hostCell
                // in the separating case, this connection is lost
                if (separation) {
                    requiredAngle1 += 180.0f + constructionData.shapeResult.angle;
                }
                auto requiredAngle2 = constructionData.shapeResult.requiredNodeAngle2[i];
                if (ObjectConnectionProcessor::tryAddConnectionWithAbsAngle(data, newObject, otherObject, desiredDistance, requiredAngle1, requiredAngle2)) {
                    ++numConnectedObjects;
                }
            }
            otherObject->releaseLock();
        }
        if (numConnectedObjects == constructionData.shapeResult.numAdditionalConnections) {
            break;
        }
    }

    setHeadCellOnFirstNode(newObject, hostObject, constructionData);
    activateNewObjectOnLastNode(newObject, hostObject, constructionData);

    newObject->releaseLock();
    return newObject;
}

__inline__ __device__ void ConstructorProcessor::getObjectsToConnect(
    Object* result[],
    int& numResultCells,
    SimulationData& data,
    Object* hostObject,
    float2 const& newObjectPos,
    ConstructionData const& constructionData)
{
    numResultCells = 0;
    if (constructionData.shapeResult.numAdditionalConnections == 0) {
        return;
    }

    for (int i = 0; i < constructionData.shapeResult.numAdditionalConnections; ++i) {
        result[i] = nullptr;
    }

    data.objectMap.executeForEach(newObjectPos, SimulationParameters::attackerCreatureSensorRange, hostObject->detached(), [&](auto const& otherObject) {
        if (numResultCells == constructionData.shapeResult.numAdditionalConnections) {
            return;
        }
        if (otherObject->type != ObjectType_Cell) {
            return;
        }
        if (otherObject == hostObject || (otherObject->typeData.cell.cellState != CellState_Constructing && otherObject->typeData.cell.activationTime == 0)
            || otherObject->typeData.cell.creature != constructionData.creature
            || otherObject->typeData.cell.parentNodeIndex != hostObject->typeData.cell.nodeIndex) {
            return;
        }
        for (int i = 0; i < constructionData.shapeResult.numAdditionalConnections; ++i) {
            if (result[i] == nullptr && otherObject->typeData.cell.nodeIndex == constructionData.shapeResult.requiredNodeId[i]
                && otherObject->typeData.cell.concatenationIndex == constructionData.currentConcatenation
                && otherObject->typeData.cell.branchIndex == constructionData.currentBranch) {
                result[i] = otherObject;
                ++numResultCells;
                return;
            }
        }
    });
}

__inline__ __device__ Object* ConstructorProcessor::constructCellIntern(
    SimulationData& data,
    SimulationStatistics& statistics,
    uint64_t& objectIndex,
    Object* hostObject,
    float2 posOfNewObject,
    ConstructionData const& constructionData)
{
    auto& constructor = hostObject->typeData.cell.constructor;

    data.objectMap.correctPosition(posOfNewObject);

    EntityFactory factory;
    factory.init(&data);
    Object* result = factory.createCellFromNode(
        objectIndex,
        constructionData.creature,
        constructor.geneIndex,
        constructionData.currentNodeIndex,
        constructionData.gene->homogeneousCellType,
        hostObject->typeData.cell.nodeIndex,
        constructionData.currentConcatenation,
        constructionData.currentBranch,
        posOfNewObject,
        hostObject->vel,
        constructionData.neededUsableEnergy);
    result->typeData.cell.headUpdateId = constructionData.creature->headUpdateId;

    constructor.lastConstructedCellId = result->id;

    // Inherit free energy provision from parent in case that offspring constructs a non-separating gene
    if (constructor.provideEnergy == ProvideEnergy_Free && result->typeData.cell.constructorAvailable) {
        auto const& offspringConstructor = result->typeData.cell.constructor;
        auto const& offspringGenome = constructionData.creature->genome;
        if (offspringConstructor.geneIndex < offspringGenome->numGenes) {
            if (!offspringConstructor.separation) {
                result->typeData.cell.constructor.provideEnergy = ProvideEnergy_Free;
            }
        }
    }

    statistics.incNumCreatedCells(hostObject->color);

    return result;
}

__inline__ __device__ bool ConstructorProcessor::checkAndReduceHostEnergy(SimulationData& data, Object* hostObject, ConstructionData const& constructionData)
{
    auto& hostCell = hostObject->typeData.cell;
    auto& constructor = hostCell.constructor;
    if (constructor.provideEnergy == ProvideEnergy_Free) {
        return true;
    }

    auto requiredEnergy = constructionData.neededUsableEnergy + constructionData.neededReservedEnergy + constructionData.neededDepotEnergy;
    auto normalCellEnergy = cudaSimulationParameters.normalCellEnergy.value[hostObject->color];
    auto hostReservedEnergy = hostCell.constructor.reservedEnergy;

    // Check if enough energy available
    auto availableEnergyForConstruction = max(0.0f, hostCell.usableEnergy + hostReservedEnergy - normalCellEnergy);
    if (availableEnergyForConstruction < requiredEnergy) {

        // ... if not = > requesting external energy if possible
        if (isExternalEnergyInflowAllowed(hostObject)) {
            auto thresholdEnergy = requiredEnergy * cudaSimulationParameters.externalEnergyInflowThresholdFactor.value[hostObject->color];
            if (availableEnergyForConstruction >= thresholdEnergy) {
                constructor.energyNeeded = true;
            }
        }
        return false;
    }

    // Reduce reserved energy
    auto energyNeededFromReserved = min(hostCell.constructor.reservedEnergy, requiredEnergy);
    hostCell.constructor.reservedEnergy -= energyNeededFromReserved;
    requiredEnergy -= energyNeededFromReserved;

    // Reduce usable energy
    hostCell.usableEnergy -= requiredEnergy;
    return true;
}

__inline__ __device__ bool ConstructorProcessor::isExternalEnergyInflowAllowed(Object const* hostObject)
{
    if (!cudaSimulationParameters.externalEnergyControlToggle.value || cudaSimulationParameters.externalEnergyInflowFactor.value[hostObject->color] <= 0) {
        return false;
    }
    if (cudaSimulationParameters.externalEnergyInflowOnlyForFirstOffspring.value && hostObject->typeData.cell.constructor.currentOffspring > 0) {
        return false;
    }
    return true;
}

__inline__ __device__ void ConstructorProcessor::activateNewObjectOnLastNode(Object* newObject, Object* hostObject, ConstructionData const& constructionData)
{
    if (constructionData.isLastNode) {
        newObject->typeData.cell.cellState = CellState_Activating;
    }
}

__inline__ __device__ void ConstructorProcessor::setHeadCellOnFirstNode(Object* newObject, Object* hostObject, ConstructionData const& constructionData)
{
    auto const& constructor = hostObject->typeData.cell.constructor;

    // Head cell should be first (=> connections[0] points to nodeIndex=1 in each concatenation)
    if (constructionData.isFirstNode && (constructionData.isSeparation || constructor.geneIndex == 0)) {
        newObject->typeData.cell.headCell = true;
    }
}
