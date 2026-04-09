#pragma once

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/ShapeGenerator.h>

#include "CellProcessor.cuh"
#include "ObjectConnectionProcessor.cuh"
#include "ConstructorHelper.cuh"
#include "Genome.cuh"
#include "MuscleProcessor.cuh"
#include "SimulationStatistics.cuh"

class ConstructorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics, bool isPreview);

private:
    struct ConstructionData
    {
        // Creature and genome data
        Creature* creature;
        Gene* gene;
        Node* node;
        bool isSeparation;

        // Construction position
        bool isFirstNode;
        bool isFirstNodeOfFirstConcatenation;
        bool isLastNode;
        bool isLastNodeOfLastConcatenation;
        bool hasInfiniteConcatenations;

        // Construction data
        Object* lastConstructionObject;
        float angle;
        float neededUsableEnergy;
        float neededReservedEnergy;
        float depotEnergy;
        int numAdditionalConnections;  // -1 = none
        int requiredNodeId1;           // -1 = none
        int requiredNodeId2;           // -1 = none
        int requiredNodeId3;           // -1 = none
    };
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Object* object, bool isPreview);
    __inline__ __device__ static Creature* findOrCreateNewCreature(SimulationData& data, Object* object);
    __inline__ __device__ static ConstructionData createConstructionData(Object* object);

    __inline__ __device__ static Object* tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Object* hostObject, ConstructionData const& constructionData);
    __inline__ __device__ static void tryScheduleMutations(SimulationData& data, Object* hostObject);

    __inline__ __device__ static Object* getLastConstructedCellOnBranch(Object* hostObject);
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

    __inline__ __device__ static bool checkForValidConstruction(Object* hostObject);
    __inline__ __device__ static bool checkAndReduceHostEnergy(SimulationData& data, Object* hostObject, ConstructionData const& constructionData);
    __inline__ __device__ static void activateNewObjectOnLastNode(Object* newObject, Object* hostObject, ConstructionData const& constructionData);

    //
    // Assumption: object1 is connected with object2 and object2 is connected with object3
    //
    // If object3 is connected to object1 directly or via further cells (not object2):
    //  Calculates the inner angle sum of the n-polygon spanned by
    //      (1) object1
    //      (2) object2
    //      (3) object3
    //      + possibly further cells between (3) and (1)
    //  and
    //      set the angle on cell (3) between the connected cells (2) and (4)
    //      (where (4) can be (1) if no further cells are in the polygon, i.e. n=3)
    //      such that the inner angle sum of the polygon is (n - 2) * 180 deg
    // Else:
    //  No angle correction
    //
    __inline__ __device__ static void correctAnglesByInnerAngleSum(Object* object1, Object* object2, Object* object3);

    __inline__ __device__ static float getRequiredEnergyForNodes(Gene* gene);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ConstructorProcessor::process(SimulationData& data, SimulationStatistics& statistics, bool isPreview)
{
    auto const partition = calcSystemThreadPartition(data.entities.objects.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        auto object = data.entities.objects.at(i);
        if (object->type != ObjectType_Cell) {
            continue;
        }   
        if (!object->typeData.cell.constructorAvailable) {
            continue;
        }
        if (!CellProcessor::isCellReady(data, object)) {
            continue;
        }
        processCell(data, statistics, object, isPreview);
    }
}

__inline__ __device__ void ConstructorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Object* object, bool isPreview)
{
    auto& constructor = object->typeData.cell.constructor;
    if (NeuronProcessor::isAutoOrManuallyTriggered(data, object, constructor.autoTriggerInterval, isPreview)) {
        tryScheduleMutations(data, object); // TODO only when energy for new cell is available

        constructor.offspring = findOrCreateNewCreature(data, object);

        if (ConstructorHelper::isFinished(constructor, *constructor.offspring->genome)) {
            return;
        }

        if (!checkForValidConstruction(object)) {
            constructor.currentNodeIndex = 0;
            constructor.currentConcatenation = 0;
        }

        auto constructionData = createConstructionData(object);
        if (tryConstructCell(data, statistics, object, constructionData)) {
            object->typeData.cell.signal.channels[Channels::ConstructorSuccess] = 1;  // Successful

            ++constructionData.creature->numCells;
            if (!constructionData.isLastNode) {
                ++constructor.currentNodeIndex;
            } else {
                constructor.currentNodeIndex = 0;
                ++constructor.currentConcatenation;
            }
            if (constructionData.isLastNodeOfLastConcatenation) {
                constructor.currentConcatenation = 0;
                if (!constructionData.isSeparation) {
                    ++constructor.currentBranch;
                } else {
                    ++constructor.currentOffspring;
                    if (constructor.provideEnergy == ProvideEnergy_FreeGeneration) {
                        constructor.provideEnergy = ProvideEnergy_CellOnly;
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

    // No separation => same creature
    auto& genome = object->typeData.cell.creature->genome;
    if (constructor.geneIndex < genome->numGenes) {
        auto const& gene = ConstructorHelper::getCurrentGene(constructor, *genome);
        if (!gene->separation) {
            return object->typeData.cell.creature;
        }
    }

    // Current branch under construction => use creature reference from there
    if (!(ConstructorHelper::isFirstNode(constructor) && ConstructorHelper::isFirstConcatenation(constructor))) {
        auto lastConstructionCell = getLastConstructedCellOnBranch(object);
        if (lastConstructionCell) {
            return lastConstructionCell->typeData.cell.creature;
        }
    }

    // Other branches already constructed => same creature
    if (constructor.currentBranch > 0) {
        return object->typeData.cell.creature;
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

    auto isFirstConcatenation = ConstructorHelper::isFirstConcatenation(constructor);

    ConstructionData result;
    result.creature = constructor.offspring;
    result.gene = ConstructorHelper::getCurrentGene(constructor, *genome);
    result.node = ConstructorHelper::getCurrentNode(constructor, *genome);
    result.isSeparation = result.gene->separation;
    result.isFirstNode = ConstructorHelper::isFirstNode(constructor);
    result.isFirstNodeOfFirstConcatenation = result.isFirstNode && isFirstConcatenation;
    result.isLastNode = ConstructorHelper::isLastNode(constructor, *genome);
    result.isLastNodeOfLastConcatenation = result.isLastNode && ConstructorHelper::isLastConcatenation(constructor, *genome);

    result.hasInfiniteConcatenations = ConstructorHelper::hasInfiniteConcatenations(result.gene);
    result.lastConstructionObject = getLastConstructedCellOnBranch(object);
    result.angle = result.node->referenceAngle;
    result.neededUsableEnergy = cudaSimulationParameters.normalCellEnergy.value[object->color];
    result.neededReservedEnergy = 0;
    if (result.node->constructorAvailable) {
        auto const& constructorNode = result.node->constructor;
        if (constructor.provideEnergy == ProvideEnergy_CellAndGene && constructorNode.geneIndex < result.creature->genome->numGenes) {
            auto& referencedGene = result.creature->genome->genes[constructorNode.geneIndex];
            if (!referencedGene.separation) {
                auto requiredEnergyForNodes = getRequiredEnergyForNodes(&referencedGene);
                if (!ConstructorHelper::hasInfiniteConcatenations(&referencedGene)) {
                    result.neededReservedEnergy += requiredEnergyForNodes * referencedGene.numBranches * referencedGene.numConcatenations;
                } else {
                    result.neededReservedEnergy += requiredEnergyForNodes;
                }
            }
        }
    }
    result.depotEnergy = result.node->cellType == CellType_Depot ? result.node->cellTypeData.depot.initialStoredUsableEnergy : 0.0f;
    result.numAdditionalConnections = result.node->numAdditionalConnections;

    ShapeGenerator shapeGenerator;
    auto shape = result.gene->shape;
    if (shape != ConstructorShape_Custom && !ConstructorHelper::isFirstNode(constructor) /*&& !result.isLastNode*/) {
        result.gene->angleAlignment = shapeGenerator.getConstructorAngleAlignment(shape);
        for (int i = 0; i <= constructor.currentNodeIndex; ++i) {
            auto generationResult = shapeGenerator.generateNextConstructionData(shape);
            if (i == constructor.currentNodeIndex) {
                if (!result.isLastNode) {
                    result.angle = generationResult.angle;
                }
                result.numAdditionalConnections = generationResult.numAdditionalConnections;
                result.requiredNodeId1 = generationResult.requiredNodeId1;
                result.requiredNodeId2 = generationResult.requiredNodeId2;
                result.requiredNodeId3 = generationResult.requiredNodeId3;
            }
        }
    } else {
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
        result.requiredNodeId3 = -1;
    }

    if (result.gene->numNodes == 1) {
        result.numAdditionalConnections = 0;
    }

    if (result.isFirstNode) {
        if (result.isFirstNodeOfFirstConcatenation && ConstructorHelper::isFirstBranch(constructor)) {
            result.angle = constructor.constructionAngle;
        } else if (isFirstConcatenation) {
            result.angle = 0;
        } else {
            result.angle = result.node->referenceAngle;
        }
    } else if (result.isLastNode) {
        result.angle = result.node->referenceAngle;
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

__inline__ __device__ void ConstructorProcessor::tryScheduleMutations(SimulationData& data, Object* hostObject)
{
    atomicCAS(&hostObject->typeData.cell.creature->mutationState, MutationState_NotMutated, MutationState_MutationInProgress);
    hostObject->typeData.cell.constructor.offspring = nullptr;
}

__inline__ __device__ Object* ConstructorProcessor::getLastConstructedCellOnBranch(Object* hostObject)
{
    auto const& constructor = hostObject->typeData.cell.constructor;
    if (constructor.lastConstructedCellId != VALUE_NOT_SET_UINT64) {
        for (int i = 0; i < hostObject->numConnections; ++i) {
            auto const& connectedObject = hostObject->connections[i].object;
            if (connectedObject->id == constructor.lastConstructedCellId) {
                return connectedObject;
            }
        }
    }
    return nullptr;
}

__inline__ __device__ Object* ConstructorProcessor::startConstructionOnNewBranch(
    SimulationData& data,
    SimulationStatistics& statistics,
    Object* hostObject,
    ConstructionData const& constructionData)
{
    auto& constructor = hostObject->typeData.cell.constructor;

    if (hostObject->numConnections == MAX_OBJECT_CONNECTIONS) {
        return nullptr;
    }
    auto anglesForNewConnection = ObjectConnectionProcessor::calcLargestGapReferenceAndActualAngle(data, hostObject, constructionData.angle);

    auto newObjectDirection = Math::unitVectorOfAngle(anglesForNewConnection.actualAngle);
    float2 newObjectPos = hostObject->pos + newObjectDirection / 2;

    if (ObjectConnectionProcessor::existCrossingConnections(
            data, hostObject->pos, newObjectPos, cudaSimulationParameters.constructorConnectingCellDistance.value[hostObject->color], hostObject->detached)) {
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
            MuscleProcessor::restoreInitialAngleFromPrevious(connectedObject, hostObject, i);

            // Update newObject position and direction for corrected angle
            anglesForNewConnection = ObjectConnectionProcessor::calcLargestGapReferenceAndActualAngle(data, hostObject, constructionData.angle);
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
        if (!ObjectConnectionProcessor::tryAddConnection(data, hostObject, newObject, anglesForNewConnection.referenceAngle, 0, distance)) {
            ObjectConnectionProcessor::scheduleDeleteObject(data, cellPointerIndex);
        }
    }

    // Head cell should be first (=> connection[0] points to nodeIndex=1 in each concatenation)
    if ((constructionData.isSeparation || constructor.geneIndex == 0) && constructionData.isFirstNode) {
        newObject->typeData.cell.headCell = true;
    }
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
    auto angleFromPreviousForNewObject = 180.0f - constructionData.angle;

    auto desiredDistance = constructionData.gene->connectionDistance;
    //if (Math::length(posDelta) <= cudaSimulationParameters.minObjectDistance.value
    //    || desiredDistance < cudaSimulationParameters.minObjectDistance.value) {
    //    return nullptr;
    //}

    auto newObjectPos = hostObject->pos + posDelta;
    if (ObjectConnectionProcessor::existCrossingConnections(
            data,
            hostObject->pos,
            constructionData.lastConstructionObject->pos,
            cudaSimulationParameters.constructorConnectingCellDistance.value[hostObject->color],
            hostObject->detached)) {
        return nullptr;
    }

    Object* objectsToConnect[3];
    int numObjectsToConnect;
    getObjectsToConnect(objectsToConnect, numObjectsToConnect, data, hostObject, newObjectPos, constructionData);

    if (numObjectsToConnect < constructionData.numAdditionalConnections) {
        return nullptr;
    }

    if (!checkAndReduceHostEnergy(data, hostObject, constructionData)) {
        return nullptr;
    }

    // For bending muscle cells: Reset front angle and restore initial angle
    if (lastObject->typeData.cell.cellType == CellType_Muscle && lastObject->typeData.cell.cellTypeData.muscle.isBendingMuscle()) {
        lastObject->typeData.cell.frontAngle = VALUE_NOT_SET_FLOAT;
        auto connectionIndex = hostObject->getConnectionIndex(lastObject);
        MuscleProcessor::restoreInitialAngleFromPrevious(lastObject, hostObject, connectionIndex);
    }

    uint64_t cellPointerIndex;
    Object* newObject = constructCellIntern(data, statistics, cellPointerIndex, hostObject, newObjectPos, constructionData);

    if (!newObject->tryLock()) {
        return nullptr;
    }
    if (constructionData.lastConstructionObject->typeData.cell.cellState == CellState_Dying) {
        newObject->typeData.cell.cellState = CellState_Dying;
    }

    float origAngleFromPreviousOnLastConstructedCell;
    for (int i = 0; i < constructionData.lastConstructionObject->numConnections; ++i) {
        if (constructionData.lastConstructionObject->connections[i].object == hostObject) {
            origAngleFromPreviousOnLastConstructedCell = constructionData.lastConstructionObject->connections[i].angleFromPrevious;
        }
    }

    // Move connection between lastConstructionCell and hostObject to a connection between lastConstructionCell and newObject
    auto separation = constructionData.isSeparation && constructionData.isLastNodeOfLastConcatenation;
    bool adaptReferenceAngles = false;
    if (!separation) {
        adaptReferenceAngles = true;
        newObject->numConnections = 2;

        // Connection between lastObject and newObject
        {
            auto& connection = lastObject->connections[0];
            connection.object = newObject;
            connection.distance = desiredDistance;
        }
        {
            auto& connection = newObject->connections[1];
            connection.object = lastObject;
            connection.distance = desiredDistance;
            connection.angleFromPrevious = 180.0f - constructionData.angle;
        }

        // Connection between newObject and hostObject
        {
            auto& connection = newObject->connections[0];
            connection.object = hostObject;
            connection.distance = desiredDistance;
            connection.angleFromPrevious = 180.0f + constructionData.angle;
        }
        {
            auto index = hostObject->getConnectionIndex(lastObject);
            auto& connection = hostObject->connections[index];
            connection.object = newObject;
            connection.distance = desiredDistance;
        }
    } else {
        newObject->numConnections = 1;

        // Connection between lastObject and newObject
        {
            auto& connection = lastObject->connections[0];
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
                if (ObjectConnectionProcessor::tryAddConnection(
                        data, newObject, otherObject, 0, 0, desiredDistance, constructionData.gene->angleAlignment)) {
                    ++numConnectedObjects;
                }
            }
            otherObject->releaseLock();
        }
        if (numConnectedObjects == constructionData.numAdditionalConnections) {
            break;
        }
    }

        // Adapt angles on other connected cells
    if (adaptReferenceAngles) {

        auto n = newObject->numConnections;
        auto lastObjectIndex = newObject->getConnectionIndex(constructionData.lastConstructionObject);
        auto hostObjectIndex = newObject->getConnectionIndex(hostObject);

        for (int i = lastObjectIndex; (i + n) % n != (hostObjectIndex + 1) % n && (i + n) % n != hostObjectIndex; --i) {
            correctAnglesByInnerAngleSum(newObject->getConnectedObject(i), newObject, newObject->getConnectedObject(i - 1));
        }
        for (int i = lastObjectIndex + 1; i % n != hostObjectIndex; ++i) {
            correctAnglesByInnerAngleSum(newObject->getConnectedObject(i - 1), newObject, newObject->getConnectedObject(i));
        }

        // Adapt angles on new cell
        //float consumedAngle1 = 0;
        //if (n > 2) {
        //    for (int i = lastObjectIndex; (i + n) % n != (hostObjectIndex + 1) % n && (i + n) % n != hostObjectIndex; --i) {
        //        consumedAngle1 += newObject->connections[(i + n) % n].angleFromPrevious;
        //    }
        //}
        //float consumedAngle2 = 0;
        //if (n > 2) {
        //    for (int i = lastObjectIndex + 1; i % n != hostObjectIndex; ++i) {
        //        consumedAngle2 += newObject->connections[i % n].angleFromPrevious;
        //    }
        //}
        //if (angleFromPreviousForNewObject - consumedAngle1 >= 0 && 360.0f - angleFromPreviousForNewObject - consumedAngle2 >= 0) {
        //    newObject->connections[(hostObjectIndex + 1) % n].angleFromPrevious = angleFromPreviousForNewObject - consumedAngle1;
        //    newObject->connections[hostObjectIndex].angleFromPrevious = 360.0f - angleFromPreviousForNewObject - consumedAngle2;
        //}
    }

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
    if (constructionData.numAdditionalConnections == 0) {
        return;
    }

    data.objectMap.executeForEach(newObjectPos, SimulationParameters::attackerCreatureSensorRange, hostObject->detached, [&](auto const& otherObject) {
        if (otherObject->type != ObjectType_Cell) {
            return;
        }
        if (otherObject == hostObject || (otherObject->typeData.cell.cellState != CellState_Constructing && otherObject->typeData.cell.activationTime == 0)
            || otherObject->typeData.cell.creature != constructionData.creature
            || otherObject->typeData.cell.parentNodeIndex != hostObject->typeData.cell.nodeIndex) {
            return;
        }
        if (constructionData.numAdditionalConnections >= 1 && otherObject->typeData.cell.nodeIndex == constructionData.requiredNodeId1) {
            result[0] = otherObject;
            ++numResultCells;
            return;
        }
        if (constructionData.numAdditionalConnections >= 2 && otherObject->typeData.cell.nodeIndex == constructionData.requiredNodeId2) {
            result[1] = otherObject;
            ++numResultCells;
            return;
        }
        if (constructionData.numAdditionalConnections >= 3 && otherObject->typeData.cell.nodeIndex == constructionData.requiredNodeId3) {
            result[2] = otherObject;
            ++numResultCells;
            return;
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
        constructor.currentNodeIndex,
        hostObject->typeData.cell.nodeIndex,
        constructor.currentConcatenation,
        constructor.currentBranch,
        posOfNewObject,
        hostObject->vel,
        constructionData.neededUsableEnergy,
        constructionData.neededReservedEnergy);
    result->typeData.cell.headUpdateId = hostObject->typeData.cell.headUpdateId;

    constructor.lastConstructedCellId = result->id;

    // Inherit free energy provision from parent in case that offspring constructs a non-separating gene
    if (constructor.provideEnergy == ProvideEnergy_FreeGeneration && result->typeData.cell.constructorAvailable) {
        auto const& offspringConstructor = result->typeData.cell.constructor;
        auto const& offspringGenome = constructionData.creature->genome;
        if (offspringConstructor.geneIndex < offspringGenome->numGenes) {
            auto const& offspringGene = offspringGenome->genes[offspringConstructor.geneIndex];
            if (!offspringGene.separation) {
                result->typeData.cell.constructor.provideEnergy = ProvideEnergy_FreeGeneration;
            }
        }
    }

    statistics.incNumCreatedCells(hostObject->color);

    return result;
}

__inline__ __device__ bool ConstructorProcessor::checkForValidConstruction(Object* hostObject)
{
    auto& constructor = hostObject->typeData.cell.constructor;
    auto& genome = constructor.offspring->genome;

    auto lastConstructionCell = getLastConstructedCellOnBranch(hostObject);
    if (!(constructor.currentNodeIndex == 0 && constructor.currentConcatenation == 0 && constructor.currentBranch == 0)) {
        if (lastConstructionCell == nullptr) {
            return false;
        }
    }
    if (lastConstructionCell && lastConstructionCell->numConnections == 1) {
        int numConstructedCells = ConstructorHelper::getNumConstructedCellsOnBranch(constructor, *genome);
        return numConstructedCells <= 1;
    }
    return true;
}


__inline__ __device__ bool ConstructorProcessor::checkAndReduceHostEnergy(SimulationData& data, Object* hostObject, ConstructionData const& constructionData)
{
    if (hostObject->typeData.cell.constructor.provideEnergy == ProvideEnergy_FreeGeneration) {
        return true;
    }

    auto requiredEnergy = constructionData.neededUsableEnergy + constructionData.neededReservedEnergy + constructionData.depotEnergy;
    auto normalCellEnergy = cudaSimulationParameters.normalCellEnergy.value[hostObject->color];

    if (cudaSimulationParameters.externalEnergyControlToggle.value && hostObject->typeData.cell.usableEnergy < requiredEnergy + normalCellEnergy
        && cudaSimulationParameters.externalEnergyInflowFactor.value[hostObject->color] > 0) {
        auto externalEnergyPortion = [&] {
            if (cudaSimulationParameters.externalEnergyInflowOnlyForFirstOffspring.value) {
                return hostObject->typeData.cell.constructor.currentOffspring == 0
                    ? requiredEnergy * cudaSimulationParameters.externalEnergyInflowFactor.value[hostObject->color]
                    : 0.0f;
            } else {
                return requiredEnergy * cudaSimulationParameters.externalEnergyInflowFactor.value[hostObject->color];
            }
        }();

        auto origExternalEnergy = alienAtomicRead(data.externalEnergy);
        if (origExternalEnergy == Infinity<float>::value) {
            hostObject->typeData.cell.usableEnergy += externalEnergyPortion;
        } else {
            externalEnergyPortion = max(0.0f, min(origExternalEnergy, externalEnergyPortion));
            auto origExternalEnergy_tickLater = atomicAdd(data.externalEnergy, -externalEnergyPortion);
            if (origExternalEnergy_tickLater >= externalEnergyPortion) {
                hostObject->typeData.cell.usableEnergy += externalEnergyPortion;
            } else {
                atomicAdd(data.externalEnergy, externalEnergyPortion);
            }
        }
    }

    auto externalEnergyConditionalInflowFactor = [&] {
        if (!cudaSimulationParameters.externalEnergyControlToggle.value) {
            return 0.0f;
        }
        if (cudaSimulationParameters.externalEnergyInflowOnlyForFirstOffspring.value) {
            return hostObject->typeData.cell.constructor.currentOffspring == 0
                ? cudaSimulationParameters.externalEnergyConditionalInflowFactor.value[hostObject->color]
                : 0.0f;
        } else {
            return cudaSimulationParameters.externalEnergyConditionalInflowFactor.value[hostObject->color];
        }
    }();

    // Use reserved energy first
    auto energyNeededFromHost = min(requiredEnergy, hostObject->typeData.cell.reservedEnergy);

    // Then use external energy supply for the remaining energy
    energyNeededFromHost += (requiredEnergy - energyNeededFromHost) * (1.0f - externalEnergyConditionalInflowFactor);

    if (externalEnergyConditionalInflowFactor < 1.0f
        && hostObject->typeData.cell.usableEnergy + hostObject->typeData.cell.reservedEnergy < normalCellEnergy + energyNeededFromHost) {
        return false;
    }
    auto energyNeededFromExternalSource = requiredEnergy - energyNeededFromHost;
    auto orig = atomicAdd(data.externalEnergy, -energyNeededFromExternalSource);

    float finalEnergyNeededFromHost;
    if (orig < energyNeededFromExternalSource) {
        atomicAdd(data.externalEnergy, energyNeededFromExternalSource);
        if (hostObject->typeData.cell.usableEnergy + hostObject->typeData.cell.reservedEnergy < normalCellEnergy + requiredEnergy) {
            return false;
        }
        finalEnergyNeededFromHost = requiredEnergy;
    } else {
        finalEnergyNeededFromHost = energyNeededFromHost;
    }

    hostObject->typeData.cell.reservedEnergy -= finalEnergyNeededFromHost;
    if (hostObject->typeData.cell.reservedEnergy < 0) {
        hostObject->typeData.cell.usableEnergy += hostObject->typeData.cell.reservedEnergy;
        hostObject->typeData.cell.reservedEnergy = 0.0f;
    }
    return true;
}

__inline__ __device__ void ConstructorProcessor::activateNewObjectOnLastNode(Object* newObject, Object* hostObject, ConstructionData const& constructionData)
{
    if (constructionData.isLastNode) {
        newObject->typeData.cell.cellState = CellState_Activating;
    }
}

__inline__ __device__ void ConstructorProcessor::correctAnglesByInnerAngleSum(Object* object1, Object* object2, Object* object3)
{
    // Check if object3 connects back to object1 (directly or via further objects, not through object2)
    // to form a closed polygon

    // Determine traversal direction to find minimal polygon
    // Find indices of object1 and object3 in object2's connections (which are sorted clockwise)
    int object1IndexInObject2 = object2->getConnectionIndex(object1);
    int object3IndexInObject2 = object2->getConnectionIndex(object3);

    // Determine if we go clockwise or counter-clockwise from object3 to find object1
    // The polygon is: object1 -> object2 -> object3 -> ... -> object1
    // From object2's perspective, we go clockwise from object1 to object3
    // So from object3, we should continue in a consistent direction to find object1
    bool goClockwiseFromObject3;
    if (object3IndexInObject2 == (object1IndexInObject2 + 1) % object2->numConnections) {
        // object3 is clockwise from object1 in object2's connections
        goClockwiseFromObject3 = true;
    } else {
        // object3 is counter-clockwise from object1 (wrapped around)
        goClockwiseFromObject3 = false;
    }

    // Find the minimal path from object3 to object1 (not going through object2)
    Object* currentObject = object3;
    Object* previousObject = object2;
    int numIntermediateObjects = 0;
    bool foundPolygon = false;
    Object* object4 = nullptr;  // The next object after object3 in the polygon

    // Find object2's index in object3's connections to determine traversal direction
    int object2IndexInObject3 = object3->getConnectionIndex(object2);

    constexpr int maxPolygonSize = 50;
    float currentAngleSum = 0.0f;
    for (int step = 0; step < maxPolygonSize; ++step) {
        Object* nextObject = nullptr;

        if (step == 0) {
            int startIndex = goClockwiseFromObject3 ? object2IndexInObject3 + 1 : object2IndexInObject3 - 1;

            for (int i = 0; i < object3->numConnections; ++i) {
                int index = goClockwiseFromObject3 ? startIndex + i : startIndex - i;

                Object* candidate = object3->getConnectedObject(index);
                if (candidate == object1) {
                    nextObject = candidate;
                    foundPolygon = true;
                    break;
                } else if (candidate != object2) {
                    nextObject = candidate;
                    object4 = candidate;
                    break;
                }
            }
        } else {
            // Subsequent steps: find next object that's not the previous one
            int prevIndex = currentObject->getConnectionIndex(previousObject);

            // Continue in the same general direction
            for (int i = 1; i < currentObject->numConnections; ++i) {
                int index = goClockwiseFromObject3 ? prevIndex + i : prevIndex - i;

                Object* candidate = currentObject->getConnectedObject(index);
                if (candidate == object1) {
                    nextObject = candidate;
                    foundPolygon = true;
                    break;
                } else if (candidate != object2) {
                    nextObject = candidate;
                    break;
                }
            }
        }

        if (step > 0) {
            if (!goClockwiseFromObject3) {
                currentAngleSum += currentObject->getAngelSpan(nextObject, previousObject);
            } else {
                currentAngleSum += currentObject->getAngelSpan(previousObject, nextObject);
            }
        }

        if (foundPolygon || nextObject == nullptr) {
            break;
        }

        previousObject = currentObject;
        currentObject = nextObject;
        numIntermediateObjects++;
    }

    if (!foundPolygon) {
        // No closed polygon found, no angle correction based on polygon possible
        return;
    }

    // If object4 is still null, it means object3 connects directly to object1
    if (object4 == nullptr) {
        object4 = object1;
    }

    // Number of vertices in the polygon
    int numVertices = 3 + numIntermediateObjects;  // object1, object2, object3, + intermediate objects

    // Calculate expected inner angle sum for an n-sided polygon: (n - 2) * 180 degrees
    float expectedAngleSum = (numVertices - 2) * 180.0f;

    Object* lastObjectBeforeObject1 = currentObject;  // This is the last object we visited before reaching object1
    if (!goClockwiseFromObject3) {
        currentAngleSum += object1->getAngelSpan(object2, lastObjectBeforeObject1);
        currentAngleSum += object2->getAngelSpan(object3, object1);
        currentAngleSum += object3->getAngelSpan(object4, object2);
    } else {
        currentAngleSum += object1->getAngelSpan(lastObjectBeforeObject1, object2);
        currentAngleSum += object2->getAngelSpan(object1, object3);
        currentAngleSum += object3->getAngelSpan(object2, object4);
    }
    float angleCorrection = expectedAngleSum - currentAngleSum;

    int object2Index = object3->getConnectionIndex(object2);

    if (!goClockwiseFromObject3) {
        object3->increaseAngle(object2Index, angleCorrection);

        // If adapted angle is 0, try fallback
        if (abs(object3->getConnection(object2Index).angleFromPrevious) < NEAR_ZERO) {
            object3->increaseAngle(object2Index, -angleCorrection);  // Revert
            object2->increaseAngle(object1IndexInObject2, angleCorrection);

            // Other angle also 0 => revert
            if (abs(object2->getConnection(object1IndexInObject2).angleFromPrevious) < NEAR_ZERO) {
                object2->increaseAngle(object1IndexInObject2, -angleCorrection);
            }
        }
    } else {
        object3->increaseAngle(object2Index, -angleCorrection);

        // If adapted angle is 0, try fallback
        if (abs(object3->getConnection(object2Index + 1).angleFromPrevious) < NEAR_ZERO) {
            object3->increaseAngle(object2Index, angleCorrection);  // Revert
            object2->increaseAngle(object1IndexInObject2, -angleCorrection);

            // Other angle also 0 => revert 
            if (abs(object2->getConnection(object1IndexInObject2 + 1).angleFromPrevious) < NEAR_ZERO) {
                object2->increaseAngle(object1IndexInObject2, angleCorrection);
            }
        }
    }
}

__inline__ __device__ float ConstructorProcessor::getRequiredEnergyForNodes(Gene* gene)
{
    auto result = 0.0f;
    for (int i = 0, j = gene->numNodes; i < j; ++i) {
        auto const& node = &gene->nodes[i];
        result += cudaSimulationParameters.normalCellEnergy.value[node->color];
        if (node->cellType == CellType_Depot) {
            result += node->cellTypeData.depot.initialStoredUsableEnergy;
        }
    }
    return result;
}
