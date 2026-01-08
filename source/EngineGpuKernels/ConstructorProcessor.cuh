#pragma once

#include <EngineInterface/CellTypeConstants.h>

#include "CellConnectionProcessor.cuh"
#include "ConstructorHelper.cuh"
#include "CudaShapeGenerator.cuh"
#include "Genome.cuh"
#include "GenomeProcessor.cuh"
#include "MuscleProcessor.cuh"
#include "SignalProcessor.cuh"
#include <EngineImpl/SimulationCudaFacade.cuh>
#include "SimulationStatistics.cuh"

class ConstructorProcessor
{
public:
    __inline__ __device__ static void preprocess(SimulationData& data);
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
        bool isFirstNodeOfFirstConcatenation;
        bool isLastNode;
        bool isLastNodeOfLastConcatenation;
        bool hasInfiniteConcatenations;

        // Construction data
        Cell* lastConstructionCell;
        float angle;
        float cellEnergy;
        float depotEnergy;
        int numAdditionalConnections;  // -1 = none
        int requiredNodeId1;           // -1 = none
        int requiredNodeId2;           // -1 = none
    };
    //
    //    __inline__ __device__ static void completenessCheck(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    //
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell, bool isPreview);
    __inline__ __device__ static Creature* findOrCreateNewCreature(SimulationData& data, Cell* cell);
    __inline__ __device__ static ConstructionData createConstructionData(Cell* cell);

    __inline__ __device__ static Cell*
    tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static Cell* getLastConstructedCellOnBranch(Cell* hostCell);
    __inline__ __device__ static Cell*
    startConstructionOnNewBranch(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);
    __inline__ __device__ static Cell*
    continueConstructionOnBranch(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static void getCellsToConnect(
        Cell* result[],
        int& numResultCells,
        SimulationData& data,
        Cell* hostCell,
        float2 const& newCellPos,
        ConstructionData const& constructionData);

    __inline__ __device__ static Cell* constructCellIntern(
        SimulationData& data,
        SimulationStatistics& statistics,
        uint64_t& cellIndex,
        Cell* hostCell,
        float2 newCellPos,
        ConstructionData const& constructionData);

    __inline__ __device__ static bool checkForValidConstruction(Cell* hostCell);
    __inline__ __device__ static bool checkAndReduceHostEnergy(SimulationData& data, Cell* hostCell, ConstructionData const& constructionData);
    __inline__ __device__ static void activateNewCellOnLastNode(Cell* newCell, Cell* hostCell, ConstructionData const& constructionData);

    //
    // Assumption: cell1 is connected with cell2 and cell2 is connected with cell3
    // 
    // If cell3 is connected to cell1 directly or via further cells (not cell2):
    //  Calculates the inner angle sum of the n-polygon spanned by
    //      (1) cell1
    //      (2) cell2
    //      (3) cell3
    //      + possibly further cells between (3) and (1)
    //  and 
    //      set the angle on cell (3) between the connected cells (2) and (4)
    //      (where (4) can be (1) if no further cells are in the polygon, i.e. n=3)
    //      such that the inner angle sum of the polygon is (n - 2) * 180 deg
    // Else:
    //  No angle correction
    //
    __inline__ __device__ static void correctAnglesByInnerAngleSum(Cell* cell1, Cell* cell2, Cell* cell3);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ConstructorProcessor::preprocess(SimulationData& data)
{
    //auto& operations = data.cellTypeOperations[CellType_Constructor];
    //auto partition = calcAllThreadsPartition(operations.getNumEntries());
    //for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
    //    completenessCheck(data, statistics, operations.at(i).cell);
    //}
}

__inline__ __device__ void ConstructorProcessor::process(SimulationData& data, SimulationStatistics& statistics, bool isPreview)
{
    auto& operations = data.cellTypeOperations[CellType_Constructor];
    auto partition = calcSystemThreadPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; i += partition.step) {
        processCell(data, statistics, operations.at(i).cell, isPreview);
    }
}

//__inline__ __device__ void ConstructorProcessor::completenessCheck(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
//{
//    if (!cudaSimulationParameters.constructorCompletenessCheck.value) {
//        return;
//    }
//    auto& constructor = cell->cellTypeData.constructor;
//    if (!GenomeDecoder::isFirstNode(constructor)) {
//        return;
//    }
//    if (!SignalProcessor::isAutoTriggered(data, cell, cell->cellTypeData.constructor.autoTriggerInterval)) {
//        return;
//    }
//
//    if (constructor.numExpectedCells == 0 || GenomeDecoder::isFinished(constructor) || !GenomeDecoder::containsSelfReplication(constructor)) {
//        constructor.isReady = true;
//        return;
//    }
//
//    uint32_t tagBit = 1 << toInt(cell->id % 30);
//    atomicOr(&cell->tempValue, toInt(tagBit));
//    auto actualCells = 1;
//
//    auto constexpr QueueLength = 512;
//    Cell* taggedCells[QueueLength];
//    taggedCells[0] = cell;
//    int numTaggedCells = 1;
//    int currentTaggedCellIndex = 0;
//    do {
//        auto currentCell = taggedCells[currentTaggedCellIndex];
//
//        if ((numTaggedCells + 1) % QueueLength != currentTaggedCellIndex) {
//            for (int i = 0, j = currentCell->numConnections; i < j; ++i) {
//                auto& nextCell = currentCell->connections[i].cell;
//                if (nextCell->creatureId == cell->creatureId) {
//                    auto origTagBit = static_cast<uint32_t>(atomicOr(&nextCell->tempValue, toInt(tagBit)));
//                    if ((origTagBit & tagBit) == 0) {
//                        taggedCells[numTaggedCells] = nextCell;
//                        numTaggedCells = (numTaggedCells + 1) % QueueLength;
//                        ++actualCells;
//                    }
//                }
//            }
//        }
//
//        currentTaggedCellIndex = (currentTaggedCellIndex + 1) % QueueLength;
//        if (currentTaggedCellIndex == numTaggedCells) {
//            break;
//        }
//    } while (true);
//    constructor.isReady = (actualCells >= constructor.numExpectedCells);
//}
//
__inline__ __device__ void ConstructorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell, bool isPreview)
{
    if (cell->creature == nullptr) {
        return;
    }
    auto& constructor = cell->cellTypeData.constructor;
    if (SignalProcessor::isAutoOrManuallyTriggered(data, cell, constructor.autoTriggerInterval, isPreview)) {
        constructor.offspring = findOrCreateNewCreature(data, cell);

        if (ConstructorHelper::isFinished(constructor, *constructor.offspring->genome)) {
            return;
        }

        if (!checkForValidConstruction(cell)) {
            constructor.currentNodeIndex = 0;
            constructor.currentConcatenation = 0;
        }

        auto constructionData = createConstructionData(cell);
        if (tryConstructCell(data, statistics, cell, constructionData)) {

            cell->signal.channels[Channels::ConstructorSuccess] = 1;  // Successful

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
                    if (constructor.provideEnergy == ProvideEnergy_FreeGeneration) {
                        constructor.provideEnergy = ProvideEnergy_CellOnly;
                    }
                    constructor.offspring = nullptr;

                    // HACK for preview mode: Do not construct more than one offspring + move seed away
                    if (isPreview) {
                        cell->pos.y += toFloat(PREVIEW_HEIGHT / 3);
                    }
                }
            }
        } else {
            cell->signal.channels[Channels::ConstructorSuccess] = 0;  // Failed
        }
    }
}

__inline__ __device__ Creature* ConstructorProcessor::findOrCreateNewCreature(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellTypeData.constructor;

    if (constructor.offspring != nullptr) {
        return constructor.offspring;
    }

    // No separation => same creature
    auto& genome = cell->creature->genome;
    if (constructor.geneIndex < genome->numGenes) {
        auto const& gene = ConstructorHelper::getCurrentGene(constructor, *genome);
        if (!gene->separation) {
            return cell->creature;
        }
    }

    // Current branch under construction => use creature reference from there
    if (!(ConstructorHelper::isFirstNode(constructor) && ConstructorHelper::isFirstConcatenation(constructor))) {
        auto lastConstructionCell = getLastConstructedCellOnBranch(cell);
        if (lastConstructionCell) {
            return lastConstructionCell->creature;
        }
    }

    // Other branches already constructed => same creature
    if (constructor.currentBranch > 0) {
        return cell->creature;
    }

    // Nothing found => clone creature
    ObjectFactory factory;
    factory.init(&data);
    auto result = factory.cloneCreature(cell->creature);
    result->numCells = 0;
    return result;
}

__inline__ __device__ ConstructorProcessor::ConstructionData ConstructorProcessor::createConstructionData(Cell* cell)
{
    auto& constructor = cell->cellTypeData.constructor;
    auto& genome = constructor.offspring->genome;

    auto isFirstNode = ConstructorHelper::isFirstNode(constructor);
    auto isFirstConcatenation = ConstructorHelper::isFirstConcatenation(constructor);

    ConstructionData result;
    result.creature = constructor.offspring;
    result.gene = ConstructorHelper::getCurrentGene(constructor, *genome);
    result.node = ConstructorHelper::getCurrentNode(constructor, *genome);
    result.isSeparation = result.gene->separation;
    result.isFirstNodeOfFirstConcatenation = isFirstNode && isFirstConcatenation;
    result.isLastNode = ConstructorHelper::isLastNode(constructor, *genome);
    result.isLastNodeOfLastConcatenation = result.isLastNode && ConstructorHelper::isLastConcatenation(constructor, *genome);

    result.hasInfiniteConcatenations = ConstructorHelper::hasInfiniteConcatenations(result.gene);
    result.lastConstructionCell = getLastConstructedCellOnBranch(cell);
    result.angle = result.node->referenceAngle;
    result.cellEnergy = cudaSimulationParameters.normalCellEnergy.value[cell->color];
    if (result.node->cellType == CellTypeGenome_Constructor) {
        auto const& constructorNode = result.node->cellTypeData.constructor;
        if (constructor.provideEnergy == ProvideEnergy_CellAndGene && constructorNode.geneIndex < result.creature->genome->numGenes) {
            auto& referencedGene = result.creature->genome->genes[constructorNode.geneIndex];
            if (!referencedGene.separation) {
                auto requiredEnergyForNodes = GenomeProcessor::getRequiredEnergyForNodes(&referencedGene);
                if (!ConstructorHelper::hasInfiniteConcatenations(&referencedGene)) {
                    result.cellEnergy += requiredEnergyForNodes * referencedGene.numBranches * referencedGene.numConcatenations;
                } else {
                    result.cellEnergy += requiredEnergyForNodes;
                }
            }
        }
    }
    result.depotEnergy = result.node->cellType == CellTypeGenome_Depot ? result.node->cellTypeData.depot.initialStoredUsableEnergy : 0.0f;
    result.numAdditionalConnections = result.node->numAdditionalConnections;

    CudaShapeGenerator shapeGenerator;
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
            }
        }
    } else {
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    }

    if (result.gene->numNodes == 1) {
        result.numAdditionalConnections = 0;
    }

    if (isFirstNode) {
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

__inline__ __device__ Cell*
ConstructorProcessor::tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData)
{
    if (!hostCell->tryLock()) {
        return nullptr;
    }
    if (constructionData.isFirstNodeOfFirstConcatenation) {
        auto newCell = startConstructionOnNewBranch(data, statistics, hostCell, constructionData);

        hostCell->releaseLock();
        return newCell;
    } else {
        if (!constructionData.lastConstructionCell->tryLock()) {
            hostCell->releaseLock();
            return nullptr;
        }
        auto newCell = continueConstructionOnBranch(data, statistics, hostCell, constructionData);

        constructionData.lastConstructionCell->releaseLock();
        hostCell->releaseLock();
        return newCell;
    }
}

__inline__ __device__ Cell* ConstructorProcessor::getLastConstructedCellOnBranch(Cell* hostCell)
{
    auto const& constructor = hostCell->cellTypeData.constructor;
    if (constructor.lastConstructedCellId != VALUE_NOT_SET_UINT64) {
        for (int i = 0; i < hostCell->numConnections; ++i) {
            auto const& connectedCell = hostCell->connections[i].cell;
            if (connectedCell->id == constructor.lastConstructedCellId) {
                return connectedCell;
            }
        }
    }
    return nullptr;
}

__inline__ __device__ Cell* ConstructorProcessor::startConstructionOnNewBranch(
    SimulationData& data,
    SimulationStatistics& statistics,
    Cell* hostCell,
    ConstructionData const& constructionData)
{
    auto& constructor = hostCell->cellTypeData.constructor;

    if (hostCell->numConnections == MAX_CELL_BONDS) {
        return nullptr;
    }
    auto anglesForNewConnection = CellConnectionProcessor::calcLargestGapReferenceAndActualAngle(data, hostCell, constructionData.angle);

    auto newCellDirection = Math::unitVectorOfAngle(anglesForNewConnection.actualAngle);
    float2 newCellPos = hostCell->pos + newCellDirection;

    if (CellConnectionProcessor::existCrossingConnections(
            data, hostCell->pos, newCellPos, cudaSimulationParameters.constructorConnectingCellDistance.value[hostCell->color], hostCell->detached)) {
        return nullptr;
    }

    if (cudaSimulationParameters.constructorCompletenessCheck.value && !constructor.isReady) {
        return nullptr;
    }

    if (!checkAndReduceHostEnergy(data, hostCell, constructionData)) {
        return nullptr;
    }

    // For bending muscle cells: Reset front angle and restore initial angle
    for (int i = 0; i < hostCell->numConnections; ++i) {
        auto const& connectedCell = hostCell->connections[i].cell;
        if (connectedCell->cellType == CellType_Muscle && connectedCell->cellTypeData.muscle.isBendingMuscle()) {
            connectedCell->frontAngle = VALUE_NOT_SET_FLOAT;
            MuscleProcessor::restoreInitialAngleFromPrevious(connectedCell, hostCell, i);

            // Update newCell position and direction for corrected angle
            anglesForNewConnection = CellConnectionProcessor::calcLargestGapReferenceAndActualAngle(data, hostCell, constructionData.angle);
            newCellDirection = Math::unitVectorOfAngle(anglesForNewConnection.actualAngle);
            newCellPos = hostCell->pos + newCellDirection;
        }
    }

    uint64_t cellPointerIndex;
    Cell* newCell = constructCellIntern(data, statistics, cellPointerIndex, hostCell, newCellPos, constructionData);

    if (!newCell->tryLock()) {
        return nullptr;
    }

    if (!constructionData.isLastNodeOfLastConcatenation || !constructionData.isSeparation) {
        auto distance = constructionData.isLastNodeOfLastConcatenation && !constructionData.isSeparation
            ? constructionData.gene->connectionDistance
            : constructionData.gene->connectionDistance + cudaSimulationParameters.constructorAdditionalOffspringDistance;
        if (!CellConnectionProcessor::tryAddConnections(data, hostCell, newCell, anglesForNewConnection.referenceAngle, 0, distance)) {
            CellConnectionProcessor::scheduleDeleteCell(data, cellPointerIndex);
        }
    }
    if ((constructionData.isSeparation || constructor.geneIndex == 0) && constructionData.isFirstNodeOfFirstConcatenation && constructor.currentBranch == 0) {
        newCell->headCell = true;
    }
    activateNewCellOnLastNode(newCell, hostCell, constructionData);

    newCell->releaseLock();
    return newCell;
}

__inline__ __device__ Cell* ConstructorProcessor::continueConstructionOnBranch(
    SimulationData& data,
    SimulationStatistics& statistics,
    Cell* hostCell,
    ConstructionData const& constructionData)
{
    auto const& lastCell = constructionData.lastConstructionCell;
    auto posDelta = data.cellMap.getCorrectedDirection(lastCell->pos - hostCell->pos);
    auto angleFromPreviousForNewCell = 180.0f - constructionData.angle;

    auto desiredDistance = constructionData.gene->connectionDistance;
    auto constructionSiteDistance = hostCell->getRefDistance(lastCell);
    posDelta = Math::getNormalized(posDelta) * (constructionSiteDistance - desiredDistance);
    if (Math::length(posDelta) <= cudaSimulationParameters.minCellDistance.value
        || constructionSiteDistance - desiredDistance < cudaSimulationParameters.minCellDistance.value) {
        return nullptr;
    }

    auto newCellPos = hostCell->pos + posDelta;
    if (CellConnectionProcessor::existCrossingConnections(
            data,
            hostCell->pos,
            constructionData.lastConstructionCell->pos,
            cudaSimulationParameters.constructorConnectingCellDistance.value[hostCell->color],
            hostCell->detached)) {
        return nullptr;
    }

    Cell* cellsToConnect[MAX_CELL_BONDS];
    int numCellsToConnect;
    getCellsToConnect(cellsToConnect, numCellsToConnect, data, hostCell, newCellPos, constructionData);

    if (constructionData.numAdditionalConnections != -1) {
        if (numCellsToConnect < constructionData.numAdditionalConnections) {
            return nullptr;
        }
    }

    if (!checkAndReduceHostEnergy(data, hostCell, constructionData)) {
        return nullptr;
    }

    // For bending muscle cells: Reset front angle and restore initial angle
    if (lastCell->cellType == CellType_Muscle && lastCell->cellTypeData.muscle.isBendingMuscle()) {
        lastCell->frontAngle = VALUE_NOT_SET_FLOAT;
        auto connectionIndex = hostCell->getConnectionIndex(lastCell);
        MuscleProcessor::restoreInitialAngleFromPrevious(lastCell, hostCell, connectionIndex);
    }

    uint64_t cellPointerIndex;
    Cell* newCell = constructCellIntern(data, statistics, cellPointerIndex, hostCell, newCellPos, constructionData);

    if (!newCell->tryLock()) {
        return nullptr;
    }
    if (constructionData.lastConstructionCell->cellState == CellState_Dying) {
        newCell->cellState = CellState_Dying;
    }

    float origAngleFromPreviousOnHostCell;
    for (int i = 0; i < hostCell->numConnections; ++i) {
        if (hostCell->connections[i].cell == constructionData.lastConstructionCell) {
            origAngleFromPreviousOnHostCell = hostCell->connections[i].angleFromPrevious;
            break;
        }
    }

    float origAngleFromPreviousOnLastConstructedCell;
    for (int i = 0; i < constructionData.lastConstructionCell->numConnections; ++i) {
        if (constructionData.lastConstructionCell->connections[i].cell == hostCell) {
            origAngleFromPreviousOnLastConstructedCell = constructionData.lastConstructionCell->connections[i].angleFromPrevious;
        }
    }

    // Move connection between lastConstructionCell and hostCell to a connection between lastConstructionCell and newCell
    for (int i = 0; i < lastCell->numConnections; ++i) {
        auto& connection = lastCell->connections[i];
        if (connection.cell == hostCell) {
            connection.cell = newCell;
            connection.distance = desiredDistance;
            connection.angleFromPrevious = origAngleFromPreviousOnLastConstructedCell;
            newCell->numConnections = 1;
            newCell->connections[0].cell = lastCell;
            newCell->connections[0].distance = desiredDistance;
            newCell->connections[0].angleFromPrevious = 360.0f;
            CellConnectionProcessor::deleteConnectionOneWay(hostCell, lastCell);
            break;
        }
    }

    // Possibly connect newCell to hostCell
    bool adaptReferenceAngles = false;
    if (!constructionData.isLastNodeOfLastConcatenation || !constructionData.isSeparation) {

        auto distance = constructionData.gene->connectionDistance;
        if (!constructionData.isLastNodeOfLastConcatenation) {
            distance += cudaSimulationParameters.constructorAdditionalOffspringDistance;
        }

        if (!CellConnectionProcessor::tryAddConnections(data, newCell, hostCell, 0, origAngleFromPreviousOnHostCell, distance)) {
            CellConnectionProcessor::scheduleDeleteCell(data, cellPointerIndex);
            hostCell->cellState = CellState_Dying;
            for (int i = 0; i < hostCell->numConnections; ++i) {
                auto const& connectedCell = hostCell->connections[i].cell;
                if (connectedCell->creature == hostCell->creature) {
                    connectedCell->cellState = CellState_Detaching;
                }
            }
        } else {
            adaptReferenceAngles = true;
        }
    }

    // Get surrounding cells
    if (numCellsToConnect > 0 && constructionData.numAdditionalConnections != 0) {

        // Sort surrounding cells by distance from newCell
        bubbleSort(cellsToConnect, numCellsToConnect, [&](auto const& cell1, auto const& cell2) {
            auto dist1 = data.cellMap.getDistance(cell1->pos, newCellPos);
            auto dist2 = data.cellMap.getDistance(cell2->pos, newCellPos);
            return dist1 < dist2;
        });

        // Connect surrounding cells if possible
        int numConnectedCells = 0;
        for (int i = 0; i < numCellsToConnect; ++i) {
            Cell* otherCell = cellsToConnect[i];

            if (otherCell->tryLock()) {
                if (newCell->numConnections < MAX_CELL_BONDS && otherCell->numConnections < MAX_CELL_BONDS) {
                    if (CellConnectionProcessor::tryAddConnections(data, newCell, otherCell, 0, 0, desiredDistance, constructionData.gene->angleAlignment)) {
                        ++numConnectedCells;
                    }
                }
                otherCell->releaseLock();
            }
            if (constructionData.numAdditionalConnections != -1) {
                if (numConnectedCells == constructionData.numAdditionalConnections) {
                    break;
                }
            }
        }
    }

    if (adaptReferenceAngles) {

        // Adapt angles on new cell
        auto n = newCell->numConnections;
        auto lastCellIndex = newCell->getConnectionIndex(constructionData.lastConstructionCell);
        auto hostCellIndex = newCell->getConnectionIndex(hostCell);

        float consumedAngle1 = 0;
        if (n > 2) {
            for (int i = lastCellIndex; (i + n) % n != (hostCellIndex + 1) % n && (i + n) % n != hostCellIndex; --i) {
                consumedAngle1 += newCell->connections[(i + n) % n].angleFromPrevious;
            }
        }

        float consumedAngle2 = 0;
        if (n > 2) {
            for (int i = lastCellIndex + 1; i % n != hostCellIndex; ++i) {
                consumedAngle2 += newCell->connections[i % n].angleFromPrevious;
            }
        }
        if (angleFromPreviousForNewCell - consumedAngle1 >= 0 && 360.0f - angleFromPreviousForNewCell - consumedAngle2 >= 0) {
            newCell->connections[(hostCellIndex + 1) % n].angleFromPrevious = angleFromPreviousForNewCell - consumedAngle1;
            newCell->connections[hostCellIndex].angleFromPrevious = 360.0f - angleFromPreviousForNewCell - consumedAngle2;
        }

        // Adapt angles on other connected cells
        for (int i = lastCellIndex; (i + n) % n != (hostCellIndex + 1) % n && (i + n) % n != hostCellIndex; --i) {
            correctAnglesByInnerAngleSum(newCell->getConnectedCell(i), newCell, newCell->getConnectedCell(i - 1));
        }
        for (int i = lastCellIndex + 1; i % n != hostCellIndex; ++i) {
            correctAnglesByInnerAngleSum(newCell->getConnectedCell(i - 1), newCell, newCell->getConnectedCell(i));
        }
    }

    activateNewCellOnLastNode(newCell, hostCell, constructionData);

    newCell->releaseLock();
    return newCell;
}

__inline__ __device__ void ConstructorProcessor::getCellsToConnect(
    Cell* result[],
    int& numResultCells,
    SimulationData& data,
    Cell* hostCell,
    float2 const& newCellPos,
    ConstructionData const& constructionData)
{
    numResultCells = 0;

    if (constructionData.numAdditionalConnections == 0) {
        return;
    }

    Cell* nearCells[MAX_CELL_BONDS * 4];
    int numNearCells;
    data.cellMap.getMatchingCells(
        nearCells,
        MAX_CELL_BONDS * 4,
        numNearCells,
        newCellPos,
        cudaSimulationParameters.constructorConnectingCellDistance.value[hostCell->color],
        hostCell->detached,
        [&](Cell* const& otherCell) { return otherCell != hostCell && otherCell != constructionData.lastConstructionCell; });

    Cell* otherCellCandidates[MAX_CELL_BONDS * 2];
    int numOtherCellCandidates;

    if (constructionData.requiredNodeId1 == -1) {
        auto const& lastConstructionCell = constructionData.lastConstructionCell;

        float angleFromPrevious1;
        float angleFromPrevious2;
        for (int i = 0; i < lastConstructionCell->numConnections; ++i) {
            if (lastConstructionCell->connections[i].cell == hostCell) {
                angleFromPrevious1 = lastConstructionCell->connections[i].angleFromPrevious;
                angleFromPrevious2 = lastConstructionCell->connections[(i + 1) % lastConstructionCell->numConnections].angleFromPrevious;
                break;
            }
        }
        auto n = Math::getNormalized(hostCell->pos - lastConstructionCell->pos);
        Math::rotateQuarterClockwise(n);

        // assemble surrounding cell candidates
        data.cellMap.getMatchingCells(
            otherCellCandidates,
            MAX_CELL_BONDS * 2,
            numOtherCellCandidates,
            newCellPos,
            cudaSimulationParameters.constructorConnectingCellDistance.value[hostCell->color],
            hostCell->detached,
            [&](Cell* const& otherCell) {
                if (otherCell == constructionData.lastConstructionCell || otherCell == hostCell
                    || (otherCell->cellState != CellState_Constructing && otherCell->activationTime == 0) || otherCell->creature != constructionData.creature
                    || otherCell->parentNodeIndex != hostCell->nodeIndex) {
                    return false;
                }

                // discard cells that are not on the correct side
                if (abs(angleFromPrevious1 - angleFromPrevious2) > NEAR_ZERO) {
                    auto delta = data.cellMap.getCorrectedDirection(otherCell->pos - lastConstructionCell->pos);
                    if (angleFromPrevious2 < angleFromPrevious1) {
                        if (Math::dot(delta, n) < 0) {
                            return false;
                        }
                    }
                    if (angleFromPrevious2 > angleFromPrevious1) {
                        if (Math::dot(delta, n) > 0) {
                            return false;
                        }
                    }
                }
                return true;
            });
    } else {
        data.cellMap.getMatchingCells(
            otherCellCandidates,
            MAX_CELL_BONDS * 2,
            numOtherCellCandidates,
            newCellPos,
            cudaSimulationParameters.constructorConnectingCellDistance.value[hostCell->color],
            hostCell->detached,
            [&](Cell* const& otherCell) {
                if (otherCell->cellState != CellState_Constructing || otherCell->creature != constructionData.creature
                    || otherCell->parentNodeIndex != hostCell->nodeIndex) {
                    return false;
                }
                if (constructionData.numAdditionalConnections >= 1 && otherCell->nodeIndex == constructionData.requiredNodeId1) {
                    return true;
                }
                if (constructionData.numAdditionalConnections == 2 && otherCell->nodeIndex == constructionData.requiredNodeId2) {
                    return true;
                }
                return false;
            });
    }

    // evaluate candidates (locking is needed for the evaluation)
    for (int i = 0; i < numOtherCellCandidates; ++i) {
        Cell* otherCell = otherCellCandidates[i];
        if (otherCell->tryLock()) {
            bool crossingLinks = false;
            for (int j = 0; j < numNearCells; ++j) {
                auto nearCell = nearCells[j];
                if (otherCell == nearCell) {
                    continue;
                }
                if (nearCell->tryLock()) {
                    for (int k = 0; k < nearCell->numConnections; ++k) {
                        if (nearCell->connections[k].cell == otherCell) {
                            continue;
                        }
                        if (Math::crossing(newCellPos, otherCell->pos, nearCell->pos, nearCell->connections[k].cell->pos)) {
                            crossingLinks = true;
                        }
                    }
                    nearCell->releaseLock();
                } else {
                    // crossingLinks = true;
                }
            }
            if (!crossingLinks) {
                auto delta = data.cellMap.getCorrectedDirection(newCellPos - otherCell->pos);
                if (CellConnectionProcessor::hasAngleSpace(data, otherCell, Math::angleOfVector(delta), constructionData.gene->angleAlignment)) {
                    result[numResultCells++] = otherCell;
                }
            }
            otherCell->releaseLock();
        }
        if (numResultCells == MAX_CELL_BONDS) {
            break;
        }
    }
}

__inline__ __device__ Cell* ConstructorProcessor::constructCellIntern(
    SimulationData& data,
    SimulationStatistics& statistics,
    uint64_t& cellIndex,
    Cell* hostCell,
    float2 posOfNewCell,
    ConstructionData const& constructionData)
{
    auto& constructor = hostCell->cellTypeData.constructor;

    data.cellMap.correctPosition(posOfNewCell);

    ObjectFactory factory;
    factory.init(&data);
    Cell* result = factory.createCellFromNode(
        cellIndex,
        constructionData.creature,
        constructor.geneIndex,
        constructor.currentNodeIndex,
        hostCell->nodeIndex,
        posOfNewCell,
        hostCell->vel,
        constructionData.cellEnergy);
    result->frontAngleId = hostCell->frontAngleId;

    constructor.lastConstructedCellId = result->id;

    // Inherit free energy provision from parent in case that offspring constructs a non-separating gene
    if (constructor.provideEnergy == ProvideEnergy_FreeGeneration && result->cellType == CellType_Constructor) {
        auto const& offspringConstructor = result->cellTypeData.constructor;
        auto const& offspringGenome = constructionData.creature->genome;
        if (offspringConstructor.geneIndex < offspringGenome->numGenes) {
            auto const& offspringGene = offspringGenome->genes[offspringConstructor.geneIndex];
            if (!offspringGene.separation) {
                result->cellTypeData.constructor.provideEnergy = ProvideEnergy_FreeGeneration;
            }
        }
    }

    statistics.incNumCreatedCells(hostCell->color);

    return result;
}

__inline__ __device__ bool ConstructorProcessor::checkForValidConstruction(Cell* hostCell)
{
    auto& constructor = hostCell->cellTypeData.constructor;
    auto& genome = constructor.offspring->genome;

    auto lastConstructionCell = getLastConstructedCellOnBranch(hostCell);
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


__inline__ __device__ bool ConstructorProcessor::checkAndReduceHostEnergy(SimulationData& data, Cell* hostCell, ConstructionData const& constructionData)
{
    if (hostCell->cellTypeData.constructor.provideEnergy == ProvideEnergy_FreeGeneration) {
        return true;
    }

    auto requiredEnergy = constructionData.cellEnergy + constructionData.depotEnergy;
    auto normalCellEnergy = cudaSimulationParameters.normalCellEnergy.value[hostCell->color];

    if (cudaSimulationParameters.externalEnergyControlToggle.value && hostCell->usableEnergy < requiredEnergy + normalCellEnergy
        && cudaSimulationParameters.externalEnergyInflowFactor.value[hostCell->color] > 0) {
        auto externalEnergyPortion = [&] {
            if (cudaSimulationParameters.externalEnergyInflowOnlyForNonSelfReplicators.value) {
                return !constructionData.isSeparation && !ConstructorHelper::isFinished(hostCell->cellTypeData.constructor, *constructionData.creature->genome)
                    ? requiredEnergy * cudaSimulationParameters.externalEnergyInflowFactor.value[hostCell->color]
                    : 0.0f;
            } else {
                return requiredEnergy * cudaSimulationParameters.externalEnergyInflowFactor.value[hostCell->color];
            }
        }();

        auto origExternalEnergy = alienAtomicRead(data.externalEnergy);
        if (origExternalEnergy == Infinity<float>::value) {
            hostCell->usableEnergy += externalEnergyPortion;
        } else {
            externalEnergyPortion = max(0.0f, min(origExternalEnergy, externalEnergyPortion));
            auto origExternalEnergy_tickLater = atomicAdd(data.externalEnergy, -externalEnergyPortion);
            if (origExternalEnergy_tickLater >= externalEnergyPortion) {
                hostCell->usableEnergy += externalEnergyPortion;
            } else {
                atomicAdd(data.externalEnergy, externalEnergyPortion);
            }
        }
    }

    auto externalEnergyConditionalInflowFactor = [&] {
        if (!cudaSimulationParameters.externalEnergyControlToggle.value) {
            return 0.0f;
        }
        if (cudaSimulationParameters.externalEnergyInflowOnlyForNonSelfReplicators.value) {
            return !constructionData.isSeparation ? cudaSimulationParameters.externalEnergyConditionalInflowFactor.value[hostCell->color] : 0.0f;
        } else {
            return cudaSimulationParameters.externalEnergyConditionalInflowFactor.value[hostCell->color];
        }
    }();

    auto energyNeededFromHost =
        max(0.0f, requiredEnergy - normalCellEnergy) + min(requiredEnergy, normalCellEnergy) * (1.0f - externalEnergyConditionalInflowFactor);

    if (externalEnergyConditionalInflowFactor < 1.0f && hostCell->usableEnergy < normalCellEnergy + energyNeededFromHost) {
        return false;
    }
    auto energyNeededFromExternalSource = requiredEnergy - energyNeededFromHost;
    auto orig = atomicAdd(data.externalEnergy, -energyNeededFromExternalSource);
    if (orig < energyNeededFromExternalSource) {
        atomicAdd(data.externalEnergy, energyNeededFromExternalSource);
        if (hostCell->usableEnergy < normalCellEnergy + requiredEnergy) {
            return false;
        }
        hostCell->usableEnergy -= requiredEnergy;
    } else {
        hostCell->usableEnergy -= energyNeededFromHost;
    }
    return true;
}

__inline__ __device__ void ConstructorProcessor::activateNewCellOnLastNode(Cell* newCell, Cell* hostCell, ConstructionData const& constructionData)
{
    if (constructionData.isLastNode) {
        newCell->cellState = CellState_Activating;
        alienAtomicAdd32(&newCell->creature->frontAngleId, static_cast<uint32_t>(1));
    }
}

__inline__ __device__ void ConstructorProcessor::correctAnglesByInnerAngleSum(Cell* cell1, Cell* cell2, Cell* cell3)
{
    // Check if cell3 connects back to cell1 (directly or via further cells, not through cell2)
    // to form a closed polygon
    
    // Determine traversal direction to find minimal polygon
    // Find indices of cell1 and cell3 in cell2's connections (which are sorted clockwise)
    int cell1IndexInCell2 = cell2->getConnectionIndex(cell1);
    int cell3IndexInCell2 = cell2->getConnectionIndex(cell3);

    // Determine if we go clockwise or counter-clockwise from cell3 to find cell1
    // The polygon is: cell1 -> cell2 -> cell3 -> ... -> cell1
    // From cell2's perspective, we go clockwise from cell1 to cell3
    // So from cell3, we should continue in a consistent direction to find cell1
    bool goClockwiseFromCell3;
    if (cell3IndexInCell2 == (cell1IndexInCell2 + 1) % cell2->numConnections) {
        // cell3 is clockwise from cell1 in cell2's connections
        goClockwiseFromCell3 = true;
    } else {
        // cell3 is counter-clockwise from cell1 (wrapped around)
        goClockwiseFromCell3 = false;
    }
    
    // Find the minimal path from cell3 to cell1 (not going through cell2)
    Cell* currentCell = cell3;
    Cell* previousCell = cell2;
    int numIntermediateCells = 0;
    bool foundPolygon = false;
    Cell* cell4 = nullptr; // The next cell after cell3 in the polygon
    
    // Find cell2's index in cell3's connections to determine traversal direction
    int cell2IndexInCell3 = cell3->getConnectionIndex(cell2);
    
    constexpr int maxPolygonSize = 50;
    float currentAngleSum = 0.0f;
    for (int step = 0; step < maxPolygonSize; ++step) {
        Cell* nextCell = nullptr;
        
        if (step == 0) {
            int startIndex = goClockwiseFromCell3 ? cell2IndexInCell3 + 1 : cell2IndexInCell3 - 1;
            
            for (int i = 0; i < cell3->numConnections; ++i) {
                int index = goClockwiseFromCell3 ? startIndex + i : startIndex - i;
                    
                Cell* candidate = cell3->getConnectedCell(index);
                if (candidate == cell1) {
                    nextCell = candidate;
                    foundPolygon = true;
                    break;
                } else if (candidate != cell2) {
                    nextCell = candidate;
                    cell4 = candidate;
                    break;
                }
            }
        } else {
            // Subsequent steps: find next cell that's not the previous one
            int prevIndex = currentCell->getConnectionIndex(previousCell);
            
            // Continue in the same general direction
            for (int i = 1; i < currentCell->numConnections; ++i) {
                int index = goClockwiseFromCell3 ? prevIndex + i : prevIndex - i;
                    
                Cell* candidate = currentCell->getConnectedCell(index);
                if (candidate == cell1) {
                    nextCell = candidate;
                    foundPolygon = true;
                    break;
                } else if (candidate != cell2) {
                    nextCell = candidate;
                    break;
                }
            }
        }

        if (step > 0) {
            if (!goClockwiseFromCell3) {
                currentAngleSum += currentCell->getAngelSpan(nextCell, previousCell);
            } else {
                currentAngleSum += currentCell->getAngelSpan(previousCell, nextCell);
            }
        }

        if (foundPolygon || nextCell == nullptr) {
            break;
        }
        
        previousCell = currentCell;
        currentCell = nextCell;
        numIntermediateCells++;
    }
    
    if (!foundPolygon) {
        // No closed polygon found, no angle correction based on polygon possible
        return;
    }
    
    // If cell4 is still null, it means cell3 connects directly to cell1
    if (cell4 == nullptr) {
        cell4 = cell1;
    }
    
    // Number of vertices in the polygon
    int numVertices = 3 + numIntermediateCells; // cell1, cell2, cell3, + intermediate cells
    
    // Calculate expected inner angle sum for an n-sided polygon: (n - 2) * 180 degrees
    float expectedAngleSum = (numVertices - 2) * 180.0f;
    
    Cell* lastCellBeforeCell1 = currentCell; // This is the last cell we visited before reaching cell1
    if (!goClockwiseFromCell3) {
        currentAngleSum += cell1->getAngelSpan(cell2, lastCellBeforeCell1);
        currentAngleSum += cell2->getAngelSpan(cell3, cell1);
        currentAngleSum += cell3->getAngelSpan(cell4, cell2);
    } else {
        currentAngleSum += cell1->getAngelSpan(lastCellBeforeCell1, cell2);
        currentAngleSum += cell2->getAngelSpan(cell1, cell3);
        currentAngleSum += cell3->getAngelSpan(cell2, cell4);
    }
    
    float angleCorrection = expectedAngleSum - currentAngleSum;

    int cell2Index = cell3->getConnectionIndex(cell2);
    
    if (!goClockwiseFromCell3) {
        cell3->increaseAngle(cell2Index, angleCorrection);
    } else {
        cell3->increaseAngle(cell2Index, -angleCorrection);
    }
}

