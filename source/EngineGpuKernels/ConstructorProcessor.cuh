#pragma once

#include "EngineInterface/CellTypeConstants.h"

#include "SignalProcessor.cuh"
#include "SimulationCudaFacade.cuh"
#include "SimulationStatistics.cuh"
#include "CellConnectionProcessor.cuh"
#include "CudaShapeGenerator.cuh"
#include "ConstructorHelper.cuh"
#include "Genome.cuh"

class ConstructorProcessor
{
public:
    __inline__ __device__ static void preprocess(SimulationData& data);
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics, bool isPreview);

private:
    struct ConstructionData
    {
        bool isPreview;

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
        float energy;
        int numAdditionalConnections;  // -1 = none
        int requiredNodeId1;    // -1 = none
        int requiredNodeId2;    // -1 = none
    };
//
//    __inline__ __device__ static void completenessCheck(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
//
    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell, bool isPreview);
    __inline__ __device__ static Creature* findOrCreateNewCreature(SimulationData& data, Cell* cell);
    __inline__ __device__ static ConstructionData createConstructionData(Cell* cell, bool isPreview);

    __inline__ __device__ static Cell* tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static Cell* getLastConstructedCellOnBranch(Cell* hostCell);
    __inline__ __device__ static Cell* startConstructionOnNewBranch(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);
    __inline__ __device__ static Cell* continueConstructionOnBranch(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);

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

    //    __inline__ __device__ static float calcNumCells(int color, uint8_t* genome, uint16_t genomeSize);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ConstructorProcessor::preprocess(SimulationData& data)
{
    //auto& operations = data.cellTypeOperations[CellType_Constructor];
    //auto partition = calcAllThreadsPartition(operations.getNumEntries());
    //for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
    //    completenessCheck(data, statistics, operations.at(i).cell);
    //}
}

__inline__ __device__ void ConstructorProcessor::process(SimulationData& data, SimulationStatistics& statistics, bool isPreview)
{
    auto& operations = data.cellTypeOperations[CellType_Constructor];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
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

        if (ConstructorHelper::isFinished(constructor, constructor.offspring->genome)) {
            return;
        }

        if (!checkForValidConstruction(cell)) {
            constructor.currentNodeIndex = 0;
            constructor.currentConcatenation = 0;
        }

        auto constructionData = createConstructionData(cell, isPreview);
        if (tryConstructCell(data, statistics, cell, constructionData)) {
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
                    constructor.offspring = nullptr;

                    // HACK for preview mode: Do not construct more than one offspring + move seed away
                    if (isPreview) {
                        constructor.currentConcatenation = constructionData.gene->numConcatenations;  
                        cell->pos.y += toFloat(PREVIEW_HEIGHT / 3);
                    }
                }
            }
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
    if (constructor.geneIndex < genome.numGenes) {
        auto const& gene = ConstructorHelper::getCurrentGene(constructor, genome);
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
    return factory.cloneCreature(cell->creature);
}

__inline__ __device__ ConstructorProcessor::ConstructionData ConstructorProcessor::createConstructionData(Cell* cell, bool isPreview)
{
    auto& constructor = cell->cellTypeData.constructor;
    auto& genome = constructor.offspring->genome;

    auto isFirstNode = ConstructorHelper::isFirstNode(constructor);
    auto isFirstConcatenation = ConstructorHelper::isFirstConcatenation(constructor);

    ConstructionData result;
    result.isPreview = isPreview;
    result.creature = constructor.offspring;
    result.gene = ConstructorHelper::getCurrentGene(constructor, genome);
    result.node = ConstructorHelper::getCurrentNode(constructor, genome);
    result.isSeparation = result.gene->separation;
    result.isFirstNodeOfFirstConcatenation = isFirstNode && isFirstConcatenation;
    result.isLastNode = ConstructorHelper::isLastNode(constructor, genome);
    result.isLastNodeOfLastConcatenation = result.isLastNode && ConstructorHelper::isLastConcatenation(constructor, genome);
    
    result.hasInfiniteConcatenations = ConstructorHelper::hasInfiniteConcatenations(result.gene);
    result.lastConstructionCell = getLastConstructedCellOnBranch(cell);
    result.angle = result.node->referenceAngle;
    result.energy = cudaSimulationParameters.normalCellEnergy.value[cell->color];
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
    if (constructionData.isSeparation && constructionData.isFirstNodeOfFirstConcatenation && constructor.currentBranch == 0) {
        newCell->isFrontAngleRefCell = true;
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
    posDelta = Math::normalized(posDelta) * (constructionSiteDistance - desiredDistance);

    if (Math::length(posDelta) <= cudaSimulationParameters.minCellDistance.value
        || constructionSiteDistance - desiredDistance < cudaSimulationParameters.minCellDistance.value) {
        return nullptr;
    }

    auto newCellPos = hostCell->pos + posDelta;

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
    bool adaptReferenceAngle = false;
    if (!constructionData.isLastNodeOfLastConcatenation || !constructionData.isSeparation) {

        auto distance = constructionData.gene->connectionDistance;
        if (!constructionData.isLastNodeOfLastConcatenation) {
            distance += cudaSimulationParameters.constructorAdditionalOffspringDistance;
        }

        if (!CellConnectionProcessor::tryAddConnections(
                data,
                newCell,
                hostCell,
                0,
                origAngleFromPreviousOnHostCell,
                distance)) {
            CellConnectionProcessor::scheduleDeleteCell(data, cellPointerIndex);
            hostCell->cellState = CellState_Dying;
            for (int i = 0; i < hostCell->numConnections; ++i) {
                auto const& connectedCell = hostCell->connections[i].cell;
                if (connectedCell->creature == hostCell->creature) {
                    connectedCell->cellState = CellState_Detaching;
                }
            }
        } else {
            adaptReferenceAngle = true;
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

    // Adapt angles according to genome
    if (adaptReferenceAngle) {
        auto n = newCell->numConnections;
        int constructionIndex = 0;
        for (; constructionIndex < n; ++constructionIndex) {
            if (newCell->connections[constructionIndex].cell == constructionData.lastConstructionCell) {
                break;
            }
        }
        int hostIndex = 0;
        for (; hostIndex < n; ++hostIndex) {
            if (newCell->connections[hostIndex].cell == hostCell) {
                break;
            }
        }

        float consumedAngle1 = 0;
        if (n > 2) {
            for (int i = constructionIndex; (i + n) % n != (hostIndex + 1) % n && (i + n) % n != hostIndex; --i) {
                consumedAngle1 += newCell->connections[(i + n) % n].angleFromPrevious;
            }
        }

        float consumedAngle2 = 0;
        if (n > 2) {
            for (int i = constructionIndex + 1; i % n != hostIndex; ++i) {
                consumedAngle2 += newCell->connections[i % n].angleFromPrevious;
            }
        }
        if (angleFromPreviousForNewCell - consumedAngle1 >= 0 && 360.0f - angleFromPreviousForNewCell - consumedAngle2 >= 0) {
            newCell->connections[(hostIndex + 1) % n].angleFromPrevious = angleFromPreviousForNewCell - consumedAngle1;
            newCell->connections[hostIndex].angleFromPrevious = 360.0f - angleFromPreviousForNewCell - consumedAngle2;
        }
    }

    activateNewCellOnLastNode(newCell, hostCell, constructionData);

    // Edge case for bending muscle cells with one connection: Reset initial angle
    if (lastCell->numConnections == 1 && lastCell->cellType == CellType_Muscle && lastCell->cellTypeData.muscle.isBendingMuscle()) {
        if (lastCell->cellTypeData.muscle.mode == MuscleMode_AutoBending) {
            lastCell->cellTypeData.muscle.modeData.angleBending.initialAngle = 0;
        } else if (lastCell->cellTypeData.muscle.mode == MuscleMode_ManualBending) {
            lastCell->cellTypeData.muscle.modeData.manualBending.initialAngle = 0;
        } else if (lastCell->cellTypeData.muscle.mode == MuscleMode_AngleBending) {
            lastCell->cellTypeData.muscle.modeData.angleBending.initialAngle = 0;
        }
    }

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
        auto n = Math::normalized(hostCell->pos - lastConstructionCell->pos);
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
                    || (otherCell->cellState != CellState_Constructing && otherCell->activationTime == 0)
                    || otherCell->creature != constructionData.creature || otherCell->parentNodeIndex != hostCell->nodeIndex) {
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

__inline__ __device__ Cell*
ConstructorProcessor::constructCellIntern(
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
        cellIndex, constructionData.creature, constructor.geneIndex, constructor.currentNodeIndex, hostCell->nodeIndex, posOfNewCell, hostCell->vel, constructionData.energy);

    constructor.lastConstructedCellId = result->id;

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
        int numConstructedCells = ConstructorHelper::getNumConstructedCellsOnBranch(constructor, genome);
        return numConstructedCells <= 1;
    }
    return true;
}


__inline__ __device__ bool ConstructorProcessor::checkAndReduceHostEnergy(SimulationData& data, Cell* hostCell, ConstructionData const& constructionData)
{
    // HACK for preview mode: Construction does not consume energy
    if (constructionData.isPreview) {
        return true;
    }

    auto normalCellEnergy = cudaSimulationParameters.normalCellEnergy.value[hostCell->color];

    if (cudaSimulationParameters.externalEnergyControlToggle.value && hostCell->energy < constructionData.energy + normalCellEnergy
        && cudaSimulationParameters.externalEnergyInflowFactor.value[hostCell->color] > 0) {
        auto externalEnergyPortion = [&] {
            if (cudaSimulationParameters.externalEnergyInflowOnlyForNonSelfReplicators.value) {
                return !constructionData.isSeparation && !ConstructorHelper::isFinished(hostCell->cellTypeData.constructor, constructionData.creature->genome)
                    ? constructionData.energy * cudaSimulationParameters.externalEnergyInflowFactor.value[hostCell->color]
                    : 0.0f;
            } else {
                return constructionData.energy * cudaSimulationParameters.externalEnergyInflowFactor.value[hostCell->color];
            }
        }();

        auto origExternalEnergy = alienAtomicRead(data.externalEnergy);
        if (origExternalEnergy == Infinity<float>::value) {
            hostCell->energy += externalEnergyPortion;
        } else {
            externalEnergyPortion = max(0.0f, min(origExternalEnergy, externalEnergyPortion));
            auto origExternalEnergy_tickLater = atomicAdd(data.externalEnergy, -externalEnergyPortion);
            if (origExternalEnergy_tickLater >= externalEnergyPortion) {
                hostCell->energy += externalEnergyPortion;
            } else {
                atomicAdd(data.externalEnergy, externalEnergyPortion);
            }
        }
    }

    auto externalEnergyConditionalInflowFactor =
        [&] {
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
        max(0.0f, constructionData.energy - normalCellEnergy) + min(constructionData.energy, normalCellEnergy) * (1.0f - externalEnergyConditionalInflowFactor);

    if (externalEnergyConditionalInflowFactor < 1.0f && hostCell->energy < normalCellEnergy + energyNeededFromHost) {
        return false;
    }
    auto energyNeededFromExternalSource = constructionData.energy - energyNeededFromHost;
    auto orig = atomicAdd(data.externalEnergy, -energyNeededFromExternalSource);
    if (orig < energyNeededFromExternalSource) {
        atomicAdd(data.externalEnergy, energyNeededFromExternalSource);
        if (hostCell->energy < normalCellEnergy + constructionData.energy) {
            return false;
        }
        hostCell->energy -= constructionData.energy;
    } else {
        hostCell->energy -= energyNeededFromHost;
    }
    return true;
}

__inline__ __device__ void ConstructorProcessor::activateNewCellOnLastNode(Cell* newCell, Cell* hostCell, ConstructionData const& constructionData)
{
    // TODO implement better logic for frontAngle setting
    if (/*constructionData.isLastNodeOfLastConcatenation || (*/constructionData.isLastNode /*&& constructionData.hasInfiniteConcatenations)*/) {
        newCell->cellState = CellState_Activating;
        alienAtomicAdd32(&newCell->creature->frontAngleId, static_cast<uint32_t>(1));
        //if (constructionData.isSeparation) {
        //    newCell->frontAngle = constructionData.creature->genome.frontAngle;
        //} else {
        //    if (hostCell->numConnections > 1) {
        //        newCell->frontAngle =
        //            Math::normalizedAngle(hostCell->frontAngle + (180.0f - hostCell->getAngelSpan(hostCell->connections[0].cell, newCell)), -180.0f);
        //        if (newCell->numConnections > 1) {
        //            newCell->frontAngle =
        //                Math::normalizedAngle(newCell->frontAngle + newCell->getAngelSpan(newCell->connections[0].cell, hostCell), -180.0f);
        //        }
        //    } else {
        //        if (newCell->numConnections > 1) {
        //            newCell->frontAngle =
        //                Math::normalizedAngle(hostCell->frontAngle + (180.0f - newCell->getAngelSpan(hostCell, newCell->connections[0].cell)), -180.0f);
        //        } else {
        //            newCell->frontAngle = hostCell->frontAngle - 180.0f;
        //        }
        //    }
        //}
    }
}

//
//__inline__ __device__ float ConstructorProcessor::calcNumCells(int color, uint8_t* genome, uint16_t genomeSize)
//{
//    auto result = 0.0f;
//
//    auto lastDepth = 0;
//    auto numRamifications = 1;
//    auto numCellsRamificationFactor =
//        cudaSimulationParameters.numCellsMeasurementToggle.value ? cudaSimulationParameters.numCellsRamificationFactor.value[color] : 0.0f;
//    auto sizeFactor =
//        cudaSimulationParameters.numCellsMeasurementToggle.value ? cudaSimulationParameters.numCellsSizeFactor.value[color] : 1.0f;
//    auto depthLevel = cudaSimulationParameters.numCellsMeasurementToggle.value ? cudaSimulationParameters.numCellsDepthLevel.value[color] : 3;
//    GenomeDecoder::executeForEachNodeRecursively(genome, toInt(genomeSize), false, false, [&](int depth, int nodeAddress, int repetitions) {
//        auto ramificationFactor = depth > lastDepth ? numCellsRamificationFactor * toFloat(numRamifications) : 0.0f;
//        if (depth <= depthLevel) {
//            result += /* (1.0f + toFloat(depth)) * */ toFloat(repetitions) * (ramificationFactor + sizeFactor);
//        }
//        lastDepth = depth;
//        if (ramificationFactor > 0) {
//            ++numRamifications;
//        }
//    });
//
//    return result;
//}
