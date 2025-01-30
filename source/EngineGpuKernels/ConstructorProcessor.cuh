#pragma once

#include "EngineInterface/CellTypeConstants.h"

#include "SignalProcessor.cuh"
#include "SimulationCudaFacade.cuh"
#include "SimulationStatistics.cuh"
#include "CellConnectionProcessor.cuh"
#include "MutationProcessor.cuh"
#include "GenomeDecoder.cuh"
#include "CudaShapeGenerator.cuh"

class ConstructorProcessor
{
public:
    __inline__ __device__ static void preprocess(SimulationData& data, SimulationStatistics& statistics);
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    struct ConstructionData
    {
        //genome-wide data
        GenomeHeader genomeHeader;

        //node position data
        int genomeCurrentBytePosition;
        bool isLastNode;
        bool isLastNodeOfLastRepetition;
        bool hasInfiniteRepetitions;

        //node data
        float angle;
        float energy;
        int numRequiredAdditionalConnections;  // -1 = none
        int requiredNodeId1;    // -1 = none
        int requiredNodeId2;    // -1 = none
        int color;
        CellType cellType;

        //construction data
        Cell* lastConstructionCell;
        bool containsSelfReplication;
    };

    __inline__ __device__ static void completenessCheck(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static ConstructionData readConstructionData(Cell* cell);

    __inline__ __device__ static Cell* tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static Cell* getLastConstructedCell(Cell* hostCell);
    __inline__ __device__ static Cell* startNewConstruction(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);
    __inline__ __device__ static Cell* continueConstruction(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);

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
        uint64_t& cellPointerIndex,
        Cell* hostCell,
        float2 const& newCellPos,
        ConstructionData const& constructionData);

    __inline__ __device__ static bool checkAndReduceHostEnergy(SimulationData& data, Cell* hostCell, ConstructionData const& constructionData);
    __inline__ __device__ static void activateNewCell(Cell* newCell, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static bool isSelfReplicator(Cell* cell);
    __inline__ __device__ static float calcGenomeComplexity(int color, uint8_t* genome, uint16_t genomeSize);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ConstructorProcessor::preprocess(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Constructor];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        completenessCheck(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void ConstructorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellTypeOperations[CellType_Constructor];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void ConstructorProcessor::completenessCheck(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (!cudaSimulationParameters.cellTypeConstructorCheckCompletenessForSelfReplication) {
        return;
    }
    auto& constructor = cell->cellTypeData.constructor;
    if (!GenomeDecoder::isFirstNode(constructor)) {
        return;
    }
    if (!SignalProcessor::isTriggeredAndCreateSignalIfTriggered(data, cell, cell->cellTypeData.constructor.autoTriggerInterval)) {
        return;
    }

    if (constructor.numInheritedGenomeNodes == 0 || GenomeDecoder::isFinished(constructor) || !GenomeDecoder::containsSelfReplication(constructor)) {
        constructor.isReady = true;
        return;
    }

    uint32_t tagBit = 1 << toInt(cell->id % 30);
    atomicOr(&cell->tag, toInt(tagBit));
    auto actualCells = 1;

    auto constexpr QueueLength = 512;
    Cell* taggedCells[QueueLength];
    taggedCells[0] = cell;
    int numTaggedCells = 1;
    int currentTaggedCellIndex = 0;
    do {
        auto currentCell = taggedCells[currentTaggedCellIndex];

        if ((numTaggedCells + 1) % QueueLength != currentTaggedCellIndex) {
            for (int i = 0, j = currentCell->numConnections; i < j; ++i) {
                auto& nextCell = currentCell->connections[i].cell;
                if (nextCell->creatureId == cell->creatureId) {
                    auto origTagBit = static_cast<uint32_t>(atomicOr(&nextCell->tag, toInt(tagBit)));
                    if ((origTagBit & tagBit) == 0) {
                        taggedCells[numTaggedCells] = nextCell;
                        numTaggedCells = (numTaggedCells + 1) % QueueLength;
                        ++actualCells;
                    }
                }
            }
        }

        currentTaggedCellIndex = (currentTaggedCellIndex + 1) % QueueLength;
        if (currentTaggedCellIndex == numTaggedCells) {
            break;
        }
    } while (true);
    constructor.isReady = (actualCells >= constructor.numInheritedGenomeNodes);
}

__inline__ __device__ void ConstructorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    auto& constructor = cell->cellTypeData.constructor;
    if (!GenomeDecoder::isFinished(constructor)) {
        if (SignalProcessor::isTriggeredAndCreateSignalIfTriggered(data, cell, cell->cellTypeData.constructor.autoTriggerInterval)) {
            auto constructionData = readConstructionData(cell);
            if (tryConstructCell(data, statistics, cell, constructionData)) {
                cell->signal.active = true;
                cell->signal.channels[0] = 1;
                if (GenomeDecoder::isLastNode(constructor)) {
                    constructor.genomeCurrentNodeIndex = 0;
                    if (!constructionData.genomeHeader.hasInfiniteRepetitions()) {
                        ++constructor.genomeCurrentRepetition;
                        if (constructor.genomeCurrentRepetition == constructionData.genomeHeader.numRepetitions) {
                            constructor.genomeCurrentRepetition = 0;
                            if (!constructionData.genomeHeader.separateConstruction) {
                                ++constructor.genomeCurrentBranch;
                            }
                        }
                    }
                } else {
                    ++constructor.genomeCurrentNodeIndex;
                }
            } else {
                cell->signal.channels[0] = 0;
            }
        } else {
            cell->signal.active = false;
        }
    }
}

__inline__ __device__ ConstructorProcessor::ConstructionData ConstructorProcessor::readConstructionData(Cell* cell)
{
    auto& constructor = cell->cellTypeData.constructor;

    ConstructionData result;
    result.genomeHeader = GenomeDecoder::readGenomeHeader(constructor);
    result.hasInfiniteRepetitions = GenomeDecoder::hasInfiniteRepetitions(constructor);
    result.containsSelfReplication = isSelfReplicator(cell);
    auto genomeNodesPerRepetition = GenomeDecoder::getNumNodes(constructor.genome, constructor.genomeSize);
    if (!GenomeDecoder::hasInfiniteRepetitions(constructor) && constructor.genomeCurrentNodeIndex == 0 && constructor.genomeCurrentRepetition == 0) {
        result.lastConstructionCell = nullptr;
    } else {
        result.lastConstructionCell = getLastConstructedCell(cell);
    }

    if (!result.lastConstructionCell) {
        //finished => reset indices
        constructor.genomeCurrentNodeIndex = 0;
        constructor.genomeCurrentRepetition = 0;
    } else if (result.lastConstructionCell->numConnections == 1 && constructor.numInheritedGenomeNodes > 1) {
        int numConstructedCells = constructor.genomeCurrentRepetition * genomeNodesPerRepetition + constructor.genomeCurrentNodeIndex;
        if (numConstructedCells > 1) {

            //construction is broken => reset indices
            constructor.genomeCurrentNodeIndex = 0;
            constructor.genomeCurrentRepetition = 0;
        }
    }
    result.genomeCurrentBytePosition = GenomeDecoder::getNodeAddress(constructor.genome, constructor.genomeSize, constructor.genomeCurrentNodeIndex);
    result.isLastNode = GenomeDecoder::isLastNode(constructor);
    result.isLastNodeOfLastRepetition = result.isLastNode && GenomeDecoder::isLastRepetition(constructor);

    CudaShapeGenerator shapeGenerator;
    auto shape = result.genomeHeader.shape % ConstructionShape_Count;
    if (shape != ConstructionShape_Custom) {
        for (int i = 0; i <= constructor.genomeCurrentNodeIndex; ++i) {
            auto generationResult = shapeGenerator.generateNextConstructionData(shape);
            if (i == constructor.genomeCurrentNodeIndex) {
                result.numRequiredAdditionalConnections = generationResult.numRequiredAdditionalConnections;
                result.angle = generationResult.angle;
                result.genomeHeader.angleAlignment = shapeGenerator.getConstructorAngleAlignment(shape);
                result.requiredNodeId1 = generationResult.requiredNodeId1;
                result.requiredNodeId2 = generationResult.requiredNodeId2;
            }
        }
    } else {
        result.requiredNodeId1 = -1;
        result.requiredNodeId2 = -1;
    }

    result.cellType = GenomeDecoder::readByte(constructor, result.genomeCurrentBytePosition) % CellType_Count;
    auto angle = GenomeDecoder::readAngle(constructor, result.genomeCurrentBytePosition);
    result.energy = GenomeDecoder::readEnergy(constructor, result.genomeCurrentBytePosition);
    int numRequiredAdditionalConnections = GenomeDecoder::readByte(constructor, result.genomeCurrentBytePosition) % MAX_CELL_BONDS;
    result.color = GenomeDecoder::readByte(constructor, result.genomeCurrentBytePosition) % MAX_COLORS;

    if (result.genomeHeader.shape == ConstructionShape_Custom) {
        result.angle = angle;
        result.numRequiredAdditionalConnections = numRequiredAdditionalConnections;
    }

    if (genomeNodesPerRepetition == 1) {
        result.numRequiredAdditionalConnections = 0;
    }

    auto isAtFirstNode = GenomeDecoder::isFirstNode(constructor);
    if (isAtFirstNode) {
        if (GenomeDecoder::isFirstRepetition(constructor)) {
            result.angle = constructor.constructionAngle1;
        } else {
            result.angle = result.genomeHeader.concatenationAngle1;
        }
    }
    if (result.isLastNode && !isAtFirstNode) {
        if (result.isLastNodeOfLastRepetition) {
            result.angle = -constructor.constructionAngle2;
        } else {
            result.angle = result.genomeHeader.concatenationAngle2;
        }
    }
    return result;
}

__inline__ __device__ Cell*
ConstructorProcessor::tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData)
{
    if (!hostCell->tryLock()) {
        return nullptr;
    }
    if (constructionData.lastConstructionCell) {
        if (!constructionData.lastConstructionCell->tryLock()) {
            hostCell->releaseLock();
            return nullptr;
        }
        auto newCell = continueConstruction(data, statistics, hostCell, constructionData);

        constructionData.lastConstructionCell->releaseLock();
        hostCell->releaseLock();
        return newCell;
    } else {
        auto newCell = startNewConstruction(data, statistics, hostCell, constructionData);

        hostCell->releaseLock();
        return newCell;
    }
}

__inline__ __device__ Cell* ConstructorProcessor::getLastConstructedCell(Cell* hostCell)
{
    auto const& constructor = hostCell->cellTypeData.constructor;
    if (constructor.lastConstructedCellId != 0) {
        for (int i = 0; i < hostCell->numConnections; ++i) {
            auto const& connectedCell = hostCell->connections[i].cell;
            if (connectedCell->id == constructor.lastConstructedCellId) {
                return connectedCell;
            }
        }
    }

    //if lastConstructedCellId is not set (in older version or if cells got new ids)
    else {
        for (int i = 0; i < hostCell->numConnections; ++i) {
            auto const& connectedCell = hostCell->connections[i].cell;
            if (connectedCell->livingState == LivingState_UnderConstruction) {
                return connectedCell;
            }
        }
        for (int i = 0; i < hostCell->numConnections; ++i) {
            auto const& connectedCell = hostCell->connections[i].cell;
            if (connectedCell->livingState == LivingState_Dying) {
                return connectedCell;
            }
        }
    }
    return nullptr;
}

__inline__ __device__ Cell*
ConstructorProcessor::startNewConstruction(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData)
{
    auto& constructor = hostCell->cellTypeData.constructor;

    if (hostCell->numConnections == MAX_CELL_BONDS) {
        return nullptr;
    }
    auto anglesForNewConnection = CellConnectionProcessor::calcLargestGapReferenceAndActualAngle(data, hostCell, constructionData.angle);

    auto newCellDirection = Math::unitVectorOfAngle(anglesForNewConnection.actualAngle);
    float2 newCellPos = hostCell->pos + newCellDirection;

    if (CellConnectionProcessor::existCrossingConnections(
            data, hostCell->pos, newCellPos, cudaSimulationParameters.cellTypeConstructorConnectingCellMaxDistance[hostCell->color], hostCell->detached)) {
        return nullptr;
    }

    if (cudaSimulationParameters.cellTypeConstructorCheckCompletenessForSelfReplication && !constructor.isReady) {
        return nullptr;
    }

    if (!checkAndReduceHostEnergy(data, hostCell, constructionData)) {
        return nullptr;
    }

    if (constructionData.containsSelfReplication) {
        constructor.offspringCreatureId = 1 + data.numberGen1.random(65535);

        hostCell->genomeComplexity = calcGenomeComplexity(hostCell->color, constructor.genome, constructor.genomeSize);
    } else {
        constructor.offspringCreatureId = hostCell->creatureId;
    }

    uint64_t cellPointerIndex;
    Cell* newCell = constructCellIntern(data, statistics, cellPointerIndex, hostCell, newCellPos, constructionData);

    if (!newCell->tryLock()) {
        return nullptr;
    }

    if (!constructionData.isLastNodeOfLastRepetition || !constructionData.genomeHeader.separateConstruction) {
        auto distance = constructionData.isLastNodeOfLastRepetition && !constructionData.genomeHeader.separateConstruction
            ? constructionData.genomeHeader.connectionDistance
            : constructionData.genomeHeader.connectionDistance + cudaSimulationParameters.cellTypeConstructorAdditionalOffspringDistance;
        if (!CellConnectionProcessor::tryAddConnections(data, hostCell, newCell, anglesForNewConnection.referenceAngle, 0, distance)) {
            CellConnectionProcessor::scheduleDeleteCell(data, cellPointerIndex);
        }
    }
    activateNewCell(newCell, hostCell, constructionData);

    newCell->releaseLock();
    return newCell;
}

__inline__ __device__ Cell* ConstructorProcessor::continueConstruction(
    SimulationData& data,
    SimulationStatistics& statistics,
    Cell* hostCell,
    ConstructionData const& constructionData)
{
    auto const& lastCell = constructionData.lastConstructionCell;
    auto posDelta = data.cellMap.getCorrectedDirection(lastCell->pos - hostCell->pos);
    auto angleFromPreviousForNewCell = 180.0f - constructionData.angle;

    auto desiredDistance = constructionData.genomeHeader.connectionDistance;
    auto constructionSiteDistance = hostCell->getRefDistance(lastCell);
    posDelta = Math::normalized(posDelta) * (constructionSiteDistance - desiredDistance);

    if (Math::length(posDelta) <= cudaSimulationParameters.cellMinDistance
        || constructionSiteDistance - desiredDistance < cudaSimulationParameters.cellMinDistance) {
        return nullptr;
    }

    auto newCellPos = hostCell->pos + posDelta;

    Cell* cellsToConnect[MAX_CELL_BONDS];
    int numCellsToConnect;
    getCellsToConnect(cellsToConnect, numCellsToConnect, data, hostCell, newCellPos, constructionData);

    if (constructionData.numRequiredAdditionalConnections != -1) {
        if (numCellsToConnect < constructionData.numRequiredAdditionalConnections) {
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
    if (constructionData.lastConstructionCell->livingState == LivingState_Dying) {
        newCell->livingState = LivingState_Dying;
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
     
    // move connection between lastConstructionCell and hostCell to a connection between lastConstructionCell and newCell
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

    // possibly connect newCell to hostCell
    bool adaptReferenceAngle = false;
    if (!constructionData.isLastNodeOfLastRepetition || !constructionData.genomeHeader.separateConstruction) {

        auto distance = constructionData.isLastNodeOfLastRepetition && !constructionData.genomeHeader.separateConstruction
                ? constructionData.genomeHeader.connectionDistance
                : constructionData.genomeHeader.connectionDistance + 0.8f;
        if (!CellConnectionProcessor::tryAddConnections(
                data,
                newCell,
                hostCell,
                0,
                origAngleFromPreviousOnHostCell,
                distance)) {
            CellConnectionProcessor::scheduleDeleteCell(data, cellPointerIndex);
            hostCell->livingState = LivingState_Dying;
            for (int i = 0; i < hostCell->numConnections; ++i) {
                auto const& connectedCell = hostCell->connections[i].cell;
                if (connectedCell->creatureId == hostCell->creatureId) {
                    connectedCell->livingState = LivingState_Detaching;
                }
            }
        } else {
            adaptReferenceAngle = true;
        }
    }

    Math::normalize(posDelta);
    Math::rotateQuarterClockwise(posDelta);

    // get surrounding cells
    if (numCellsToConnect > 0 && constructionData.numRequiredAdditionalConnections != 0) {

        // sort surrounding cells by distance from newCell
        bubbleSort(cellsToConnect, numCellsToConnect, [&](auto const& cell1, auto const& cell2) {
            auto dist1 = data.cellMap.getDistance(cell1->pos, newCellPos);
            auto dist2 = data.cellMap.getDistance(cell2->pos, newCellPos);
            return dist1 < dist2;
        });

        // connect surrounding cells if possible
        int numConnectedCells = 0;
        for (int i = 0; i < numCellsToConnect; ++i) {
            Cell* otherCell = cellsToConnect[i];

            if (otherCell->tryLock()) {
                if (newCell->numConnections < MAX_CELL_BONDS && otherCell->numConnections < MAX_CELL_BONDS) {
                    if (CellConnectionProcessor::tryAddConnections(data, newCell, otherCell, 0, 0, desiredDistance, constructionData.genomeHeader.angleAlignment)) {
                        ++numConnectedCells; 
                    }
                }
                otherCell->releaseLock();
            }
            if (constructionData.numRequiredAdditionalConnections != -1) {
                if (numConnectedCells == constructionData.numRequiredAdditionalConnections) {
                    break;
                }
            }
        }
    }

    // adapt angles according to genome
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

    activateNewCell(newCell, hostCell, constructionData);

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

    if (constructionData.numRequiredAdditionalConnections == 0) {
        return;
    }

    Cell* nearCells[MAX_CELL_BONDS * 4];
    int numNearCells;
    data.cellMap.getMatchingCells(
        nearCells,
        MAX_CELL_BONDS * 4,
        numNearCells,
        newCellPos,
        cudaSimulationParameters.cellTypeConstructorConnectingCellMaxDistance[hostCell->color],
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
            cudaSimulationParameters.cellTypeConstructorConnectingCellMaxDistance[hostCell->color],
            hostCell->detached,
            [&](Cell* const& otherCell) {
                if (otherCell == constructionData.lastConstructionCell || otherCell == hostCell
                    || (otherCell->livingState != LivingState_UnderConstruction && otherCell->activationTime == 0)
                    || otherCell->creatureId != hostCell->cellTypeData.constructor.offspringCreatureId) {
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
            cudaSimulationParameters.cellTypeConstructorConnectingCellMaxDistance[hostCell->color],
            hostCell->detached,
            [&](Cell* const& otherCell) {
                if (otherCell->livingState != LivingState_UnderConstruction
                    || otherCell->creatureId != hostCell->cellTypeData.constructor.offspringCreatureId) {
                    return false;
                }
                if (constructionData.numRequiredAdditionalConnections >= 1 && otherCell->genomeNodeIndex == constructionData.requiredNodeId1) {
                    return true;
                }
                if (constructionData.numRequiredAdditionalConnections == 2 && otherCell->genomeNodeIndex == constructionData.requiredNodeId2) {
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
                    crossingLinks = true;
                }
            }
            if (!crossingLinks) {
                auto delta = data.cellMap.getCorrectedDirection(newCellPos - otherCell->pos);
                if (CellConnectionProcessor::hasAngleSpace(data, otherCell, Math::angleOfVector(delta), constructionData.genomeHeader.angleAlignment)) {
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
    uint64_t& cellPointerIndex,
    Cell* hostCell,
    float2 const& posOfNewCell,
    ConstructionData const& constructionData)
{
    auto& constructor = hostCell->cellTypeData.constructor;

    ObjectFactory factory;
    factory.init(&data);

    Cell* result = factory.createCell(cellPointerIndex);
    constructor.lastConstructedCellId = result->id;
    result->energy = constructionData.energy;
    result->stiffness = constructionData.genomeHeader.stiffness;
    result->pos = posOfNewCell;
    data.cellMap.correctPosition(result->pos);
    result->numConnections = 0;
    result->livingState = LivingState_UnderConstruction;
    result->creatureId = constructor.offspringCreatureId;
    result->mutationId = constructor.offspringMutationId;
    result->ancestorMutationId = static_cast<uint8_t>(hostCell->mutationId & 0xff);
    result->cellType = constructionData.cellType;
    result->color = constructionData.color;
    result->absAngleToConnection0 = 0;

    result->activationTime = constructionData.containsSelfReplication ? constructor.constructionActivationTime : 0;
    result->genomeComplexity = hostCell->genomeComplexity;
    result->genomeNodeIndex = constructor.genomeCurrentNodeIndex;

    auto genomeCurrentBytePosition = constructionData.genomeCurrentBytePosition;

    result->signalRoutingRestriction.active = GenomeDecoder::readBool(constructor, genomeCurrentBytePosition);
    result->signalRoutingRestriction.baseAngle = GenomeDecoder::readAngle(constructor, genomeCurrentBytePosition);
    result->signalRoutingRestriction.openingAngle = GenomeDecoder::readAngle(constructor, genomeCurrentBytePosition);

    result->neuralNetwork = data.objects.auxiliaryData.getTypedSubArray<NeuralNetwork>(1);
    for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
        result->neuralNetwork->weights[i] = GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition) * 4;
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result->neuralNetwork->biases[i] = GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition) * 4;
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        result->neuralNetwork->activationFunctions[i] = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % ActivationFunction_Count;
    }
    switch (constructionData.cellType) {
    case CellType_Base: {
    } break;
    case CellType_Depot: {
        result->cellTypeData.transmitter.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % EnergyDistributionMode_Count;
    } break;
    case CellType_Constructor: {
        auto& newConstructor = result->cellTypeData.constructor;
        newConstructor.autoTriggerInterval = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
        newConstructor.constructionActivationTime = GenomeDecoder::readWord(constructor, genomeCurrentBytePosition) % MAX_ACTIVATION_TIME;
        newConstructor.lastConstructedCellId = 0;
        newConstructor.genomeCurrentBranch = 0;
        newConstructor.genomeCurrentNodeIndex = 0;
        newConstructor.genomeCurrentRepetition = 0;
        newConstructor.constructionAngle1 = GenomeDecoder::readAngle(constructor, genomeCurrentBytePosition);
        newConstructor.constructionAngle2 = GenomeDecoder::readAngle(constructor, genomeCurrentBytePosition);
        GenomeDecoder::copyGenome(data, constructor, genomeCurrentBytePosition, newConstructor);
        auto numInheritedGenomeNodes = 
            GenomeDecoder::getNumNodesRecursively(newConstructor.genome, newConstructor.genomeSize, true, false);
        newConstructor.numInheritedGenomeNodes = static_cast<uint16_t>(min(NPP_MAX_16U, numInheritedGenomeNodes));
        newConstructor.genomeGeneration = constructor.genomeGeneration + 1;
        newConstructor.offspringMutationId = constructor.offspringMutationId;
        if (GenomeDecoder::containsSelfReplication(newConstructor)) {
            statistics.incNumCreatedReplicators(hostCell->color);
        }
        newConstructor.isReady = true;
    } break;
    case CellType_Sensor: {
        result->cellTypeData.sensor.autoTriggerInterval = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
        result->cellTypeData.sensor.minDensity = (GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition) + 1.0f) / 2;
        result->cellTypeData.sensor.restrictToColor = GenomeDecoder::readOptionalByte(constructor, genomeCurrentBytePosition, MAX_COLORS);
        result->cellTypeData.sensor.restrictToMutants =
            GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % SensorRestrictToMutants_Count;
        result->cellTypeData.sensor.minRange = GenomeDecoder::readOptionalByte(constructor, genomeCurrentBytePosition);
        result->cellTypeData.sensor.maxRange = GenomeDecoder::readOptionalByte(constructor, genomeCurrentBytePosition);
    } break;
    case CellType_Oscillator: {
        result->cellTypeData.oscillator.autoTriggerInterval = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
        result->cellTypeData.oscillator.alternationInterval = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
        result->cellTypeData.oscillator.numPulses = 0;
    } break;
    case CellType_Attacker: {
        result->cellTypeData.attacker.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % EnergyDistributionMode_Count;
    } break;
    case CellType_Injector: {
        result->cellTypeData.injector.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % InjectorMode_Count;
        result->cellTypeData.injector.counter = 0;
        GenomeDecoder::copyGenome(data, constructor, genomeCurrentBytePosition, result->cellTypeData.injector);
        result->cellTypeData.injector.genomeGeneration = constructor.genomeGeneration + 1;
    } break;
    case CellType_Muscle: {
        result->cellTypeData.muscle.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % MuscleMode_Count;
        if (result->cellTypeData.muscle.mode == MuscleMode_Bending) {
            result->cellTypeData.muscle.modeData.bending.maxAngleDeviation = abs(GenomeDecoder::readAngle(constructor, genomeCurrentBytePosition));
            result->cellTypeData.muscle.modeData.bending.frontBackVelRatio = abs(GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition));
            result->cellTypeData.muscle.modeData.bending.initialAngle = 0;
            result->cellTypeData.muscle.modeData.bending.lastAngle = 0;
            result->cellTypeData.muscle.modeData.bending.forward = true;
            result->cellTypeData.muscle.modeData.bending.activation = 0;
            result->cellTypeData.muscle.modeData.bending.activationCountdown = 0;
            result->cellTypeData.muscle.modeData.bending.impulseAlreadyApplied = false;
        }
        result->cellTypeData.muscle.frontAngle = constructionData.genomeHeader.frontAngle;
        result->cellTypeData.muscle.lastMovementX = 0;
        result->cellTypeData.muscle.lastMovementY = 0;
    } break;
    case CellType_Defender: {
        result->cellTypeData.defender.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % DefenderMode_Count;
    } break;
    case CellType_Reconnector: {
        result->cellTypeData.reconnector.restrictToColor = GenomeDecoder::readOptionalByte(constructor, genomeCurrentBytePosition, MAX_COLORS);
        result->cellTypeData.reconnector.restrictToMutants =
            GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % ReconnectorRestrictToMutants_Count;
    } break;
    case CellType_Detonator: {
        result->cellTypeData.detonator.state = DetonatorState_Ready;
        result->cellTypeData.detonator.countdown = GenomeDecoder::readWord(constructor, genomeCurrentBytePosition);
    } break;
    }

    statistics.incNumCreatedCells(hostCell->color);
    return result;
}

__inline__ __device__ bool ConstructorProcessor::checkAndReduceHostEnergy(SimulationData& data, Cell* hostCell, ConstructionData const& constructionData)
{
    if (cudaSimulationParameters.features.externalEnergyControl && hostCell->energy < constructionData.energy + cudaSimulationParameters.cellNormalEnergy[hostCell->color]
        && cudaSimulationParameters.externalEnergyInflowFactor[hostCell->color] > 0) {
        auto externalEnergyPortion = [&] {
            if (cudaSimulationParameters.externalEnergyInflowOnlyForNonSelfReplicators) {
                return !constructionData.containsSelfReplication && !GenomeDecoder::isFinished(hostCell->cellTypeData.constructor)
                    ? constructionData.energy * cudaSimulationParameters.externalEnergyInflowFactor[hostCell->color]
                    : 0.0f;
            } else {
                return constructionData.energy * cudaSimulationParameters.externalEnergyInflowFactor[hostCell->color];
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
        if (!cudaSimulationParameters.features.externalEnergyControl) {
            return 0.0f;
        }
        if (cudaSimulationParameters.externalEnergyInflowOnlyForNonSelfReplicators) {
            return !constructionData.containsSelfReplication ? cudaSimulationParameters.externalEnergyConditionalInflowFactor[hostCell->color] : 0.0f;
        } else {
            return cudaSimulationParameters.externalEnergyConditionalInflowFactor[hostCell->color];
        }
    }();

    auto energyNeededFromHost = max(0.0f, constructionData.energy - cudaSimulationParameters.cellNormalEnergy[hostCell->color])
        + min(constructionData.energy, cudaSimulationParameters.cellNormalEnergy[hostCell->color]) * (1.0f - externalEnergyConditionalInflowFactor);

    if (externalEnergyConditionalInflowFactor < 1.0f && hostCell->energy < cudaSimulationParameters.cellNormalEnergy[hostCell->color] + energyNeededFromHost) {
        return false;
    }
    auto energyNeededFromExternalSource = constructionData.energy - energyNeededFromHost;
    auto orig = atomicAdd(data.externalEnergy, -energyNeededFromExternalSource);
    if (orig < energyNeededFromExternalSource) {
        atomicAdd(data.externalEnergy, energyNeededFromExternalSource);
        if (hostCell->energy < cudaSimulationParameters.cellNormalEnergy[hostCell->color] + constructionData.energy) {
            return false;
        }
        hostCell->energy -= constructionData.energy;
    } else {
        hostCell->energy -= energyNeededFromHost;
    }
    return true;
}

__inline__ __device__ void ConstructorProcessor::activateNewCell(Cell* newCell, Cell* hostCell, ConstructionData const& constructionData)
{
    if (constructionData.isLastNodeOfLastRepetition || (constructionData.isLastNode && constructionData.hasInfiniteRepetitions)) {
        newCell->livingState = LivingState_Activating;
        if (!constructionData.genomeHeader.separateConstruction) {
            if (hostCell->numConnections > 1) {
                newCell->absAngleToConnection0 =
                    Math::normalizedAngle(hostCell->absAngleToConnection0 + hostCell->getAngelSpan(hostCell->connections[0].cell, newCell), -180.0f);
            }
            if (newCell->numConnections > 1) {
                newCell->absAngleToConnection0 =
                    Math::normalizedAngle(
                    newCell->absAngleToConnection0 - (180.0f - newCell->getAngelSpan(hostCell, newCell->connections[0].cell)), -180.0f);
            }
        }
    }
}

__inline__ __device__ bool ConstructorProcessor::isSelfReplicator(Cell* cell)
{
    if (cell->cellType != CellType_Constructor) {
        return false;
    }
    return GenomeDecoder::containsSelfReplication(cell->cellTypeData.constructor);
}

__inline__ __device__ float ConstructorProcessor::calcGenomeComplexity(int color, uint8_t* genome, uint16_t genomeSize)
{
    auto result = 0.0f;

    auto lastDepth = 0;
    auto numRamifications = 1;
    auto genomeComplexityRamificationFactor =
        cudaSimulationParameters.features.genomeComplexityMeasurement ? cudaSimulationParameters.genomeComplexityRamificationFactor[color] : 0.0f;
    auto sizeFactor =
        cudaSimulationParameters.features.genomeComplexityMeasurement ? cudaSimulationParameters.genomeComplexitySizeFactor[color] : 1.0f;
    auto depthLevel = cudaSimulationParameters.features.genomeComplexityMeasurement ? cudaSimulationParameters.genomeComplexityDepthLevel[color] : 3;
    GenomeDecoder::executeForEachNodeRecursively(genome, toInt(genomeSize), false, false, [&](int depth, int nodeAddress, int repetitions) {
        auto ramificationFactor = depth > lastDepth ? genomeComplexityRamificationFactor * toFloat(numRamifications) : 0.0f;
        auto cellType = GenomeDecoder::getNextCellType(genome, nodeAddress);
        auto neuronFactor = cellType == CellType_Base ? cudaSimulationParameters.genomeComplexityNeuronFactor[color] : 0.0f;
        if (depth <= depthLevel) {
            result += /* (1.0f + toFloat(depth)) * */ toFloat(repetitions) * (ramificationFactor + sizeFactor + neuronFactor);
        }
        lastDepth = depth;
        if (ramificationFactor > 0) {
            ++numRamifications;
        }
    });

    return result;
}
