#pragma once

#include "EngineInterface/CellFunctionConstants.h"

#include "CellFunctionProcessor.cuh"
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
        int numRequiredAdditionalConnections;
        int executionOrderNumber;
        int color;
        int inputExecutionOrderNumber;
        bool outputBlocked;
        CellFunction cellFunction;

        //construction data
        Cell* lastConstructionCell;
    };

    __inline__ __device__ static void completenessCheck(SimulationData& data, SimulationStatistics& statistics, Cell* cell);

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static ConstructionData readConstructionData(Cell* cell);
    __inline__ __device__ static bool isConstructionTriggered(SimulationData const& data, Cell* cell, Activity const& activity);

    __inline__ __device__ static bool tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static Cell* getLastConstructedCell(Cell* hostCell);
    __inline__ __device__ static bool startNewConstruction(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);
    __inline__ __device__ static bool continueConstruction(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static bool isConnectable(int numConnections, int maxConnections, bool adaptMaxConnections);

    __inline__ __device__ static Cell* constructCellIntern(
        SimulationData& data,
        uint64_t& cellPointerIndex,
        Cell* hostCell,
        float2 const& newCellPos,
        int maxConnections,
        ConstructionData const& constructionData);

    __inline__ __device__ static bool checkAndReduceHostEnergy(SimulationData& data, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static bool isSelfReplicator(Cell* cell);
    __inline__ __device__ static int calcGenomeComplexity(int color, uint8_t* genome, uint16_t genomeSize);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void ConstructorProcessor::preprocess(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Constructor];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        completenessCheck(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void ConstructorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Constructor];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void ConstructorProcessor::completenessCheck(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    if (!cudaSimulationParameters.cellFunctionConstructorCheckCompletenessForSelfReplication) {
        return;
    }
    auto& constructor = cell->cellFunctionData.constructor;
    if (!GenomeDecoder::isFirstNode(constructor)) {
        return;
    }
    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    if (!isConstructionTriggered(data, cell, activity)) {
        return;
    }

    if (constructor.numInheritedGenomeNodes == 0 || GenomeDecoder::isFinished(constructor) || !GenomeDecoder::containsSelfReplication(constructor)) {
        constructor.isComplete = true;
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
    constructor.isComplete = (actualCells >= constructor.numInheritedGenomeNodes);
}

__inline__ __device__ void ConstructorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    MutationProcessor::applyRandomMutation(data, cell);

    auto& constructor = cell->cellFunctionData.constructor;
    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    if (!GenomeDecoder::isFinished(constructor)) {
        auto constructionData = readConstructionData(cell);
        auto cellBuilt = false;
        if (isConstructionTriggered(data, cell, activity)) {
            if (tryConstructCell(data, statistics, cell, constructionData)) {
                cellBuilt = true;
            } 
        }

        if (cellBuilt) {
            activity.channels[0] = 1;
            if (GenomeDecoder::isLastNode(constructor)) {
                constructor.genomeCurrentNodeIndex = 0;
                if (!constructionData.genomeHeader.hasInfiniteRepetitions()) {
                    ++constructor.genomeCurrentRepetition;
                    if (constructor.genomeCurrentRepetition == constructionData.genomeHeader.numRepetitions) {
                        constructor.genomeCurrentRepetition = 0;
                        if (!constructionData.genomeHeader.separateConstruction) {
                            ++constructor.currentBranch;
                        }
                    }
                }
            } else {
                ++constructor.genomeCurrentNodeIndex;
            }
        } else {
            activity.channels[0] = 0;
        }
    }
    CellFunctionProcessor::setActivity(cell, activity);
}

__inline__ __device__ ConstructorProcessor::ConstructionData ConstructorProcessor::readConstructionData(Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;

    ConstructionData result;
    result.genomeHeader = GenomeDecoder::readGenomeHeader(constructor);
    result.hasInfiniteRepetitions = GenomeDecoder::hasInfiniteRepetitions(constructor);
    if (!GenomeDecoder::hasInfiniteRepetitions(constructor) && constructor.genomeCurrentNodeIndex == 0 && constructor.genomeCurrentRepetition == 0) {
        result.lastConstructionCell = nullptr;
    } else {
        result.lastConstructionCell = getLastConstructedCell(cell);
    }

    if (!result.lastConstructionCell) {
        //finished => reset indices
        constructor.genomeCurrentNodeIndex = 0;
        constructor.genomeCurrentRepetition = 0;
    } else if (result.lastConstructionCell->numConnections == 1) {
        int numConstructedCells = constructor.genomeCurrentRepetition * result.genomeHeader.numRepetitions + constructor.genomeCurrentNodeIndex;
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
            }
        }
    }

    result.cellFunction = GenomeDecoder::readByte(constructor, result.genomeCurrentBytePosition) % CellFunction_Count;
    auto angle = GenomeDecoder::readAngle(constructor, result.genomeCurrentBytePosition);
    result.energy = GenomeDecoder::readEnergy(constructor, result.genomeCurrentBytePosition);
    int numRequiredAdditionalConnections = GenomeDecoder::readByte(constructor, result.genomeCurrentBytePosition);
    numRequiredAdditionalConnections = numRequiredAdditionalConnections > 127 ? -1 : numRequiredAdditionalConnections % (MAX_CELL_BONDS + 1);
    result.executionOrderNumber =
        GenomeDecoder::readByte(constructor, result.genomeCurrentBytePosition) % cudaSimulationParameters.cellNumExecutionOrderNumbers;
    result.color = GenomeDecoder::readByte(constructor, result.genomeCurrentBytePosition) % MAX_COLORS;
    result.inputExecutionOrderNumber =
        GenomeDecoder::readOptionalByte(constructor, result.genomeCurrentBytePosition, cudaSimulationParameters.cellNumExecutionOrderNumbers);
    result.outputBlocked = GenomeDecoder::readBool(constructor, result.genomeCurrentBytePosition);

    if (result.genomeHeader.shape == ConstructionShape_Custom) {
        result.angle = angle;
        result.numRequiredAdditionalConnections = numRequiredAdditionalConnections;
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
            result.angle = constructor.constructionAngle2;
        } else {
            result.angle = result.genomeHeader.concatenationAngle2;
        }
    }
    return result;
}

__inline__ __device__ bool
ConstructorProcessor::isConstructionTriggered(SimulationData const& data, Cell* cell, Activity const& activity)
{
    if (cell->cellFunctionData.constructor.activationMode == 0
        && abs(activity.channels[0]) < cudaSimulationParameters.cellFunctionConstructorActivityThreshold[cell->color]) {
        return false;
    }
    if (cell->cellFunctionData.constructor.activationMode > 0
        && (data.timestep % (cudaSimulationParameters.cellNumExecutionOrderNumbers * cell->cellFunctionData.constructor.activationMode) != cell->executionOrderNumber)) {
        return false;
    }
    return true;
}

__inline__ __device__ bool
ConstructorProcessor::tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData)
{
    if (!hostCell->tryLock()) {
        return false;
    }
    if (constructionData.lastConstructionCell) {
        if (!constructionData.lastConstructionCell->tryLock()) {
            hostCell->releaseLock();
            return false;
        }
        auto success = continueConstruction(data, statistics, hostCell, constructionData);

        constructionData.lastConstructionCell->releaseLock();
        hostCell->releaseLock();
        return success;
    } else {
        auto success = startNewConstruction(data, statistics, hostCell, constructionData);

        hostCell->releaseLock();
        return success;
    }
}

__inline__ __device__ Cell* ConstructorProcessor::getLastConstructedCell(Cell* hostCell)
{
    auto const& constructor = hostCell->cellFunctionData.constructor;
    if (constructor.lastConstructedCellId != 0) {
        for (int i = 0; i < hostCell->numConnections; ++i) {
            auto const& connectedCell = hostCell->connections[i].cell;
            if (connectedCell->id == constructor.lastConstructedCellId) {
                return connectedCell;
            }
        }
    }

    //for compatibility with older simulations in case that lastConstructedCellId is not set yet
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

__inline__ __device__ bool
ConstructorProcessor::startNewConstruction(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData)
{
    auto& constructor = hostCell->cellFunctionData.constructor;

    if (!isConnectable(hostCell->numConnections, hostCell->maxConnections, true)) {
        return false;
    }
    auto anglesForNewConnection = CellFunctionProcessor::calcLargestGapReferenceAndActualAngle(data, hostCell, constructionData.angle);

    auto newCellDirection = Math::unitVectorOfAngle(anglesForNewConnection.actualAngle);
    float2 newCellPos = hostCell->pos + newCellDirection;

    if (CellConnectionProcessor::existCrossingConnections(data, hostCell->pos, newCellPos, hostCell->detached)) {
        return false;
    }

    if (cudaSimulationParameters.cellFunctionConstructorCheckCompletenessForSelfReplication && !constructor.isComplete) {
        return false;
    }

    if (!checkAndReduceHostEnergy(data, hostCell, constructionData)) {
        return false;
    }

    if (GenomeDecoder::containsSelfReplication(constructor)) {
        constructor.offspringCreatureId = 1 + data.numberGen1.random(65535);

        hostCell->genomeComplexity = calcGenomeComplexity(hostCell->color, constructor.genome, constructor.genomeSize);
    } else {
        constructor.offspringCreatureId = hostCell->creatureId;
    }

    uint64_t cellPointerIndex;
    Cell* newCell = constructCellIntern(data, cellPointerIndex, hostCell, newCellPos, 0, constructionData);

    if (!newCell->tryLock()) {
        return false;
    }

    if (!constructionData.isLastNodeOfLastRepetition || !constructionData.genomeHeader.separateConstruction) {
        auto distance = constructionData.isLastNodeOfLastRepetition && !constructionData.genomeHeader.separateConstruction
            ? 1.0f
            : cudaSimulationParameters.cellFunctionConstructorOffspringDistance[hostCell->color];
        if(!CellConnectionProcessor::tryAddConnections(data, hostCell, newCell, anglesForNewConnection.referenceAngle, 0, distance)) {
            CellConnectionProcessor::scheduleDeleteCell(data, cellPointerIndex);
        }
    }
    if (constructionData.isLastNodeOfLastRepetition || (constructionData.isLastNode && constructionData.hasInfiniteRepetitions)) {
        newCell->livingState = LivingState_Activating;
    }
    hostCell->maxConnections = max(hostCell->numConnections, hostCell->maxConnections);
    newCell->maxConnections = max(newCell->numConnections, newCell->maxConnections);

    newCell->releaseLock();

    statistics.incNumCreatedCells(hostCell->color);
    return true;
}

__inline__ __device__ bool ConstructorProcessor::continueConstruction(
    SimulationData& data,
    SimulationStatistics& statistics,
    Cell* hostCell,
    ConstructionData const& constructionData)
{
    auto posDelta = constructionData.lastConstructionCell->pos - hostCell->pos;
    data.cellMap.correctDirection(posDelta);

    auto desiredDistance = constructionData.genomeHeader.connectionDistance;
    auto constructionSiteDistance = data.cellMap.getDistance(hostCell->pos, constructionData.lastConstructionCell->pos);
    posDelta = Math::normalized(posDelta) * (constructionSiteDistance - desiredDistance);

    if (Math::length(posDelta) <= cudaSimulationParameters.cellMinDistance
        || constructionSiteDistance - desiredDistance < cudaSimulationParameters.cellMinDistance) {
        return false;
    }

    auto newCellPos = hostCell->pos + posDelta;

    //get surrounding cells
    Cell* otherCellCandidates[MAX_CELL_BONDS * 2];
    int numOtherCellCandidates = 0;
    data.cellMap.getMatchingCells(
        otherCellCandidates,
        MAX_CELL_BONDS * 2,
        numOtherCellCandidates,
        newCellPos,
        cudaSimulationParameters.cellFunctionConstructorConnectingCellMaxDistance[hostCell->color],
        hostCell->detached,
        [&](Cell* const& otherCell) {
            if (otherCell == constructionData.lastConstructionCell || otherCell == hostCell
                || (otherCell->livingState != LivingState_UnderConstruction
                && otherCell->activationTime == 0) || otherCell->creatureId != hostCell->cellFunctionData.constructor.offspringCreatureId) {
                return false;
            }
            return true;
        });

    //assemble surrounding cell candidates
    Cell* otherCells[MAX_CELL_BONDS];
    int numOtherCells = 0;
    for (int i = 0; i < numOtherCellCandidates; ++i) {
        Cell* otherCell = otherCellCandidates[i];
        if (otherCell->tryLock()) {
            if (!CellConnectionProcessor::wouldResultInOverlappingConnection(otherCell, newCellPos)) {
                otherCells[numOtherCells++] = otherCell;
            }
            otherCell->releaseLock();
        }
        if (numOtherCells == MAX_CELL_BONDS) {
            break;
        }
    }
    if (constructionData.numRequiredAdditionalConnections != -1) {
        if (numOtherCells < constructionData.numRequiredAdditionalConnections) {
            return false;
        }
    }

    if (!checkAndReduceHostEnergy(data, hostCell, constructionData)) {
        return false;
    }
    uint64_t cellPointerIndex;
    Cell* newCell = constructCellIntern(data, cellPointerIndex, hostCell, newCellPos, 0, constructionData);

    if (!newCell->tryLock()) {
        return false;
    }
    if (constructionData.lastConstructionCell->livingState == LivingState_Dying) {
        newCell->livingState = LivingState_Dying;
    }

    if (constructionData.isLastNodeOfLastRepetition || (constructionData.isLastNode && constructionData.hasInfiniteRepetitions)) {
        newCell->livingState = LivingState_Activating;
    }

    float angleFromPreviousForUnderConstructionCell;
    for (int i = 0; i < constructionData.lastConstructionCell->numConnections; ++i) {
        if (constructionData.lastConstructionCell->connections[i].cell == hostCell) {
            angleFromPreviousForUnderConstructionCell = constructionData.lastConstructionCell->connections[i].angleFromPrevious;
            break;
        }
    }

    //possibly connect newCell to hostCell
    bool adaptReferenceAngle = false;
    if (!constructionData.isLastNodeOfLastRepetition || !constructionData.genomeHeader.separateConstruction) {

        //move connection
        auto distance = constructionData.isLastNodeOfLastRepetition && !constructionData.genomeHeader.separateConstruction
            ? 1.0f
            : cudaSimulationParameters.cellFunctionConstructorOffspringDistance[hostCell->color];
        for (int i = 0; i < hostCell->numConnections; ++i) {
            auto& connectedCell = hostCell->connections[i];
            if (connectedCell.cell == constructionData.lastConstructionCell) {
                connectedCell.cell = newCell;
                connectedCell.distance = distance;
                newCell->numConnections = 1;
                newCell->connections[0].cell = hostCell;
                newCell->connections[0].distance = distance;
                newCell->connections[0].angleFromPrevious = 360.0f;
                adaptReferenceAngle = true;
                CellConnectionProcessor::deleteConnectionOneWay(constructionData.lastConstructionCell, hostCell);
                break;
            }
        }
    } else {

        //cut connections
        CellConnectionProcessor::deleteConnections(hostCell, constructionData.lastConstructionCell);
    }

    //connect newCell to lastConstructionCell
    auto angleFromPreviousForNewCell = 180.0f - constructionData.angle;
    if (!CellConnectionProcessor::tryAddConnections(
            data,
            newCell,
            constructionData.lastConstructionCell,
            /*angleFromPreviousForNewCell*/ 0,
            angleFromPreviousForUnderConstructionCell,
            desiredDistance)) {
        adaptReferenceAngle = false;
        CellConnectionProcessor::scheduleDeleteCell(data, cellPointerIndex);
        hostCell->livingState = LivingState_Dying;
    }

    Math::normalize(posDelta);
    Math::rotateQuarterClockwise(posDelta);

    //get surrounding cells
    if (numOtherCells > 0) {

        //sort surrounding cells by distance from newCell
        bubbleSort(otherCells, numOtherCells, [&](auto const& cell1, auto const& cell2) {
            auto dist1 = data.cellMap.getDistance(cell1->pos, newCellPos);
            auto dist2 = data.cellMap.getDistance(cell2->pos, newCellPos);
            return dist1 < dist2;
        });

        if (constructionData.numRequiredAdditionalConnections != -1) {
            //it is already ensured that numOtherCells is not less than constructionData.numRequiredAdditionalConnections
            numOtherCells = constructionData.numRequiredAdditionalConnections;
        }

        //connect surrounding cells if possible
        for (int i = 0; i < numOtherCells; ++i) {
            Cell* otherCell = otherCells[i];

            if (otherCell->tryLock()) {
                if (isConnectable(newCell->numConnections, newCell->maxConnections, true)
                    && isConnectable(otherCell->numConnections, otherCell->maxConnections, true)) {

                    CellConnectionProcessor::tryAddConnections(data, newCell, otherCell, 0, 0, desiredDistance, constructionData.genomeHeader.angleAlignment);
                    otherCell->maxConnections = max(otherCell->numConnections, otherCell->maxConnections);
                }
                otherCell->releaseLock();
            }
        }
    }

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
    hostCell->maxConnections = max(hostCell->numConnections, hostCell->maxConnections);
    newCell->maxConnections = max(newCell->numConnections, newCell->maxConnections);

    newCell->releaseLock();

    statistics.incNumCreatedCells(hostCell->color);
    return true;
}

__inline__ __device__ bool ConstructorProcessor::isConnectable(int numConnections, int maxConnections, bool adaptMaxConnections)
{
    if (!adaptMaxConnections) {
        if (numConnections >= maxConnections) {
            return false;
        }
    } else {
        if (numConnections >= MAX_CELL_BONDS) {
            return false;
        }
    }
    return true;
}

__inline__ __device__ Cell*
ConstructorProcessor::constructCellIntern(
    SimulationData& data,
    uint64_t& cellPointerIndex,
    Cell* hostCell,
    float2 const& posOfNewCell,
    int maxConnections,
    ConstructionData const& constructionData)
{
    auto& constructor = hostCell->cellFunctionData.constructor;

    ObjectFactory factory;
    factory.init(&data);

    Cell* result = factory.createCell(cellPointerIndex);
    constructor.lastConstructedCellId = result->id;
    result->energy = constructionData.energy;
    result->stiffness = constructionData.genomeHeader.stiffness;
    result->pos = posOfNewCell;
    data.cellMap.correctPosition(result->pos);
    result->maxConnections = maxConnections;
    result->numConnections = 0;
    result->executionOrderNumber = constructionData.executionOrderNumber;
    result->livingState = LivingState_UnderConstruction;
    result->creatureId = constructor.offspringCreatureId;
    result->mutationId = constructor.offspringMutationId;
    result->cellFunction = constructionData.cellFunction;
    result->color = constructionData.color;
    result->inputExecutionOrderNumber = constructionData.inputExecutionOrderNumber;
    result->outputBlocked = constructionData.outputBlocked;

    result->activationTime = constructor.constructionActivationTime;
    result->genomeComplexity = hostCell->genomeComplexity;

    auto genomeCurrentBytePosition = constructionData.genomeCurrentBytePosition;
    switch (constructionData.cellFunction) {
    case CellFunction_Neuron: {
        result->cellFunctionData.neuron.neuronState = data.objects.auxiliaryData.getTypedSubArray<NeuronFunction::NeuronState>(1);
        for (int i = 0; i < MAX_CHANNELS *  MAX_CHANNELS; ++i) {
            result->cellFunctionData.neuron.neuronState->weights[i] = GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition) * 4;
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result->cellFunctionData.neuron.neuronState->biases[i] = GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition) * 4;
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result->cellFunctionData.neuron.activationFunctions[i] =
                GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % NeuronActivationFunction_Count;
        }
    } break;
    case CellFunction_Transmitter: {
        result->cellFunctionData.transmitter.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % EnergyDistributionMode_Count;
    } break;
    case CellFunction_Constructor: {
        auto& newConstructor = result->cellFunctionData.constructor;
        newConstructor.activationMode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
        newConstructor.constructionActivationTime = GenomeDecoder::readWord(constructor, genomeCurrentBytePosition) % MaxActivationTime;
        newConstructor.lastConstructedCellId = 0;
        newConstructor.currentBranch = 0;
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
    } break;
    case CellFunction_Sensor: {
        result->cellFunctionData.sensor.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % SensorMode_Count;
        result->cellFunctionData.sensor.angle = GenomeDecoder::readAngle(constructor, genomeCurrentBytePosition);
        result->cellFunctionData.sensor.minDensity = (GenomeDecoder::readFloat(constructor, genomeCurrentBytePosition) + 1.0f) / 2;
        result->cellFunctionData.sensor.restrictToColor = GenomeDecoder::readOptionalByte(constructor, genomeCurrentBytePosition, MAX_COLORS);
        result->cellFunctionData.sensor.restrictToOtherMutants = GenomeDecoder::readBool(constructor, genomeCurrentBytePosition);
        result->cellFunctionData.sensor.memoryChannel1 = 0;
        result->cellFunctionData.sensor.memoryChannel2 = 0;
        result->cellFunctionData.sensor.memoryChannel3 = 0;

    } break;
    case CellFunction_Nerve: {
        result->cellFunctionData.nerve.pulseMode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
        result->cellFunctionData.nerve.alternationMode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition);
    } break;
    case CellFunction_Attacker: {
        result->cellFunctionData.attacker.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % EnergyDistributionMode_Count;
    } break;
    case CellFunction_Injector: {
        result->cellFunctionData.injector.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % InjectorMode_Count;
        result->cellFunctionData.injector.counter = 0;
        GenomeDecoder::copyGenome(data, constructor, genomeCurrentBytePosition, result->cellFunctionData.injector);
        result->cellFunctionData.injector.genomeGeneration = constructor.genomeGeneration + 1;
    } break;
    case CellFunction_Muscle: {
        result->cellFunctionData.muscle.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % MuscleMode_Count;
        result->cellFunctionData.muscle.lastBendingDirection = MuscleBendingDirection_None;
        result->cellFunctionData.muscle.consecutiveBendingAngle = 0;
    } break;
    case CellFunction_Defender: {
        result->cellFunctionData.defender.mode = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % DefenderMode_Count;
    } break;
    case CellFunction_Reconnector: {
        result->cellFunctionData.reconnector.color = GenomeDecoder::readByte(constructor, genomeCurrentBytePosition) % MAX_COLORS;
    } break;
    case CellFunction_Detonator: {
        result->cellFunctionData.detonator.state = DetonatorState_Ready;
        result->cellFunctionData.detonator.countdown = GenomeDecoder::readWord(constructor, genomeCurrentBytePosition);
    } break;
    }

    return result;
}

__inline__ __device__ bool ConstructorProcessor::checkAndReduceHostEnergy(SimulationData& data, Cell* hostCell, ConstructionData const& constructionData)
{
    if (cudaSimulationParameters.features.externalEnergyControl && hostCell->energy < constructionData.energy + cudaSimulationParameters.cellNormalEnergy[hostCell->color]
        && cudaSimulationParameters.externalEnergyInflowFactor[hostCell->color] > 0) {
        auto externalEnergyPortion = constructionData.energy * cudaSimulationParameters.externalEnergyInflowFactor[hostCell->color];

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

    auto cellFunctionConstructorPumpEnergyFactor = cudaSimulationParameters.externalEnergyConditionalInflowFactor[hostCell->color];
    //if (isSelfReplicator(hostCell)) {
    //    externalEnergyConditionalInflowFactor = 0;
    //}

    auto energyNeededFromHost = max(0.0f, constructionData.energy - cudaSimulationParameters.cellNormalEnergy[hostCell->color])
        + min(constructionData.energy, cudaSimulationParameters.cellNormalEnergy[hostCell->color]) * (1.0f - cellFunctionConstructorPumpEnergyFactor);

    if (cellFunctionConstructorPumpEnergyFactor < 1.0f && hostCell->energy < cudaSimulationParameters.cellNormalEnergy[hostCell->color] + energyNeededFromHost) {
        return false;
    }
    auto energyNeededFromRadiation = constructionData.energy - energyNeededFromHost;
    auto orig = atomicAdd(data.externalEnergy, -energyNeededFromRadiation);
    if (orig < energyNeededFromRadiation) {
        atomicAdd(data.externalEnergy, energyNeededFromRadiation);
        if (hostCell->energy < cudaSimulationParameters.cellNormalEnergy[hostCell->color] + constructionData.energy) {
            return false;
        }
        hostCell->energy -= constructionData.energy;
    } else {
        hostCell->energy -= energyNeededFromHost;
    }
    return true;
}

__inline__ __device__ bool ConstructorProcessor::isSelfReplicator(Cell* cell)
{
    if (cell->cellFunction != CellFunction_Constructor) {
        return false;
    }
    return GenomeDecoder::containsSelfReplication(cell->cellFunctionData.constructor);
}

__inline__ __device__ int ConstructorProcessor::calcGenomeComplexity(int color, uint8_t* genome, uint16_t genomeSize)
{
    int lastDepth = 0;
    auto result = 0.0f;
    int acceleration = 1;

    auto genomeComplexityRamificationFactor =
        cudaSimulationParameters.features.genomeComplexityMeasurement ? cudaSimulationParameters.genomeComplexityRamificationFactor[color] : 0.0f;
    auto sizeFactor =
        cudaSimulationParameters.features.genomeComplexityMeasurement ? cudaSimulationParameters.genomeComplexitySizeFactor[color] : 1.0f;
    GenomeDecoder::executeForEachNodeRecursively(genome, toInt(genomeSize), false, [&](int depth, int nodeAddress, int repetitions) {
        float ramificationFactor = depth > lastDepth ? genomeComplexityRamificationFactor * toFloat(acceleration) : 0.0f;
        result += powf(2.0f, toFloat(depth)) * toFloat(repetitions) * (ramificationFactor + sizeFactor);
        lastDepth = depth;
        ++acceleration;
    });
    return toInt(result);
}
