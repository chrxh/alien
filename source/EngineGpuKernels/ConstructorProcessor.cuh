#pragma once

#include "EngineInterface/CellFunctionEnums.h"

#include "QuantityConverter.cuh"
#include "CellFunctionProcessor.cuh"
#include "CudaSimulationFacade.cuh"
#include "SimulationStatistics.cuh"
#include "CellConnectionProcessor.cuh"
#include "MutationProcessor.cuh"
#include "GenomeDecoder.cuh"

class ConstructorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    struct ConstructionData
    {
        GenomeInfo genomeInfo;

        float angle;
        float energy;
        int numRequiredAdditionalConnections;
        int executionOrderNumber;
        int color;
        int inputExecutionOrderNumber;
        bool outputBlocked;
        CellFunction cellFunction;
    };

    __inline__ __device__ static void processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell);
    __inline__ __device__ static bool isConstructionFinished(Cell* cell);
    __inline__ __device__ static bool
    isConstructionTriggered(SimulationData const& data, Cell* cell, ConstructionData const& constructionData, Activity const& activity);
    __inline__ __device__ static ConstructionData readConstructionData(Cell* cell);

    __inline__ __device__ static bool tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static Cell* getFirstCellOfConstructionSite(Cell* hostCell);
    __inline__ __device__ static bool startNewConstruction(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData);
    __inline__ __device__ static bool
    continueConstruction(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, Cell* firstConstructedCell,
        ConstructionData const& constructionData);

    __inline__ __device__ static bool isConnectable(int numConnections, int maxConnections, bool adaptMaxConnections);

    __inline__ __device__ static Cell*
    constructCellIntern(SimulationData& data, Cell* hostCell, float2 const& newCellPos, int maxConnections, ConstructionData const& constructionData);

    __inline__ __device__ static bool checkAndReduceHostEnergy(SimulationData& data, Cell* hostCell, ConstructionData const& constructionData);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ConstructorProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& operations = data.cellFunctionOperations[CellFunction_Constructor];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        processCell(data, statistics, operations.at(i).cell);
    }
}

__inline__ __device__ void ConstructorProcessor::processCell(SimulationData& data, SimulationStatistics& statistics, Cell* cell)
{
    MutationProcessor::applyRandomMutation(data, cell);

    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    if (!isConstructionFinished(cell)) {
        auto origGenomePos = cell->cellFunctionData.constructor.currentGenomePos;
        auto constructionData = readConstructionData(cell);
        if (isConstructionTriggered(data, cell, constructionData, activity)) {
           if (tryConstructCell(data, statistics, cell, constructionData)) {
                activity.channels[0] = 1;
            } else {
                activity.channels[0] = 0;
                cell->cellFunctionData.constructor.currentGenomePos = origGenomePos;
            }
            if (GenomeDecoder::isFinished(cell->cellFunctionData.constructor)) {
                auto& constructor = cell->cellFunctionData.constructor;
                if (!constructionData.genomeInfo.singleConstruction) {
                    constructor.currentGenomePos = 0;
                }
            }
        } else {
            activity.channels[0] = 0;
            cell->cellFunctionData.constructor.currentGenomePos = origGenomePos;
        }
    }
    CellFunctionProcessor::setActivity(cell, activity);
}

__inline__ __device__ bool ConstructorProcessor::isConstructionFinished(Cell* cell)
{
    return cell->cellFunctionData.constructor.currentGenomePos >= cell->cellFunctionData.constructor.genomeSize;
}

__inline__ __device__ bool
ConstructorProcessor::isConstructionTriggered(SimulationData const& data, Cell* cell, ConstructionData const& constructionData, Activity const& activity)
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

__inline__ __device__ ConstructorProcessor::ConstructionData ConstructorProcessor::readConstructionData(Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    if (constructor.currentGenomePos == 0) {
        constructor.currentGenomePos = Const::GenomeInfoSize;
    }

    ConstructionData result;
    result.cellFunction = GenomeDecoder::readByte(constructor) % CellFunction_Count;
    result.angle = GenomeDecoder::readAngle(constructor);
    result.energy = GenomeDecoder::readEnergy(constructor);
    result.numRequiredAdditionalConnections = GenomeDecoder::readByte(constructor);
    result.numRequiredAdditionalConnections = result.numRequiredAdditionalConnections > 127 ? -1 : result.numRequiredAdditionalConnections % (MAX_CELL_BONDS + 1);
    result.executionOrderNumber = GenomeDecoder::readByte(constructor) % cudaSimulationParameters.cellNumExecutionOrderNumbers;
    result.color = GenomeDecoder::readByte(constructor) % MAX_COLORS;
    result.inputExecutionOrderNumber = GenomeDecoder::readOptionalByte(constructor, cudaSimulationParameters.cellNumExecutionOrderNumbers);
    result.outputBlocked = GenomeDecoder::readBool(constructor);

    //genome-wide data
    result.genomeInfo = GenomeDecoder::readGenomeInfo(constructor);
    return result;
}

__inline__ __device__ bool
ConstructorProcessor::tryConstructCell(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData)
{
    if (!hostCell->tryLock()) {
        return false;
    }
    Cell* underConstructionCell = getFirstCellOfConstructionSite(hostCell);
    if (underConstructionCell) {
        if (!underConstructionCell->tryLock()) {
            hostCell->releaseLock();
            return false;
        }

        auto success = continueConstruction(data, statistics, hostCell, underConstructionCell, constructionData);

        underConstructionCell->releaseLock();
        hostCell->releaseLock();
        return success;
    } else {
        auto success = startNewConstruction(data, statistics, hostCell, constructionData);

        hostCell->releaseLock();
        return success;
    }
}

__inline__ __device__ Cell* ConstructorProcessor::getFirstCellOfConstructionSite(Cell* hostCell)
{
    Cell* result = nullptr;
    for (int i = 0; i < hostCell->numConnections; ++i) {
        auto const& connectingCell = hostCell->connections[i].cell;
        if (connectingCell->livingState == LivingState_UnderConstruction) {
            result = connectingCell;
        }
    }
    return result;
}

__inline__ __device__ bool
ConstructorProcessor::startNewConstruction(SimulationData& data, SimulationStatistics& statistics, Cell* hostCell, ConstructionData const& constructionData)
{
    if (!isConnectable(hostCell->numConnections, hostCell->maxConnections, true)) {
        return false;
    }

    auto anglesForNewConnection = CellFunctionProcessor::calcLargestGapReferenceAndActualAngle(data, hostCell, constructionData.angle);

    auto newCellDirection = Math::unitVectorOfAngle(anglesForNewConnection.actualAngle) * cudaSimulationParameters.cellFunctionConstructorOffspringDistance[hostCell->color];
    float2 newCellPos = hostCell->absPos + newCellDirection;

    if (CellConnectionProcessor::existCrossingConnections(data, hostCell->absPos, newCellPos, hostCell->detached)) {
        return false;
    }

    if (!checkAndReduceHostEnergy(data, hostCell, constructionData)) {
        return false;
    }

    hostCell->constructionId = data.numberGen1.random(65535);
    Cell* newCell = constructCellIntern(data, hostCell, newCellPos, 0, constructionData);

    if (!newCell->tryLock()) {
        return false;
    }

    if (!GenomeDecoder::isFinished(hostCell->cellFunctionData.constructor) || !constructionData.genomeInfo.separateConstruction) {
        auto const& constructor = hostCell->cellFunctionData.constructor;
        auto distance =
            GenomeDecoder::isFinished(constructor) && !constructionData.genomeInfo.separateConstruction && constructionData.genomeInfo.singleConstruction
            ? 1.0f
            : cudaSimulationParameters.cellFunctionConstructorOffspringDistance[hostCell->color];
        CellConnectionProcessor::tryAddConnections(data, hostCell, newCell, anglesForNewConnection.referenceAngle, 0, distance);
    }
    if (GenomeDecoder::isFinished(hostCell->cellFunctionData.constructor)) {
        newCell->livingState = LivingState_JustReady;
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
    Cell* underConstructionCell,
    ConstructionData const& constructionData)
{
    auto posDelta = underConstructionCell->absPos - hostCell->absPos;
    data.cellMap.correctDirection(posDelta);

    auto desiredDistance = 1.0f;
    auto constructionSiteDistance = data.cellMap.getDistance(hostCell->absPos, underConstructionCell->absPos);
    posDelta = Math::normalized(posDelta) * (constructionSiteDistance - desiredDistance);

    if (Math::length(posDelta) <= cudaSimulationParameters.cellMinDistance
        || constructionSiteDistance - desiredDistance < cudaSimulationParameters.cellMinDistance) {
        return false;
    }

    auto newCellPos = hostCell->absPos + posDelta;

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
            if (otherCell == underConstructionCell || otherCell == hostCell || otherCell->livingState != LivingState_UnderConstruction
                || otherCell->constructionId != hostCell->constructionId) {
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
    Cell* newCell = constructCellIntern(data, hostCell, newCellPos, 0, constructionData);

    if (!newCell->tryLock()) {
        return false;
    }

    auto const& constructor = hostCell->cellFunctionData.constructor;
    if (GenomeDecoder::isFinished(constructor)) {
        newCell->livingState = LivingState_JustReady;
    }

    float angleFromPreviousForUnderConstructionCell;
    for (int i = 0; i < underConstructionCell->numConnections; ++i) {
        if (underConstructionCell->connections[i].cell == hostCell) {
            angleFromPreviousForUnderConstructionCell = underConstructionCell->connections[i].angleFromPrevious;
            break;
        }
    }

    //cut connections
    CellConnectionProcessor::deleteConnections(hostCell, underConstructionCell);

    //possibly connect newCell to hostCell
    bool adaptReferenceAngle = false;
    if (!GenomeDecoder::isFinished(hostCell->cellFunctionData.constructor) || !constructionData.genomeInfo.separateConstruction) {
        auto distance = GenomeDecoder::isFinished(constructor) && !constructionData.genomeInfo.separateConstruction
            ? 1.0f
            : cudaSimulationParameters.cellFunctionConstructorOffspringDistance[hostCell->color];
        if (CellConnectionProcessor::tryAddConnections(data, hostCell, newCell, /*angleFromPreviousForCell*/ 0, 0, distance)) {
            adaptReferenceAngle = true;
        }
    }

    //connect newCell to underConstructionCell
    auto angleFromPreviousForNewCell = 180.0f - constructionData.angle;
    if (!CellConnectionProcessor::tryAddConnections(
        data, newCell, underConstructionCell, /*angleFromPreviousForNewCell*/0, angleFromPreviousForUnderConstructionCell, desiredDistance)) {
        adaptReferenceAngle = false;
    }

    Math::normalize(posDelta);
    Math::rotateQuarterClockwise(posDelta);

    //get surrounding cells
    if (numOtherCells > 0) {

        //sort surrounding cells by distance from newCell
        bubbleSort(otherCells, numOtherCells, [&](auto const& cell1, auto const& cell2) {
            auto dist1 = data.cellMap.getDistance(cell1->absPos, newCellPos);
            auto dist2 = data.cellMap.getDistance(cell2->absPos, newCellPos);
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

                    CellConnectionProcessor::tryAddConnections(data, newCell, otherCell, 0, 0, desiredDistance, constructionData.genomeInfo.angleAlignment);
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
            if (newCell->connections[constructionIndex].cell == underConstructionCell) {
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
    Cell* hostCell,
    float2 const& posOfNewCell,
    int maxConnections,
    ConstructionData const& constructionData)
{
    auto& constructor = hostCell->cellFunctionData.constructor;

    ObjectFactory factory;
    factory.init(&data);

    Cell * result = factory.createCell();
    result->energy = constructionData.energy;
    result->stiffness = constructionData.genomeInfo.stiffness;
    result->absPos = posOfNewCell;
    data.cellMap.correctPosition(result->absPos);
    result->maxConnections = maxConnections;
    result->numConnections = 0;
    result->executionOrderNumber = constructionData.executionOrderNumber;
    result->livingState = true;
    result->constructionId = hostCell->constructionId;
    result->cellFunction = constructionData.cellFunction;
    result->color = constructionData.color;
    result->inputExecutionOrderNumber = constructionData.inputExecutionOrderNumber;
    result->outputBlocked = constructionData.outputBlocked;

    result->activationTime = constructor.constructionActivationTime;

    switch (constructionData.cellFunction) {
    case CellFunction_Neuron: {
        result->cellFunctionData.neuron.neuronState = data.objects.auxiliaryData.getTypedSubArray<NeuronFunction::NeuronState>(1);
        for (int i = 0; i < MAX_CHANNELS *  MAX_CHANNELS; ++i) {
            result->cellFunctionData.neuron.neuronState->weights[i] = GenomeDecoder::readFloat(constructor) * 4;
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result->cellFunctionData.neuron.neuronState->biases[i] = GenomeDecoder::readFloat(constructor) * 4;
        }
    } break;
    case CellFunction_Transmitter: {
        result->cellFunctionData.transmitter.mode = GenomeDecoder::readByte(constructor) % EnergyDistributionMode_Count;
    } break;
    case CellFunction_Constructor: {
        auto& newConstructor = result->cellFunctionData.constructor;
        newConstructor.activationMode = GenomeDecoder::readByte(constructor);
        newConstructor.constructionActivationTime = GenomeDecoder::readWord(constructor);
        newConstructor.currentGenomePos = 0;
        GenomeDecoder::copyGenome(data, constructor, newConstructor);
        newConstructor.genomeGeneration = constructor.genomeGeneration + 1;
    } break;
    case CellFunction_Sensor: {
        result->cellFunctionData.sensor.mode = GenomeDecoder::readByte(constructor) % SensorMode_Count;
        result->cellFunctionData.sensor.angle = GenomeDecoder::readAngle(constructor);
        result->cellFunctionData.sensor.minDensity = (GenomeDecoder::readFloat(constructor) + 1.0f) / 2;
        result->cellFunctionData.sensor.color = GenomeDecoder::readByte(constructor) % MAX_COLORS;
    } break;
    case CellFunction_Nerve: {
        result->cellFunctionData.nerve.pulseMode = GenomeDecoder::readByte(constructor);
        result->cellFunctionData.nerve.alternationMode = GenomeDecoder::readByte(constructor);
    } break;
    case CellFunction_Attacker: {
        result->cellFunctionData.attacker.mode = GenomeDecoder::readByte(constructor) % EnergyDistributionMode_Count;
    } break;
    case CellFunction_Injector: {
        result->cellFunctionData.injector.mode = GenomeDecoder::readByte(constructor) % InjectorMode_Count;
        result->cellFunctionData.injector.counter = 0;
        GenomeDecoder::copyGenome(data, constructor, result->cellFunctionData.injector);
        result->cellFunctionData.injector.genomeGeneration = constructor.genomeGeneration + 1;
    } break;
    case CellFunction_Muscle: {
        result->cellFunctionData.muscle.mode = GenomeDecoder::readByte(constructor) % MuscleMode_Count;
        result->cellFunctionData.muscle.lastBendingDirection = MuscleBendingDirection_None;
        result->cellFunctionData.muscle.consecutiveBendingAngle = 0;
    } break;
    case CellFunction_Defender: {
        result->cellFunctionData.defender.mode = GenomeDecoder::readByte(constructor) % DefenderMode_Count;
    } break;
    case CellFunction_Placeholder: {
    } break;
    }

    return result;
}

__inline__ __device__ bool ConstructorProcessor::checkAndReduceHostEnergy(SimulationData& data, Cell* hostCell, ConstructionData const& constructionData)
{
    if (!cudaSimulationParameters.cellFunctionConstructionUnlimitedEnergy) {
        auto energyNeededFromHost = max(0.0f, constructionData.energy - cudaSimulationParameters.cellNormalEnergy[hostCell->color])
            + min(constructionData.energy, cudaSimulationParameters.cellNormalEnergy[hostCell->color])
                * (1.0f - cudaSimulationParameters.cellFunctionConstructorPumpEnergyFactor[hostCell->color]);

        if (hostCell->energy < cudaSimulationParameters.cellNormalEnergy[hostCell->color] + energyNeededFromHost) {
            return false;
        }
        auto energyNeededFromRadiation = constructionData.energy - energyNeededFromHost;
        auto orig = atomicAdd(data.residualEnergy, -energyNeededFromRadiation);
        if (orig < energyNeededFromRadiation) {
            atomicAdd(data.residualEnergy, energyNeededFromRadiation);
            if (hostCell->energy < cudaSimulationParameters.cellNormalEnergy[hostCell->color] + constructionData.energy) {
                return false;
            }
            hostCell->energy -= constructionData.energy;
        } else {
            hostCell->energy -= energyNeededFromHost;
        }
    }
    return true;
}
