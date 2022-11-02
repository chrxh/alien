#pragma once

#include "EngineInterface/Enums.h"

#include "QuantityConverter.cuh"
#include "CellFunctionProcessor.cuh"
#include "SimulationResult.cuh"

class ConstructorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);
    __inline__ __device__ static bool isConstructionFinished(Cell* cell);
    __inline__ __device__ static bool isConstructionPossible(SimulationData const& data, Cell* cell, Activity const& activity);

    struct ConstructionData
    {
        float angle;
        float distance;
        int maxConnections;
        int executionOrderNumber;
        int color;
        bool inputBlocked;
        bool outputBlocked;
        Enums::CellFunction cellFunction;
    };
    __inline__ __device__ static ConstructionData readConstructionData(Cell* cell);

    __inline__ __device__ static bool
    tryConstructCell(SimulationData& data, SimulationResult& result, Cell* hostCell, ConstructionData const& constructionData);

    __inline__ __device__ static Cell* getFirstCellOfConstructionSite(Cell* hostCell);
    __inline__ __device__ static bool
    startNewConstruction(SimulationData& data, SimulationResult& result, Cell* hostCell, ConstructionData const& constructionData);
    __inline__ __device__ static bool continueConstruction(
        SimulationData& data,
        SimulationResult& result,
        Cell* hostCell,
        Cell* firstConstructedCell,
        ConstructionData const& constructionData);

    __inline__ __device__ static bool isConnectable(int numConnections, int maxConnections, bool makeSticky);

    struct AnglesForNewConnection
    {
        float angleFromPreviousConnection;
        float angleForCell;
    };
    __inline__ __device__ static AnglesForNewConnection calcAnglesForNewConnection(SimulationData& data, Cell* cell, float angleDeviation);

    __inline__ __device__ static Cell* constructCellIntern(
        SimulationData& data,
        Cell* hostCell,
        float2 const& newCellPos,
        ConstructionData const& constructionData);

    __inline__ __device__ static bool isFinished(ConstructorFunction& constructor);
    __inline__ __device__ static bool readBool(ConstructorFunction& constructor);
    __inline__ __device__ static uint8_t readByte(ConstructorFunction& constructor);
    __inline__ __device__ static int readWord(ConstructorFunction& constructor);
    __inline__ __device__ static float readFloat(ConstructorFunction& constructor);   //return values from -1 to 1
    template <typename GenomeHolderSource, typename GenomeHolderTarget>
    __inline__ __device__ static void copyGenome(SimulationData& data, GenomeHolderSource& source, GenomeHolderTarget& target);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

namespace
{
    float constexpr offspringCellDistance = 1.6f;
    float constexpr activityThreshold = 0.25f;
}

__inline__ __device__ void ConstructorProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto& operations = data.cellFunctionOperations[Enums::CellFunction_Constructor];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto operation = operations.at(i);
        auto cell = operation.cell;
        processCell(data, result, cell);
    }
}

__inline__ __device__ void ConstructorProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    if (!isConstructionFinished(cell)) {
        if (isConstructionPossible(data, cell, activity)) {
            auto constructionData = readConstructionData(cell);
            if (tryConstructCell(data, result, cell, constructionData)) {
                activity.channels[0] = 1;
            } else {
                activity.channels[0] = 0;
            }
            if (isFinished(cell->cellFunctionData.constructor)) {
                auto& constructor = cell->cellFunctionData.constructor;
                if (!constructor.singleConstruction) {
                    constructor.currentGenomePos = 0;
                }
            }
        } else {
            activity.channels[0] = 0;
        }
    }
    CellFunctionProcessor::setActivity(cell, activity);
}

__inline__ __device__ bool ConstructorProcessor::isConstructionFinished(Cell* cell)
{
    return cell->cellFunctionData.constructor.currentGenomePos >= cell->cellFunctionData.constructor.genomeSize;
}

__inline__ __device__ bool ConstructorProcessor::isConstructionPossible(SimulationData const& data, Cell* cell, Activity const& activity)
{
    if (cell->energy < cudaSimulationParameters.cellNormalEnergy * 2) {
        return false;
    }
    if (cell->cellFunctionData.constructor.mode == 0 && abs(activity.channels[0]) < activityThreshold) {
        return false;
    }
    if (cell->cellFunctionData.constructor.mode > 0
        && (data.timestep % (cudaSimulationParameters.cellMaxExecutionOrderNumbers * cell->cellFunctionData.constructor.mode) != cell->executionOrderNumber)) {
        return false;
    }
    if (cell->cellFunctionData.constructor.currentGenomePos >= cell->cellFunctionData.constructor.genomeSize) {
        return false;
    }
    return true;
}

__inline__ __device__ ConstructorProcessor::ConstructionData ConstructorProcessor::readConstructionData(Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;

    ConstructionData result;
    result.cellFunction = readByte(constructor) % Enums::CellFunction_Count;
    result.angle = readFloat(constructor) * 180;
    result.distance = readFloat(constructor) + 1.0f;
    result.maxConnections = readByte(constructor) % (cudaSimulationParameters.cellMaxBonds + 1);
    result.executionOrderNumber = readByte(constructor) % cudaSimulationParameters.cellMaxExecutionOrderNumbers;
    result.color = readByte(constructor) % 7;
    result.inputBlocked = readBool(constructor);
    result.outputBlocked = readBool(constructor);
    return result;
}

__inline__ __device__ bool
ConstructorProcessor::tryConstructCell(SimulationData& data, SimulationResult& result, Cell* hostCell, ConstructionData const& constructionData)
{
    Cell* underConstructionCell = getFirstCellOfConstructionSite(hostCell);
    if (underConstructionCell) {
        if (!underConstructionCell->tryLock()) {
            return false;
        }

        auto success = continueConstruction(data, result, hostCell, underConstructionCell, constructionData);
        underConstructionCell->releaseLock();

        return success;
    } else {

        return startNewConstruction(data, result, hostCell, constructionData);
    }
    return true;
}

__inline__ __device__ Cell* ConstructorProcessor::getFirstCellOfConstructionSite(Cell* hostCell)
{
    Cell* result = nullptr;
    for (int i = 0; i < hostCell->numConnections; ++i) {
        auto const& connectingCell = hostCell->connections[i].cell;
        if (connectingCell->underConstruction) {
            result = connectingCell;
        }
    }
    return result;
}

__inline__ __device__ bool ConstructorProcessor::startNewConstruction(
    SimulationData& data,
    SimulationResult& result,
    Cell* hostCell,
    ConstructionData const& constructionData)
{
    auto makeSticky = hostCell->cellFunctionData.constructor.makeSticky;

    if (!isConnectable(hostCell->numConnections, hostCell->maxConnections, makeSticky)) {
        return false;
    }

    auto anglesForNewConnection = calcAnglesForNewConnection(data, hostCell, constructionData.angle);

    auto newCellDirection = Math::unitVectorOfAngle(anglesForNewConnection.angleForCell) * offspringCellDistance;
    float2 newCellPos = hostCell->absPos + newCellDirection;

    Cell* newCell = constructCellIntern(data, hostCell, newCellPos, constructionData);
    hostCell->energy -= cudaSimulationParameters.cellNormalEnergy;

    if (!newCell->tryLock()) {
        return false;
    }

    if (!isFinished(hostCell->cellFunctionData.constructor) || !hostCell->cellFunctionData.constructor.separateConstruction) {
        CellConnectionProcessor::addConnections(
            data,
            hostCell,
            newCell,
            anglesForNewConnection.angleFromPreviousConnection,
            0,
            offspringCellDistance);
    }
    if (isFinished(hostCell->cellFunctionData.constructor)) {
        newCell->underConstruction = false;
    }
    if (!makeSticky) {
        hostCell->maxConnections = hostCell->numConnections;
        newCell->maxConnections = newCell->numConnections;
    }

    newCell->releaseLock();

    result.incCreatedCell();
    return true;
}

__inline__ __device__ bool ConstructorProcessor::continueConstruction(
    SimulationData& data,
    SimulationResult& result,
    Cell* hostCell,
    Cell* underConstructionCell,
    ConstructionData const& constructionData)
{
    auto posDelta = underConstructionCell->absPos - hostCell->absPos;
    data.cellMap.correctDirection(posDelta);

    auto desiredDistance = constructionData.distance;
    posDelta = Math::normalized(posDelta) * (offspringCellDistance - desiredDistance);

    if (Math::length(posDelta) <= cudaSimulationParameters.cellMinDistance || offspringCellDistance - desiredDistance < 0) {
        return false;
    }
    auto makeSticky = hostCell->cellFunctionData.constructor.makeSticky;
    if (makeSticky && 1 == constructionData.maxConnections) {
        return false;
    }

    auto newCellPos = hostCell->absPos + posDelta;

    Cell* newCell = constructCellIntern(data, hostCell, newCellPos, constructionData);
    underConstructionCell->underConstruction = false;

    if (!newCell->tryLock()) {
        return false;
    }

    float angleFromPreviousForCell;
    for (int i = 0; i < hostCell->numConnections; ++i) {
        if (hostCell->connections[i].cell == underConstructionCell) {
            angleFromPreviousForCell = hostCell->connections[i].angleFromPrevious;
            break;
        }
    }

    float angleFromPreviousForUnderConstructionCell;
    for (int i = 0; i < underConstructionCell->numConnections; ++i) {
        if (underConstructionCell->connections[i].cell == hostCell) {
            angleFromPreviousForUnderConstructionCell = underConstructionCell->connections[i].angleFromPrevious;
            break;
        }
    }
    CellConnectionProcessor::delConnections(hostCell, underConstructionCell);
    if (!isFinished(hostCell->cellFunctionData.constructor) || !hostCell->cellFunctionData.constructor.separateConstruction) {
        CellConnectionProcessor::addConnections(
            data, hostCell, newCell, angleFromPreviousForCell, 0, offspringCellDistance);
    }
    auto angleFromPreviousForNewCell = constructionData.angle;
    CellConnectionProcessor::addConnections(
        data, newCell, underConstructionCell, angleFromPreviousForNewCell, angleFromPreviousForUnderConstructionCell, desiredDistance);

    if (isFinished(hostCell->cellFunctionData.constructor)) {
        newCell->underConstruction = false;
    }

    Math::normalize(posDelta);
    Math::rotateQuarterClockwise(posDelta);
    Cell* otherCells[18];
    int numOtherCells;
    data.cellMap.get(otherCells, 18, numOtherCells, newCellPos, offspringCellDistance);
    for (int i = 0; i < numOtherCells; ++i) {
        Cell* otherCell = otherCells[i];
        if (otherCell == underConstructionCell) {
            continue;
        }
        if (otherCell == hostCell) {
            continue;
        }

        bool connected = false;
        for (int i = 0; i < hostCell->numConnections; ++i) {
            auto const& connectedCell = hostCell->connections[i].cell;
            if (connectedCell == otherCell) {
                connected = true;
                break;
            }
        }
        if (connected) {
            continue;
        }

        auto otherPosDelta = otherCell->absPos - newCell->absPos;
        data.cellMap.correctDirection(otherPosDelta);
        Math::normalize(otherPosDelta);
        if (Math::dot(posDelta, otherPosDelta) < 0.1) {
            continue;
        }
        if (otherCell->tryLock()) {
            if (isConnectable(newCell->numConnections, newCell->maxConnections, makeSticky)
                && isConnectable(otherCell->numConnections, otherCell->maxConnections, makeSticky)) {

                CellConnectionProcessor::addConnections(data, newCell, otherCell, 0, 0, desiredDistance, hostCell->cellFunctionData.constructor.angleAlignment);
            }
            otherCell->releaseLock();
        }
    }

    if (!makeSticky) {
        hostCell->maxConnections = hostCell->numConnections;
        newCell->maxConnections = newCell->numConnections;
    }

    newCell->releaseLock();

    result.incCreatedCell();
}

__inline__ __device__ bool ConstructorProcessor::isConnectable(int numConnections, int maxConnections, bool makeSticky)
{
    if (makeSticky) {
        if (numConnections >= maxConnections) {
            return false;
        }
    } else {
        if (numConnections >= cudaSimulationParameters.cellMaxBonds) {
            return false;
        }
    }
    return true;
}

__inline__ __device__ auto ConstructorProcessor::calcAnglesForNewConnection(SimulationData& data, Cell* cell, float angleDeviation) -> AnglesForNewConnection
{
    if (0 == cell->numConnections) {
        return AnglesForNewConnection{0, 0};
    }
    auto displacement = cell->connections[0].cell->absPos - cell->absPos;
    data.cellMap.correctDirection(displacement);
    auto angle = Math::angleOfVector(displacement);
    int index = 0;
    float largestAngleGap = 0;
    float angleOfLargestAngleGap = 0;
    auto numConnections = cell->numConnections;
    for (int i = 1; i <= numConnections; ++i) {
        auto angleDiff = cell->connections[i % numConnections].angleFromPrevious;
        if (angleDiff > largestAngleGap) {
            largestAngleGap = angleDiff;
            index = i % numConnections;
            angleOfLargestAngleGap = angle;
        }
        angle += angleDiff;
    }
    auto angleFromPreviousConnection = cell->connections[index].angleFromPrevious / 2 + angleDeviation;
    if (angleFromPreviousConnection > 360.0f) {
        angleFromPreviousConnection -= 360;
    }

    angleFromPreviousConnection = max(min(angleFromPreviousConnection, cell->connections[index].angleFromPrevious), 0.0f);

    return AnglesForNewConnection{angleFromPreviousConnection, angleOfLargestAngleGap + angleFromPreviousConnection};
}

__inline__ __device__ Cell*
ConstructorProcessor::constructCellIntern(
    SimulationData& data,
    Cell* hostCell,
    float2 const& posOfNewCell,
    ConstructionData const& constructionData)
{
    ObjectFactory factory;
    factory.init(&data);

    Cell * result = factory.createCell();
    result->energy = cudaSimulationParameters.cellNormalEnergy;
    result->absPos = posOfNewCell;
    data.cellMap.correctPosition(result->absPos);
    result->maxConnections = constructionData.maxConnections;
    result->numConnections = 0;
    result->executionOrderNumber = constructionData.executionOrderNumber;
    result->underConstruction = true;
    result->cellFunction = constructionData.cellFunction;
    result->color = constructionData.color;
    result->inputBlocked = constructionData.inputBlocked;
    result->outputBlocked = constructionData.outputBlocked;

    auto& constructor = hostCell->cellFunctionData.constructor;

    switch (constructionData.cellFunction) {
    case Enums::CellFunction_Neuron: {
        result->cellFunctionData.neuron.neuronState = data.objects.auxiliaryData.getTypedSubArray<NeuronFunction::NeuronState>(1);
        for (int i = 0; i < MAX_CHANNELS *  MAX_CHANNELS; ++i) {
            result->cellFunctionData.neuron.neuronState->weights[i] = readFloat(constructor) * 2;
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            result->cellFunctionData.neuron.neuronState->bias[i] = readFloat(constructor) * 2;
        }
    } break;
    case Enums::CellFunction_Transmitter: {
    } break;
    case Enums::CellFunction_Constructor: {
        auto& newConstructor = result->cellFunctionData.constructor;
        newConstructor.mode = readByte(constructor);
        newConstructor.singleConstruction = readBool(constructor);
        newConstructor.separateConstruction = readBool(constructor);
        newConstructor.makeSticky = readBool(constructor);
        newConstructor.angleAlignment = readByte(constructor) % 7;
        newConstructor.currentGenomePos = 0;
        copyGenome(data, constructor, newConstructor);
    } break;
    case Enums::CellFunction_Sensor: {
        result->cellFunctionData.sensor.mode = readByte(constructor) % Enums::SensorMode_Count;
        result->cellFunctionData.sensor.angle = readFloat(constructor) * 180;
        result->cellFunctionData.sensor.minDensity = (readFloat(constructor) + 1.0f) / 2;
        result->cellFunctionData.sensor.color = readByte(constructor) % MAX_COLORS;
    } break;
    case Enums::CellFunction_Nerve: {
    } break;
    case Enums::CellFunction_Attacker: {
    } break;
    case Enums::CellFunction_Injector: {
        copyGenome(data, constructor, result->cellFunctionData.injector);
    } break;
    case Enums::CellFunction_Muscle: {
    } break;
    case Enums::CellFunction_Placeholder1: {
    } break;
    case Enums::CellFunction_Placeholder2: {
    } break;
    }

    return result;
}

__inline__ __device__ bool ConstructorProcessor::isFinished(ConstructorFunction& constructor) {
    return constructor.currentGenomePos >= constructor.genomeSize;
}

__inline__ __device__ bool ConstructorProcessor::readBool(ConstructorFunction& constructor)
{
    return static_cast<int8_t>(readByte(constructor)) > 0;
}

__inline__ __device__ uint8_t ConstructorProcessor::readByte(ConstructorFunction& constructor)
{
    if (isFinished(constructor)) {
        return 0;
    }
    uint8_t result = constructor.genome[constructor.currentGenomePos++];
    return result;
}

__inline__ __device__ int ConstructorProcessor::readWord(ConstructorFunction& constructor)
{
    return static_cast<int>(readByte(constructor)) | (static_cast<int>(readByte(constructor) << 8));
}

__inline__ __device__ float ConstructorProcessor::readFloat(ConstructorFunction& constructor)
{
    return static_cast<float>(static_cast<int8_t>(readByte(constructor))) / 128.0f;
}

template <typename GenomeHolderSource, typename GenomeHolderTarget>
__inline__ __device__ void ConstructorProcessor::copyGenome(SimulationData& data, GenomeHolderSource& source, GenomeHolderTarget& target)
{
    bool makeGenomeCopy = readBool(source);
    if (!makeGenomeCopy) {
        auto size = readWord(source) % MAX_GENOME_BYTES;
        target.genomeSize = size;
        target.genome = data.objects.auxiliaryData.getAlignedSubArray(size);
        //#TODO can be optimized
        for (int i = 0; i < size; ++i) {
            target.genome[i] = readByte(source);
        }
    } else {
        auto size = source.genomeSize;
        target.genomeSize = size;
        target.genome = data.objects.auxiliaryData.getAlignedSubArray(size);
        //#TODO can be optimized
        for (int i = 0; i < size; ++i) {
            target.genome[i] = source.genome[i];
        }
    }
}
