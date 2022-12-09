#pragma once

#include "EngineInterface/Enums.h"

#include "QuantityConverter.cuh"
#include "CellFunctionProcessor.cuh"
#include "CudaSimulationFacade.cuh"
#include "SimulationResult.cuh"
#include "CellConnectionProcessor.cuh"

class ConstructorProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

    __inline__ __device__ static void mutateData(SimulationData& data, Cell* cell);

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

    __inline__ __device__ static bool isConnectable(int numConnections, int maxConnections, bool adaptMaxConnections);

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

    __inline__ __device__ static bool isFinished(ConstructorFunction const& constructor);
    __inline__ __device__ static bool readBool(ConstructorFunction& constructor);
    __inline__ __device__ static uint8_t readByte(ConstructorFunction& constructor);
    __inline__ __device__ static int readWord(ConstructorFunction& constructor);
    __inline__ __device__ static float readFloat(ConstructorFunction& constructor);   //return values from -1 to 1
    __inline__ __device__ static float readAngle(ConstructorFunction& constructor);
    template <typename GenomeHolderSource, typename GenomeHolderTarget>
    __inline__ __device__ static void copyGenome(SimulationData& data, GenomeHolderSource& source, GenomeHolderTarget& target);

    __inline__ __device__ static bool convertByteToBool(uint8_t b);
    __inline__ __device__ static int convertBytesToWord(uint8_t b1, uint8_t b2);

    __inline__ __device__ static void applyMutation(SimulationData& data, Cell* cell);

    //internal constants
    static int constexpr CellFunctionMutationMaxSizeDelta = 100;
    static int constexpr CellFunctionMutationMaxGenomeSize = 50;
    static int constexpr CellBasicBytes = 8;
    static int constexpr NeuronBytes = 72;
    static int constexpr TransmitterBytes = 1;
    static int constexpr ConstructorFixedBytes = 7;
    static int constexpr SensorBytes = 4;
    static int constexpr NerveBytes = 0;
    static int constexpr AttackerBytes = 1;
    static int constexpr InjectorFixedBytes = 0;
    static int constexpr MuscleBytes = 1;
    __inline__ __device__ static int getNumGenomeCells(ConstructorFunction const& constructor);
    __inline__ __device__ static int getGenomeByteIndex(ConstructorFunction const& constructor, int cellIndex);
    __inline__ __device__ static int getGenomeCellIndex(ConstructorFunction const& constructor, int byteIndex);
    __inline__ __device__ static int getNextCellFunctionGenomeBytes(ConstructorFunction const& constructor, int genomePos);
    __inline__ __device__ static int getNextCellFunctionType(ConstructorFunction const& constructor, int genomePos);
    __inline__ __device__ static bool getNextCellFunctionMakeSelfCopy(ConstructorFunction const& constructor, int genomePos);
    __inline__ __device__ static int getCellFunctionDataSize(
        Enums::CellFunction cellFunction,
        bool makeSelfCopy,
        int genomeSize);  //genomeSize only relevant for cellFunction = constructor or injector
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ConstructorProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto& operations = data.cellFunctionOperations[Enums::CellFunction_Constructor];
    auto partition = calcAllThreadsPartition(operations.getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto const& operation = operations.at(i);
        auto const& cell = operation.cell;
        processCell(data, result, cell);
    }
}

__inline__ __device__ void ConstructorProcessor::mutateData(SimulationData& data, Cell* cell)
{
    auto& constructor = cell->cellFunctionData.constructor;
    auto numGenomeCells = getNumGenomeCells(constructor);
    if (numGenomeCells == 0) {
        return;
    }
    auto cellIndex = data.numberGen1.random(numGenomeCells - 1);
    auto genomePos = getGenomeByteIndex(constructor, cellIndex);

    //basic property mutation
    if (data.numberGen1.randomBool()) {
        //auto delta = data.numberGen1.random(CellBasicBytes - 2) + 1;  //+1 since cell function type should not be changed here
        //genomePos = (genomePos + delta) % constructor.genomeSize;
        //constructor.genome[genomePos] = data.numberGen1.randomByte();
    }

    //cell function specific mutation
    else {
        auto nextCellFunctionGenomeBytes = getNextCellFunctionGenomeBytes(constructor, genomePos);
        if (nextCellFunctionGenomeBytes > 0) {
            auto delta = data.numberGen1.random(nextCellFunctionGenomeBytes - 1);
            auto type = getNextCellFunctionType(constructor, genomePos);
            //do not override makeSelfCopy flag (relative position is CellBasicBytes + ConstructorFixedBytes)!
            if (type == Enums::CellFunction_Constructor) {
                if (delta == ConstructorFixedBytes) {
                    return;
                }
                auto makeSelfCopy = getNextCellFunctionMakeSelfCopy(constructor, genomePos);
                if (!makeSelfCopy && (delta == ConstructorFixedBytes + 1 || delta == ConstructorFixedBytes + 2)) {
                    return;
                }
            }
            if (type == Enums::CellFunction_Injector) {
                if (delta == InjectorFixedBytes) {
                    return;
                }
                auto makeSelfCopy = getNextCellFunctionMakeSelfCopy(constructor, genomePos);
                if (!makeSelfCopy && (delta == InjectorFixedBytes + 1 || delta == InjectorFixedBytes + 2)) {
                    return;
                }
            }
            genomePos = (genomePos + CellBasicBytes + delta) % constructor.genomeSize;
            constructor.genome[genomePos] = data.numberGen1.randomByte();
        }
    }
}

__inline__ __device__ void ConstructorProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    applyMutation(data, cell);

    int inputExecutionOrderNumber;
    auto activity = CellFunctionProcessor::calcInputActivity(cell, inputExecutionOrderNumber);
    if (!isConstructionFinished(cell)) {
        if (isConstructionPossible(data, cell, activity)) {
            auto origGenomePos = cell->cellFunctionData.constructor.currentGenomePos;
            auto constructionData = readConstructionData(cell);
            if (tryConstructCell(data, result, cell, constructionData)) {
                activity.channels[0] = 1;
            } else {
                activity.channels[0] = 0;
                cell->cellFunctionData.constructor.currentGenomePos = origGenomePos;
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
    if (cell->cellFunctionData.constructor.mode == 0 && abs(activity.channels[0]) < cudaSimulationParameters.cellFunctionConstructorActivityThreshold) {
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
    result.angle = readAngle(constructor);
    result.distance = readFloat(constructor) + 1.0f;
    result.maxConnections = readByte(constructor) % (cudaSimulationParameters.cellMaxBonds + 1);
    result.executionOrderNumber = readByte(constructor) % cudaSimulationParameters.cellMaxExecutionOrderNumbers;
    result.color = readByte(constructor) % MAX_COLORS;
    result.inputBlocked = readBool(constructor);
    result.outputBlocked = readBool(constructor);
    return result;
}

__inline__ __device__ bool
ConstructorProcessor::tryConstructCell(SimulationData& data, SimulationResult& result, Cell* hostCell, ConstructionData const& constructionData)
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

        auto success = continueConstruction(data, result, hostCell, underConstructionCell, constructionData);

        underConstructionCell->releaseLock();
        hostCell->releaseLock();
        return success;
    } else {
        auto success = startNewConstruction(data, result, hostCell, constructionData);

        hostCell->releaseLock();
        return success;
    }
}

__inline__ __device__ Cell* ConstructorProcessor::getFirstCellOfConstructionSite(Cell* hostCell)
{
    Cell* result = nullptr;
    for (int i = 0; i < hostCell->numConnections; ++i) {
        auto const& connectingCell = hostCell->connections[i].cell;
        if (connectingCell->constructionState == Enums::ConstructionState_UnderConstruction) {
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
    auto adaptMaxConnections = hostCell->cellFunctionData.constructor.adaptMaxConnections;

    if (!isConnectable(hostCell->numConnections, hostCell->maxConnections, adaptMaxConnections)) {
        return false;
    }

    auto anglesForNewConnection = calcAnglesForNewConnection(data, hostCell, constructionData.angle);

    auto newCellDirection = Math::unitVectorOfAngle(anglesForNewConnection.angleForCell) * cudaSimulationParameters.cellFunctionConstructorOffspringDistance;
    float2 newCellPos = hostCell->absPos + newCellDirection;

    Cell* newCell = constructCellIntern(data, hostCell, newCellPos, constructionData);
    hostCell->energy -= cudaSimulationParameters.cellNormalEnergy;

    if (!newCell->tryLock()) {
        return false;
    }

    if (!isFinished(hostCell->cellFunctionData.constructor) || !hostCell->cellFunctionData.constructor.separateConstruction) {
        auto const& constructor = hostCell->cellFunctionData.constructor;
        auto distance = isFinished(constructor) && !constructor.separateConstruction && constructor.singleConstruction
            ? constructionData.distance
            : cudaSimulationParameters.cellFunctionConstructorOffspringDistance;
        CellConnectionProcessor::addConnections(data, hostCell, newCell, anglesForNewConnection.angleFromPreviousConnection, 0, distance);
    }
    if (isFinished(hostCell->cellFunctionData.constructor)) {
        newCell->constructionState = Enums::ConstructionState_JustFinished;
    }
    if (adaptMaxConnections) {
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
    auto constructionSiteDistance = data.cellMap.getDistance(hostCell->absPos, underConstructionCell->absPos);
    posDelta = Math::normalized(posDelta) * (constructionSiteDistance - desiredDistance);

    if (Math::length(posDelta) <= cudaSimulationParameters.cellMinDistance
        || constructionSiteDistance - desiredDistance < cudaSimulationParameters.cellMinDistance) {
        return false;
    }
    auto adaptMaxConnections = hostCell->cellFunctionData.constructor.adaptMaxConnections;
    if (!adaptMaxConnections && 1 == constructionData.maxConnections) {
        return false;
    }

    auto newCellPos = hostCell->absPos + posDelta;

    Cell* newCell = constructCellIntern(data, hostCell, newCellPos, constructionData);
    hostCell->energy -= cudaSimulationParameters.cellNormalEnergy;

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
        auto const& constructor = hostCell->cellFunctionData.constructor;
        auto distance = isFinished(constructor) && !constructor.separateConstruction && constructor.singleConstruction
            ? constructionData.distance
            : cudaSimulationParameters.cellFunctionConstructorOffspringDistance;
        CellConnectionProcessor::addConnections(
            data, hostCell, newCell, angleFromPreviousForCell, 0, distance);
    }
    auto angleFromPreviousForNewCell = constructionData.angle;
    CellConnectionProcessor::addConnections(
        data, newCell, underConstructionCell, angleFromPreviousForNewCell, angleFromPreviousForUnderConstructionCell, desiredDistance);

    if (isFinished(hostCell->cellFunctionData.constructor)) {
        newCell->constructionState = Enums::ConstructionState_JustFinished;
    }

    Math::normalize(posDelta);
    Math::rotateQuarterClockwise(posDelta);
    Cell* otherCells[18];
    int numOtherCells;
    data.cellMap.get(otherCells, 18, numOtherCells, newCellPos, cudaSimulationParameters.cellFunctionConstructorConnectingCellMaxDistance);
    for (int i = 0; i < numOtherCells; ++i) {
        Cell* otherCell = otherCells[i];
        if (otherCell == underConstructionCell || otherCell == hostCell || otherCell->constructionState != Enums::ConstructionState_UnderConstruction) {
            continue;
        }
        if (cudaSimulationParameters.cellFunctionConstructionInheritColor && otherCell->color != hostCell->color) {
            continue;
        }

        if (otherCell->tryLock()) {
            if (isConnectable(newCell->numConnections, newCell->maxConnections, adaptMaxConnections)
                && isConnectable(otherCell->numConnections, otherCell->maxConnections, adaptMaxConnections)) {

                CellConnectionProcessor::addConnections(data, newCell, otherCell, 0, 0, desiredDistance, hostCell->cellFunctionData.constructor.angleAlignment);
            }
            otherCell->releaseLock();
        }
    }

    if (adaptMaxConnections) {
        hostCell->maxConnections = hostCell->numConnections;
        newCell->maxConnections = newCell->numConnections;
    }

    newCell->releaseLock();

    result.incCreatedCell();
    return true;
}

__inline__ __device__ bool ConstructorProcessor::isConnectable(int numConnections, int maxConnections, bool adaptMaxConnections)
{
    if (!adaptMaxConnections) {
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
    result->constructionState = true;
    result->cellFunction = constructionData.cellFunction;
    result->color = cudaSimulationParameters.cellFunctionConstructionInheritColor ? hostCell->color : constructionData.color;
    result->inputBlocked = constructionData.inputBlocked;
    result->outputBlocked = constructionData.outputBlocked;

    auto& constructor = hostCell->cellFunctionData.constructor;
    result->activationTime = constructor.constructionActivationTime;

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
        result->cellFunctionData.transmitter.mode = readByte(constructor) % Enums::EnergyDistributionMode_Count;
    } break;
    case Enums::CellFunction_Constructor: {
        auto& newConstructor = result->cellFunctionData.constructor;
        newConstructor.mode = readByte(constructor);
        newConstructor.singleConstruction = readBool(constructor);
        newConstructor.separateConstruction = readBool(constructor);
        newConstructor.adaptMaxConnections = readBool(constructor);
        newConstructor.angleAlignment = readByte(constructor) % 7;
        newConstructor.constructionActivationTime = readWord(constructor);
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
        result->cellFunctionData.attacker.mode = readByte(constructor) % Enums::EnergyDistributionMode_Count;
    } break;
    case Enums::CellFunction_Injector: {
        copyGenome(data, constructor, result->cellFunctionData.injector);
    } break;
    case Enums::CellFunction_Muscle: {
        result->cellFunctionData.muscle.mode = readByte(constructor) % Enums::MuscleMode_Count;
    } break;
    case Enums::CellFunction_Placeholder1: {
    } break;
    case Enums::CellFunction_Placeholder2: {
    } break;
    }

    return result;
}

__inline__ __device__ bool ConstructorProcessor::isFinished(ConstructorFunction const& constructor) {
    return constructor.currentGenomePos >= constructor.genomeSize;
}

__inline__ __device__ bool ConstructorProcessor::readBool(ConstructorFunction& constructor)
{
    return convertByteToBool(readByte(constructor));
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
    auto b1 = readByte(constructor);
    auto b2 = readByte(constructor);
    return convertBytesToWord(b1, b2);
}

__inline__ __device__ float ConstructorProcessor::readFloat(ConstructorFunction& constructor)
{
    return static_cast<float>(static_cast<int8_t>(readByte(constructor))) / 128.0f;
}

__inline__ __device__ float ConstructorProcessor::readAngle(ConstructorFunction& constructor)
{
    return static_cast<float>(static_cast<int8_t>(readByte(constructor))) / 120 * 180;
}

__inline__ __device__ bool ConstructorProcessor::convertByteToBool(uint8_t b)
{
    return static_cast<int8_t>(b) > 0;
}

__inline__ __device__ int ConstructorProcessor::convertBytesToWord(uint8_t b1, uint8_t b2)
{
    return static_cast<int>(b1) | (static_cast<int>(b2 << 8));
}

__inline__ __device__ void ConstructorProcessor::applyMutation(SimulationData& data, Cell* cell)
{
    auto cellFunctionConstructorMutationDataProbability =
        SpotCalculator::calcParameter(&SimulationParametersSpotValues::cellFunctionConstructorMutationDataProbability, data, cell->absPos);

    if (data.numberGen1.random() < cellFunctionConstructorMutationDataProbability) {
        mutateData(data, cell);
    }

    //cell function changing mutation
    //if (data.numberGen1.random() < 0.0002f) {
    //    auto numCellIndices = getNumGenomeCells(constructor);
    //    if (numCellIndices == 0) {
    //        return;
    //    }
    //    auto mutationCellIndex = data.numberGen1.random(numCellIndices - 1);
    //    auto sourceGenomePos = getGenomeByteIndex(constructor, mutationCellIndex);
    //    auto sourceRemainingGenomePos = sourceGenomePos + CellBasicBytes + getNextCellFunctionGenomeBytes(constructor, sourceGenomePos);

    //    auto targetGenomeSize = constructor.genomeSize + CellFunctionMutationMaxSizeDelta;
    //    auto targetGenome = data.objects.auxiliaryData.getAlignedSubArray(targetGenomeSize);

    //    for (int pos = 0; pos < sourceGenomePos + CellBasicBytes; ++pos) {
    //        targetGenome[pos] = constructor.genome[pos];
    //    }

    //    targetGenome[sourceGenomePos] = data.numberGen1.random(Enums::CellFunction_Count - 1);
    //    auto cellFunctionDataSize =
    //        getCellFunctionDataSize(targetGenome[sourceGenomePos], data.numberGen1.randomByte(), data.numberGen1.random(CellFunctionMutationMaxGenomeSize));
    //    for (int pos = sourceGenomePos + CellBasicBytes; pos < sourceGenomePos + CellBasicBytes + cellFunctionDataSize; ++pos) {
    //        targetGenome[pos] = data.numberGen1.randomByte();
    //    }

    //    auto targetPos = sourceGenomePos + CellBasicBytes + cellFunctionDataSize;
    //    if (sourceRemainingGenomePos > sourceGenomePos) {
    //        auto sourcePos = sourceRemainingGenomePos;
    //        for (; sourcePos < constructor.genomeSize; ++sourcePos, ++targetPos) {
    //            targetGenome[targetPos] = constructor.genome[sourcePos];
    //        }
    //    }
    //    auto cellIndex = getGenomeCellIndex(constructor, mutationCellIndex);
    //    constructor.genome = targetGenome;
    //    constructor.genomeSize = targetPos;
    //    constructor.currentGenomePos = getGenomeByteIndex(constructor, cellIndex);
    //}

    //insert mutation

    //delete mutation

    //if (data.numberGen1.random() < 0.0002f) {
    //    if (constructor.genomeSize > 0) {
    //        int index = data.numberGen1.random(toInt(constructor.genomeSize - 1));
    //        constructor.genome[index] = data.numberGen1.randomByte();
    //    }
    //}
    //if (data.numberGen1.random() < 0.0005f && data.numberGen2.random() < 0.001) {

    //    auto newGenomeSize = min(MAX_GENOME_BYTES, toInt(constructor.genomeSize) + data.numberGen1.random(100));
    //    auto newGenome = data.objects.auxiliaryData.getAlignedSubArray(newGenomeSize);
    //    for (int i = 0; i < constructor.genomeSize; ++i) {
    //        newGenome[i] = constructor.genome[i];
    //    }
    //    for (int i = constructor.genomeSize; i < newGenomeSize; ++i) {
    //        newGenome[i] = data.numberGen1.randomByte();
    //    }
    //    constructor.genome = newGenome;
    //    constructor.genomeSize = newGenomeSize;
    //}
    //if (data.numberGen1.random() < 0.0005f && data.numberGen2.random() < 0.001) {

    //    constructor.genomeSize = max(0, toInt(constructor.genomeSize) - data.numberGen1.random(100));
    //    constructor.currentGenomePos = min(constructor.genomeSize, constructor.currentGenomePos);
    //}
}

__inline__ __device__ int ConstructorProcessor::getNumGenomeCells(ConstructorFunction const& constructor)
{
    int result = 0;
    int currentByteIndex = 0;
    for (; result < constructor.genomeSize; ++result) {
        if (currentByteIndex >= constructor.genomeSize) {
            break;
        }
        currentByteIndex += CellBasicBytes + getNextCellFunctionGenomeBytes(constructor, currentByteIndex);
    }

    return result;
}

__inline__ __device__ int ConstructorProcessor::getGenomeByteIndex(ConstructorFunction const& constructor, int cellIndex)
{
    int currentByteIndex = 0;
    for (int currentCellIndex = 0; currentCellIndex < cellIndex; ++currentCellIndex) {
        if (currentByteIndex >= constructor.genomeSize) {
            break;
        }
        currentByteIndex += CellBasicBytes + getNextCellFunctionGenomeBytes(constructor, currentByteIndex);
    }

    return currentByteIndex % constructor.genomeSize;
}

__inline__ __device__ int ConstructorProcessor::getGenomeCellIndex(ConstructorFunction const& constructor, int byteIndex)
{
    int currentByteIndex = 0;
    for (int currentCellIndex = 0; currentByteIndex <= byteIndex; ++currentCellIndex) {
        if (currentByteIndex == byteIndex) {
            return currentCellIndex;
        }
        currentByteIndex += CellBasicBytes + getNextCellFunctionGenomeBytes(constructor, currentByteIndex);
    }
    return 0;
}

__inline__ __device__ int ConstructorProcessor::getNextCellFunctionGenomeBytes(ConstructorFunction const& constructor, int genomePos)
{
    auto cellFunction = getNextCellFunctionType(constructor, genomePos);
    switch (cellFunction) {
    case Enums::CellFunction_Neuron:
        return NeuronBytes;
    case Enums::CellFunction_Transmitter:
        return TransmitterBytes;
    case Enums::CellFunction_Constructor: {
        auto makeCopyIndex = genomePos + CellBasicBytes + ConstructorFixedBytes;
        auto isMakeCopy = convertByteToBool(constructor.genome[makeCopyIndex % constructor.genomeSize]);
        if (isMakeCopy) {
            return ConstructorFixedBytes + 1;
        } else {
            auto genomeSizeIndex = genomePos + CellBasicBytes + ConstructorFixedBytes + 1;
            auto genomeSize = convertBytesToWord(
                constructor.genome[genomeSizeIndex % constructor.genomeSize], constructor.genome[(genomeSizeIndex + 1)% constructor.genomeSize]);
            return ConstructorFixedBytes + 3 + genomeSize;
        }
    }
    case Enums::CellFunction_Sensor:
        return SensorBytes;
    case Enums::CellFunction_Nerve:
        return NerveBytes;
    case Enums::CellFunction_Attacker:
        return AttackerBytes;
    case Enums::CellFunction_Injector: {
        auto makeCopyIndex = genomePos + CellBasicBytes + InjectorFixedBytes;
        auto isMakeCopy = convertByteToBool(constructor.genome[makeCopyIndex % constructor.genomeSize]);
        if (isMakeCopy) {
            return InjectorFixedBytes + 1;
        } else {
            auto genomeSizeIndex = genomePos + CellBasicBytes + InjectorFixedBytes + 1;
            auto genomeSize = convertBytesToWord(
                constructor.genome[genomeSizeIndex % constructor.genomeSize], constructor.genome[(genomeSizeIndex + 1) % constructor.genomeSize]);
            return InjectorFixedBytes + 3 + genomeSize;
        }
    }
    case Enums::CellFunction_Muscle:
        return MuscleBytes;
    default: 
        return 0;
    }
}

__inline__ __device__ int ConstructorProcessor::getNextCellFunctionType(ConstructorFunction const& constructor, int genomePos)
{
    return constructor.genome[genomePos] % Enums::CellFunction_Count;
}

__inline__ __device__ bool ConstructorProcessor::getNextCellFunctionMakeSelfCopy(ConstructorFunction const& constructor, int genomePos)
{
    switch(getNextCellFunctionType(constructor, genomePos)) {
    case Enums::CellFunction_Constructor:
        return convertByteToBool(constructor.genome[(genomePos + CellBasicBytes + ConstructorFixedBytes) % constructor.genomeSize]);
    case Enums::CellFunction_Injector:
        return convertByteToBool(constructor.genome[(genomePos + CellBasicBytes + InjectorFixedBytes) % constructor.genomeSize]);
    default:
        return false;
    }
}

__inline__ __device__ int ConstructorProcessor::getCellFunctionDataSize(Enums::CellFunction cellFunction, bool makeSelfCopy, int genomeSize)
{
    switch (cellFunction) {
    case Enums::CellFunction_Neuron:
        return NeuronBytes;
    case Enums::CellFunction_Transmitter:
        return TransmitterBytes;
    case Enums::CellFunction_Constructor: {
        return makeSelfCopy ? ConstructorFixedBytes + 1 : ConstructorFixedBytes + 3 + genomeSize;
    }
    case Enums::CellFunction_Sensor:
        return SensorBytes;
    case Enums::CellFunction_Nerve:
        return NerveBytes;
    case Enums::CellFunction_Attacker:
        return AttackerBytes;
    case Enums::CellFunction_Injector: {
        return makeSelfCopy ? InjectorFixedBytes + 1 : InjectorFixedBytes + 3 + genomeSize;
    }
    case Enums::CellFunction_Muscle:
        return MuscleBytes;
    default:
        return 0;
    }
}

template <typename GenomeHolderSource, typename GenomeHolderTarget>
__inline__ __device__ void ConstructorProcessor::copyGenome(SimulationData& data, GenomeHolderSource& source, GenomeHolderTarget& target)
{
    bool makeGenomeCopy = readBool(source);
    if (!makeGenomeCopy) {
        auto size = min(readWord(source), toInt(source.genomeSize));
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
