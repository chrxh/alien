#pragma once

#include "EngineInterface/Enums.h"

#include "QuantityConverter.cuh"
#include "CellFunctionProcessor.cuh"
#include "SimulationResult.cuh"

class RibosomeProcessor
{
public:
    __inline__ __device__ static void process(SimulationData& data, SimulationResult& result);

private:
    __inline__ __device__ static void processCell(SimulationData& data, SimulationResult& result, Cell* cell);
    __inline__ __device__ static bool isConstructionFinished(Cell* cell);
    __inline__ __device__ static bool isConstructionPossible(Cell* cell, Activity const& activity);

    struct ConstructionData
    {
        float angle;
        float distance;
        int maxConnections;
        int executionOrderNumber;
        int color;
        Enums::CellFunction cellFunction;
    };
    __inline__ __device__ static ConstructionData readConstructionData(Cell* cell, bool& finished);

    __inline__ __device__ static bool
    tryConstructCell(SimulationData& data, SimulationResult& result, Cell* hostCell, ConstructionData const& constructionData, bool& finished);

    __inline__ __device__ static Cell* getFirstCellOfConstructionSite(Cell* hostCell);
    __inline__ __device__ static bool
    startNewConstruction(SimulationData& data, SimulationResult& result, Cell* hostCell, ConstructionData const& constructionData, bool& finished);
    __inline__ __device__ static bool continueConstruction(
        SimulationData& data,
        SimulationResult& result,
        Cell* hostCell,
        Cell* firstConstructedCell,
        ConstructionData const& constructionData,
        bool& finished);

    __inline__ __device__ static bool isConnectable(int numConnections, int maxConnections, bool makeSticky);

    struct AnglesForNewConnection
    {
        float angleFromPreviousConnection;
        float angleForCell;
    };
    __inline__ __device__ static AnglesForNewConnection calcAnglesForNewConnection(SimulationData& data, Cell* cell, float angleDeviation);

    __inline__ __device__ static Cell* constructCellIntern(
        SimulationData& data,
        float2 const& posOfNewCell,
        float const energyOfNewCell,
        ConstructionData const& constructionData,
        bool& finished);

    __inline__ __device__ static bool readBool(uint64_t dataSize, uint8_t* data, uint64_t& index, bool& finish);
    __inline__ __device__ static int readByte(uint64_t dataSize, uint8_t* data, uint64_t& index, bool& finish);
    __inline__ __device__ static int readWord(uint64_t dataSize, uint8_t* data, uint64_t& index, bool& finish);
    __inline__ __device__ static float readFloat(uint64_t dataSize, uint8_t* data, uint64_t& index, bool& finish);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

namespace
{
    float constexpr offspringCellDistance = 1.6f;
}

__inline__ __device__ void RibosomeProcessor::process(SimulationData& data, SimulationResult& result)
{
    auto partition = calcAllThreadsPartition(data.cellFunctionOperations[Enums::CellFunction_Nerve].getNumEntries());
    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto operation = data.cellFunctionOperations[Enums::CellFunction_Nerve].at(i);
        auto cell = operation.cell;
        processCell(data, result, cell);
    }
}

__inline__ __device__ void RibosomeProcessor::processCell(SimulationData& data, SimulationResult& result, Cell* cell)
{
    auto activity = CellFunctionProcessor::calcInputActivity(cell);
    if (!isConstructionFinished(cell)) {
        if (isConstructionPossible(cell, activity)) {
            auto finished = false;
            auto constructionData = readConstructionData(cell, finished);
            if (tryConstructCell(data, result, cell, constructionData, finished)) {
                activity.channels[0] = 1;
            } else {
                activity.channels[0] = 0;
            }
            if (finished) {
                auto& ribosome = cell->cellFunctionData.ribosome;
                if (!ribosome.singleConstruction) {
                    ribosome.currentGenomePos = 0;
                }
            }
        } else {
            activity.channels[0] = 0;
        }
    }
    CellFunctionProcessor::setActivity(cell, activity);
}

__inline__ __device__ bool RibosomeProcessor::isConstructionFinished(Cell* cell)
{
    return cell->cellFunctionData.ribosome.currentGenomePos >= cell->cellFunctionData.ribosome.genomeSize;
}

__inline__ __device__ bool RibosomeProcessor::isConstructionPossible(Cell* cell, Activity const& activity)
{
    if (cell->energy < cudaSimulationParameters.cellNormalEnergy * 2) {
        return false;
    }
    if (cell->cellFunctionData.ribosome.mode == Enums::ConstructionMode_Manual && abs(activity.channels[0]) < 0.25f) {
        return false;
    }
    return true;
}

__inline__ __device__ RibosomeProcessor::ConstructionData RibosomeProcessor::readConstructionData(Cell* cell, bool& finished)
{
    auto& ribosome = cell->cellFunctionData.ribosome;

    ConstructionData result;
    result.angle = readFloat(ribosome.genomeSize, ribosome.genome, ribosome.currentGenomePos, finished);
    result.distance = readFloat(ribosome.genomeSize, ribosome.genome, ribosome.currentGenomePos, finished);
    result.maxConnections =
        readByte(ribosome.genomeSize, ribosome.genome, ribosome.currentGenomePos, finished) % (cudaSimulationParameters.cellMaxBonds + 1);
    result.executionOrderNumber = readByte(ribosome.genomeSize, ribosome.genome, ribosome.currentGenomePos, finished)
        % cudaSimulationParameters.cellMaxExecutionOrderNumber;
    result.color = readByte(ribosome.genomeSize, ribosome.genome, ribosome.currentGenomePos, finished) % 7;
    result.cellFunction = readByte(ribosome.genomeSize, ribosome.genome, ribosome.currentGenomePos, finished) % Enums::CellFunction_Count;
    return result;
}

__inline__ __device__ bool
RibosomeProcessor::tryConstructCell(SimulationData& data, SimulationResult& result, Cell* hostCell, ConstructionData const& constructionData, bool& finished)
{
    Cell* underConstructionCell = getFirstCellOfConstructionSite(hostCell);
    if (underConstructionCell) {
        if (!underConstructionCell->tryLock()) {
            return false;
        }

        auto success = continueConstruction(data, result, hostCell, underConstructionCell, constructionData, finished);
        underConstructionCell->releaseLock();

        return success;
    } else {

        return startNewConstruction(data, result, hostCell, constructionData, finished);
    }
}

__inline__ __device__ Cell* RibosomeProcessor::getFirstCellOfConstructionSite(Cell* hostCell)
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

__inline__ __device__ bool RibosomeProcessor::startNewConstruction(
    SimulationData& data,
    SimulationResult& result,
    Cell* hostCell,
    ConstructionData const& constructionData,
    bool& finished)
{
    auto makeSticky = hostCell->cellFunctionData.ribosome.makeSticky;

    if (!isConnectable(hostCell->numConnections, hostCell->maxConnections, makeSticky)) {
        return false;
    }

    auto anglesForNewConnection = calcAnglesForNewConnection(data, hostCell, constructionData.angle);

    auto newCellDirection = Math::unitVectorOfAngle(anglesForNewConnection.angleForCell) * offspringCellDistance;
    float2 newCellPos = hostCell->absPos + newCellDirection;

    Cell* newCell = constructCellIntern(data, newCellPos, cudaSimulationParameters.cellNormalEnergy, constructionData, finished);
    hostCell->energy -= cudaSimulationParameters.cellNormalEnergy;

    if (!newCell->tryLock()) {
        return false;
    }

    if (!finished|| !hostCell->cellFunctionData.ribosome.separateConstruction) {
        CellConnectionProcessor::addConnections(
            data,
            hostCell,
            newCell,
            anglesForNewConnection.angleFromPreviousConnection,
            0,
            offspringCellDistance);
    }
    if (finished) {
        newCell->underConstruction = false;
    }
    if (!makeSticky) {
        hostCell->maxConnections = hostCell->numConnections;
        newCell->maxConnections = newCell->numConnections;
    }

    newCell->releaseLock();

    result.incCreatedCell();
}

__inline__ __device__ bool RibosomeProcessor::continueConstruction(
    SimulationData& data,
    SimulationResult& result,
    Cell* hostCell,
    Cell* underConstructionCell,
    ConstructionData const& constructionData,
    bool& finished)
{
}

__inline__ __device__ bool RibosomeProcessor::isConnectable(int numConnections, int maxConnections, bool makeSticky)
{
    if (!makeSticky) {
        if (numConnections >= cudaSimulationParameters.cellMaxBonds) {
            return false;
        }
    }
    if (makeSticky) {
        if (numConnections >= maxConnections) {
            return false;
        }
    }
    return true;
}

__inline__ __device__ auto RibosomeProcessor::calcAnglesForNewConnection(SimulationData& data, Cell* cell, float angleDeviation) -> AnglesForNewConnection
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
RibosomeProcessor::constructCellIntern(
    SimulationData& data,
    float2 const& posOfNewCell,
    float const energyOfNewCell,
    ConstructionData const& constructionData,
    bool& finished)
{
    ObjectFactory factory;
    factory.init(&data);

    Cell * result = factory.createCell();
    result->energy = energyOfNewCell;
    result->absPos = posOfNewCell;
    data.cellMap.correctPosition(result->absPos);
    result->maxConnections = constructionData.maxConnections;
    result->numConnections = 0;
    result->executionOrderNumber = constructionData.executionOrderNumber;
    result->underConstruction = true;
    result->cellFunction = constructionData.cellFunction;
    result->color = constructionData.color;

    //#TODO cellFunctionData

    return result;
}

__inline__ __device__ bool RibosomeProcessor::readBool(uint64_t dataSize, uint8_t* data, uint64_t& index, bool& finish)
{
    return true;
}

__inline__ __device__ int RibosomeProcessor::readByte(uint64_t dataSize, uint8_t* data, uint64_t& index, bool& finish)
{
    return 0;
}

__inline__ __device__ int RibosomeProcessor::readWord(uint64_t dataSize, uint8_t* data, uint64_t& index, bool& finish)
{
    return 0;
}

__inline__ __device__ float RibosomeProcessor::readFloat(uint64_t dataSize, uint8_t* data, uint64_t& index, bool& finish)
{
    return false;
}

/*
class RibosomeProcessor
{
public:
    __inline__ __device__ static void process(Token* token, SimulationData& data, SimulationResult& result);

private:
    struct ConstructionData
    {
        Enums::ConstrIn constrIn;
        bool isConstructToken;
        bool isDuplicateTokenMemory;
        bool isFinishConstruction;
        bool isSeparateConstruction;
        int angleAlignment;
        bool uniformDist;
        char angle;
        char distance;
        char maxConnections;
        char branchNumber;
        char metaData;
        char cellFunctionType;
    };
    __inline__ __device__ static void readConstructionData(Token* token, ConstructionData& data);

    __inline__ __device__ static Cell* getFirstCellOfConstructionSite(Token* token);
    __inline__ __device__ static void startNewConstruction(
        Token* token,
        SimulationData& data,
        SimulationResult& result,
        ConstructionData& constructionData);
    __inline__ __device__ static void continueConstruction(
        Token* token,
        SimulationData& data,
        SimulationResult& result,
        ConstructionData const& constructionData,
        Cell* firstConstructedCell);

    __inline__ __device__ static void constructCell(
        SimulationData& data,
        Token* token,
        float2 const& posOfNewCell,
        float const energyOfNewCell,
        ConstructionData const& constructionData,
        Cell*& result);

    enum class AdaptMaxConnections
    {
        No,
        Yes
    };
    __inline__ __device__ static AdaptMaxConnections isAdaptMaxConnections(
        ConstructionData const& data);

    __inline__ __device__ static int getMaxConnections(ConstructionData const& data);

    __inline__ __device__ static bool
    isConnectable(int numConnections, int maxConnections, AdaptMaxConnections adaptMaxConnections);

    struct AnglesForNewConnection
    {
        float angleFromPreviousConnection;
        float angleForCell;
    };
    __inline__ __device__ static AnglesForNewConnection
    calcAnglesForNewConnection(SimulationData& data, Cell* cell, float angleDeviation);

    struct EnergyForNewEntities
    {
        bool energyAvailable;
        float cell;
        float token;
    };
    __inline__ __device__ static EnergyForNewEntities adaptEnergies(Token* token, ConstructionData const& data);

    __inline__ __device__ static Token* constructToken(
        SimulationData& data,
        Cell* cell,
        Token* token,
        Cell* sourceCell,
        float energy,
        bool duplicateMemory);

//    __inline__ __device__ static void mutateToken(Token* token, SimulationData& data);
};
*/
/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
/* __inline__ __device__ void RibosomeProcessor::process(Token* token, SimulationData& data, SimulationResult& result)
{
    //    mutateToken(token, data);

    ConstructionData constructionData;
    readConstructionData(token, constructionData);

    if (Enums::ConstrIn_DoNothing == constructionData.constrIn) {
        token->memory[Enums::Constr_Output] = Enums::ConstrOut_Success;
        return;
    }

    Cell* firstCellOfConstructionSite = getFirstCellOfConstructionSite(token);

    if (firstCellOfConstructionSite) {
        if (!firstCellOfConstructionSite->tryLock()) {
            token->memory[Enums::Constr_Output] = Enums::ConstrOut_ErrorLock;
            return;
        }

        continueConstruction(token, data, result, constructionData, firstCellOfConstructionSite);
        firstCellOfConstructionSite->releaseLock();
    } else {
        startNewConstruction(token, data, result, constructionData);
    }
}

__inline__ __device__ void RibosomeProcessor::readConstructionData(Token* token, ConstructionData& data)
{
    auto const& memory = token->memory;
    data.constrIn = static_cast<unsigned char>(token->memory[Enums::Constr_Input]) % Enums::ConstrIn_Count;

    auto option = static_cast<unsigned char>(token->memory[Enums::Constr_InOption]) % Enums::ConstrInOption_Count;

    data.isConstructToken = Enums::ConstrInOption_CreateEmptyToken == option || Enums::ConstrInOption_CreateDupToken == option
        || Enums::ConstrInOption_FinishWithEmptyTokenSep == option || Enums::ConstrInOption_FinishWithDupTokenSep == option;
    data.isDuplicateTokenMemory = (Enums::ConstrInOption_CreateDupToken == option || Enums::ConstrInOption_FinishWithDupTokenSep == option)
        && !cudaSimulationParameters.cellFunctionConstructorOffspringTokenSuppressMemoryCopy;
    data.isFinishConstruction = Enums::ConstrInOption_FinishNoSep == option || Enums::ConstrInOption_FinishWithSep == option
        || Enums::ConstrInOption_FinishWithEmptyTokenSep == option || Enums::ConstrInOption_FinishWithDupTokenSep == option;
    data.isSeparateConstruction = Enums::ConstrInOption_FinishWithSep == option || Enums::ConstrInOption_FinishWithEmptyTokenSep == option
        || Enums::ConstrInOption_FinishWithDupTokenSep == option;

    data.angleAlignment = static_cast<unsigned char>(memory[Enums::Constr_InAngleAlignment]);

    data.uniformDist =
        (static_cast<unsigned char>(token->memory[Enums::Constr_InUniformDist]) % Enums::ConstrInUniformDist_Count) == Enums::ConstrInUniformDist_Yes ? true
                                                                                                                                                      : false;

    data.angle = memory[Enums::Constr_InOutAngle];
    data.distance = memory[Enums::Constr_InDist];
    data.maxConnections = memory[Enums::Constr_InCellMaxConnections];
    data.branchNumber = memory[Enums::Constr_InCellBranchNumber];
    data.metaData =
        cudaSimulationParameters.cellFunctionConstructorOffspringInheritColor ? calcMod(token->cell->metadata.color, 7) : memory[Enums::Constr_InCellColor];
    data.cellFunctionType = memory[Enums::Constr_InCellFunction];
}

__inline__ __device__ Cell* RibosomeProcessor::getFirstCellOfConstructionSite(Token* token)
{
    Cell* result = nullptr;
    auto const& cell = token->cell;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto const& connectingCell = cell->connections[i].cell;
        if (connectingCell->underConstruction) {
            result = connectingCell;
        }
    }
    return result;
}

__inline__ __device__ void RibosomeProcessor::startNewConstruction(
    Token* token,
    SimulationData& data,
    SimulationResult& result,
    ConstructionData& constructionData)
{
    auto const& cell = token->cell;
    auto const adaptMaxConnections = isAdaptMaxConnections(constructionData);

    if (!isConnectable(cell->numConnections, cell->maxConnections, adaptMaxConnections)) {
        token->memory[Enums::Constr_Output] = Enums::ConstrOut_ErrorConnection;
        return;
    }

    auto const anglesForNewConnection = calcAnglesForNewConnection(data, cell, QuantityConverter::convertDataToAngle(constructionData.angle));

    auto const distance = QuantityConverter::convertDataToDistance(token->memory[Enums::Constr_InDist]);
    auto const relPosOfNewCellDelta = Math::unitVectorOfAngle(anglesForNewConnection.angleForCell)
        * cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance;
    float2 posOfNewCell = cell->absPos + relPosOfNewCellDelta;

    constructionData.isConstructToken = false;  //not supported
    auto energyForNewEntities = adaptEnergies(token, constructionData);

    if (!energyForNewEntities.energyAvailable) {
        token->memory[Enums::Constr_Output] = Enums::ConstrOut_ErrorNoEnergy;
        return;
    }

    Cell* newCell;
    constructCell(data, token, posOfNewCell, energyForNewEntities.cell, constructionData, newCell);

    if (!newCell->tryLock()) {
        token->memory[Enums::Constr_Output] = Enums::ConstrOut_ErrorLock;
        return;
    }


    if (!constructionData.isFinishConstruction || !constructionData.isSeparateConstruction) {
        CellConnectionProcessor::addConnections(
            data,
            cell,
            newCell,
            anglesForNewConnection.angleFromPreviousConnection,
            0,
            cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance);
    }
    if (constructionData.isFinishConstruction) {
        newCell->underConstruction = false;
    }
    if (AdaptMaxConnections::Yes == adaptMaxConnections) {
        cell->maxConnections = cell->numConnections;
        newCell->maxConnections = newCell->numConnections;
    }

    newCell->releaseLock();

    token->memory[Enums::Constr_Output] = Enums::ConstrOut_Success;
    token->memory[Enums::Constr_InOutAngle] = 0;
    result.incCreatedCell();
}

__inline__ __device__ void RibosomeProcessor::continueConstruction(
    Token* token,
    SimulationData& data,
    SimulationResult& result,
    ConstructionData const& constructionData,
    Cell* firstConstructedCell)
{
    auto cell = token->cell;
    auto posDelta = firstConstructedCell->absPos - cell->absPos;
    data.cellMap.correctDirection(posDelta);

    auto desiredDistance = QuantityConverter::convertDataToDistance(constructionData.distance);
    posDelta =
        Math::normalized(posDelta) * (cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance - desiredDistance);

    if (Math::length(posDelta) <= cudaSimulationParameters.cellMinDistance
        || cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance - desiredDistance < 0) {
        token->memory[Enums::Constr_Output] = Enums::ConstrOut_ErrorDist;
        return;
    }
    auto adaptMaxConnections = isAdaptMaxConnections(constructionData);
    if (AdaptMaxConnections::No == adaptMaxConnections && 1 == constructionData.maxConnections) {
        token->memory[Enums::Constr_Output] = Enums::ConstrOut_ErrorConnection;
        return;
    }

    auto energyForNewEntities = adaptEnergies(token, constructionData);
    if (!energyForNewEntities.energyAvailable) {
        token->memory[Enums::Constr_Output] = Enums::ConstrOut_ErrorNoEnergy;
        return;
    }

    Cell* newCell;
    auto posOfNewCell = cell->absPos + posDelta;
    constructCell(data, token, posOfNewCell, energyForNewEntities.cell, constructionData, newCell);
    firstConstructedCell->underConstruction = false;

    if (!newCell->tryLock()) {
        cell->energy +=
            energyForNewEntities.token;  //token could not be constructed anymore => transfer energy back to cell
        token->memory[Enums::Constr_Output] = Enums::ConstrOut_ErrorLock;
        return;
    }

    if (constructionData.isConstructToken) {
        constructToken(data, newCell, token, cell, energyForNewEntities.token, constructionData.isDuplicateTokenMemory);
    }

    float angleFromPreviousForCell;
    for (int i = 0; i < cell->numConnections; ++i) {
        if (cell->connections[i].cell == firstConstructedCell) {
            angleFromPreviousForCell = cell->connections[i].angleFromPrevious;
            break;
        }
    }

    float angleFromPreviousForFirstConstructedCell;
    for (int i = 0; i < firstConstructedCell->numConnections; ++i) {
        if (firstConstructedCell->connections[i].cell == cell) {
            angleFromPreviousForFirstConstructedCell = firstConstructedCell->connections[i].angleFromPrevious;
            break;
        }
    }
    CellConnectionProcessor::delConnections(cell, firstConstructedCell);
    if (!constructionData.isFinishConstruction || !constructionData.isSeparateConstruction) {
        CellConnectionProcessor::addConnections(
            data,
            cell,
            newCell,
            angleFromPreviousForCell,
            0,
            cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance);
    }
    auto angleFromPreviousForNewCell = QuantityConverter::convertDataToAngle(constructionData.angle) + 180.0f;
    CellConnectionProcessor::addConnections(
        data,
        newCell,
        firstConstructedCell,
        angleFromPreviousForNewCell,
        angleFromPreviousForFirstConstructedCell,
        desiredDistance);

    if (constructionData.isFinishConstruction) {
        newCell->underConstruction = false;
    }

    Math::normalize(posDelta);
    Math::rotateQuarterClockwise(posDelta);
    Cell* otherCells[18];
    int numOtherCells;
    data.cellMap.get(
        otherCells,
        18,
        numOtherCells,
        posOfNewCell,
        cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance);
    for (int i = 0; i < numOtherCells; ++i) {
        Cell* otherCell = otherCells[i];
        if (otherCell == firstConstructedCell) {
            continue;
        }
        if (otherCell == cell) {
            continue;
        }

        bool connected = false;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto const& connectedCell = cell->connections[i].cell;
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
            if (isConnectable(newCell->numConnections, newCell->maxConnections, adaptMaxConnections)
                && isConnectable(otherCell->numConnections, otherCell->maxConnections, adaptMaxConnections)) {

                auto distance = constructionData.uniformDist ? desiredDistance : Math::length(otherPosDelta);
                CellConnectionProcessor::addConnections(
                    data, newCell, otherCell, 0, 0, distance, constructionData.angleAlignment);
            }
            otherCell->releaseLock();
        }
    }

    if (AdaptMaxConnections::Yes == adaptMaxConnections) {
        cell->maxConnections = cell->numConnections;
        newCell->maxConnections = newCell->numConnections;
    }

    newCell->releaseLock();

    token->memory[Enums::Constr_Output] = Enums::ConstrOut_Success;
    result.incCreatedCell();
}

__inline__ __device__ void RibosomeProcessor::constructCell(
    SimulationData& data,
    Token* token,
    float2 const& posOfNewCell,
    float const energyOfNewCell,
    ConstructionData const& constructionData,
    Cell*& result)
{
    ObjectFactory factory;
    factory.init(&data);
    result = factory.createCell();
    result->energy = energyOfNewCell;
    result->absPos = posOfNewCell;
    data.cellMap.correctPosition(result->absPos);
    result->maxConnections = getMaxConnections(constructionData);
    result->numConnections = 0;
    result->executionOrderNumber = static_cast<unsigned char>(constructionData.branchNumber)
        % cudaSimulationParameters.cellMaxExecutionOrderNumber;
    result->underConstruction = true;
    result->cellFunctionType = constructionData.cellFunctionType;
    result->metadata.color = constructionData.metaData;
    if (result->getCellFunctionType() != Enums::CellFunction_Neuron) {

        //encoding to support older versions
        auto len = min(48 - 1, static_cast<unsigned char>(token->memory[Enums::Constr_InCellFunctionData]));
        result->staticData[0] = len / 3;
        for (int i = 0; i < len; ++i) {
            result->staticData[i + 1] = token->memory[Enums::Constr_InCellFunctionData + i + 1];
        }
        for (int i = 0; i < 8; ++i) {
            result->mutableData[i] = token->memory[Enums::Constr_InCellFunctionData + 2 + len + i];
        }
    } else {

        //new encoding
        for (int i = 0; i < MAX_CELL_STATIC_BYTES; ++i) {
            result->staticData[i] = token->memory[(Enums::Constr_InCellFunctionData + i) % MAX_TOKEN_MEM_SIZE];
        }
        for (int i = 0; i < MAX_CELL_MUTABLE_BYTES; ++i) {
            result->mutableData[i] = token->memory[(Enums::Constr_InCellFunctionData + MAX_CELL_STATIC_BYTES + i) % MAX_TOKEN_MEM_SIZE];
        }
    }
}

__inline__ __device__ auto RibosomeProcessor::isAdaptMaxConnections(ConstructionData const& data)
    -> AdaptMaxConnections
{
    return 0 == getMaxConnections(data) ? AdaptMaxConnections::Yes : AdaptMaxConnections::No;
}

__inline__ __device__ int RibosomeProcessor::getMaxConnections(ConstructionData const& data)
{
    return static_cast<unsigned char>(data.maxConnections) % (cudaSimulationParameters.cellMaxBonds + 1);
}

__inline__ __device__ bool
RibosomeProcessor::isConnectable(int numConnections, int maxConnections, AdaptMaxConnections adaptMaxConnections)
{
    if (AdaptMaxConnections::Yes == adaptMaxConnections) {
        if (numConnections >= cudaSimulationParameters.cellMaxBonds) {
            return false;
        }
    }
    if (AdaptMaxConnections::No == adaptMaxConnections) {
        if (numConnections >= maxConnections) {
            return false;
        }
    }
    return true;
}

__inline__ __device__ auto RibosomeProcessor::calcAnglesForNewConnection(
    SimulationData& data,
    Cell* cell,
    float angleDeviation)
    -> AnglesForNewConnection
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

__inline__ __device__ auto RibosomeProcessor::adaptEnergies(Token* token, ConstructionData const& data)
    -> EnergyForNewEntities
{
    auto const& cell = token->cell;

    EnergyForNewEntities result;
    result.energyAvailable = true;
    result.cell = cudaSimulationParameters.cellFunctionConstructorOffspringCellEnergy;
    result.token = data.isConstructToken ? cudaSimulationParameters.cellFunctionConstructorOffspringTokenEnergy : 0.0f;

    if (token->energy <= result.cell + result.token + cudaSimulationParameters.tokenMinEnergy) {
        result.energyAvailable = false;
        return result;
    }

    token->energy -= (result.cell + result.token);
    if (data.isConstructToken) {
        auto const averageEnergy = (cell->energy + result.cell) / 2;
        cell->energy = averageEnergy;
        result.cell = averageEnergy;
    }

    return result;
}

__inline__ __device__ Token* RibosomeProcessor::constructToken(
    SimulationData& data,
    Cell* cell,
    Token* token,
    Cell* sourceCell,
    float energy,
    bool duplicateMemory)
{
    ObjectFactory factory;
    factory.init(&data);

    Token* result;
    if (duplicateMemory) {
        result = factory.duplicateToken(cell, token);
    } else {
        result = factory.createToken(cell, sourceCell);
    }
    result->energy = energy;
    return result;
}
*/
/*
__inline__ __device__ void ConstructorFunction::mutateToken(Token* token, SimulationData& data)
{
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorTokenDataMutationProb) {
        auto index = data.numberGen.random(255) % cudaSimulationParameters.tokenMemorySize;
        token->memory[index] = data.numberGen.random(255);
    }
    if (data.numberGen.random() < 0.01) {
        token->memory[Enums::Constr_InCellMetadata] = data.numberGen.random(255);
    }
}
*/


/*
__inline__ __device__ void ConstructorFunction::mutateConstructionData(
    SimulationData& data,
    ConstructionData& constructionData)
{
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellPropertyMutationProb) {
        constructionData.constrIn = static_cast<Enums::ConstrIn>(
            static_cast<unsigned char>(data.numberGen.random(255)) % Enums::ConstrIn_Count);
    }
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellStructureMutationProb) {
        constructionData.angle = data.numberGen.random(255);
    }
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellStructureMutationProb) {
        constructionData.distance = data.numberGen.random(255);
    }
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellPropertyMutationProb) {
        constructionData.maxConnections = data.numberGen.random(255);
    }
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellPropertyMutationProb) {
        constructionData.branchNumber = data.numberGen.random(255);
    }
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellPropertyMutationProb) {
        constructionData.metaData = data.numberGen.random(255);
    }
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellPropertyMutationProb) {
        constructionData.cellFunctionType = data.numberGen.random(255);
    }
}

__inline__ __device__ void ConstructorFunction::mutateCellFunctionData(SimulationData& data, Cell* cell)
{
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellDataMutationProb) {
        cell->numStaticBytes = data.numberGen.random(MAX_CELL_STATIC_BYTES);
    }

    for (int i = 0; i <= MAX_CELL_STATIC_BYTES; ++i) {
        if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellDataMutationProb) {
            cell->staticData[i] = data.numberGen.random(255);
        }
    }

    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellDataMutationProb) {
        cell->numMutableBytes = data.numberGen.random(MAX_CELL_MUTABLE_BYTES);
    }

    for (int i = 0; i <= MAX_CELL_MUTABLE_BYTES; ++i) {
        if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellDataMutationProb) {
            cell->mutableData[i] = data.numberGen.random(255);
        }
    }
}

__inline__ __device__ void ConstructorFunction::mutateDuplicatedToken(SimulationData& data, Token* token)
{
    auto const memoryPartition = calcPartition(MAX_TOKEN_MEM_SIZE, threadIdx.x, blockDim.x);
    for (auto index = memoryPartition.startIndex; index <= memoryPartition.endIndex; ++index) {
        if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorTokenDataMutationProb) {
            token->memory[index] = data.numberGen.random(255);
        }
    }
}
*/
