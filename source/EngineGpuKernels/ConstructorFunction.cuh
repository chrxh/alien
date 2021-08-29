#pragma once

#include "EngineInterface/ElementaryTypes.h"

#include "Math.cuh"
#include "QuantityConverter.cuh"
#include "CellConnectionProcessor.cuh"

class ConstructorFunction
{
public:
    __inline__ __device__ static void processing(Token* token, SimulationData& data);

private:
    struct ConstructionData
    {
        Enums::ConstrIn::Type constrIn;
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
        Token* token, SimulationData& data, ConstructionData& constructionData);
    __inline__ __device__ static void
    continueConstruction(Token* token, SimulationData& data, ConstructionData const& constructionData, Cell* firstConstructedCell);

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

    __inline__ __device__ static void mutateToken(Token* token, SimulationData& data);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/
__inline__ __device__ void
ConstructorFunction::processing(Token* token, SimulationData& data)
{
    mutateToken(token, data);

    ConstructionData constructionData;
    readConstructionData(token, constructionData);

    if (Enums::ConstrIn::DO_NOTHING == constructionData.constrIn) {
        token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::SUCCESS;
        return;
    }

    Cell* firstCellOfConstructionSite = getFirstCellOfConstructionSite(token);

    if (firstCellOfConstructionSite) {
        if (!firstCellOfConstructionSite->tryLock()) {
            token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::ERROR_LOCK;
            return;
        }

        continueConstruction(token, data, constructionData, firstCellOfConstructionSite);
        firstCellOfConstructionSite->releaseLock();
    } else {
        startNewConstruction(token, data, constructionData);
    }
}

__inline__ __device__ void ConstructorFunction::readConstructionData(Token* token, ConstructionData& data)
{
    auto const& memory = token->memory;
    data.constrIn = static_cast<Enums::ConstrIn::Type>(
        static_cast<unsigned char>(token->memory[Enums::Constr::INPUT]) % Enums::ConstrIn::_COUNTER);

    auto option = static_cast<Enums::ConstrInOption::Type>(
        static_cast<unsigned char>(token->memory[Enums::Constr::IN_OPTION]) % Enums::ConstrInOption::_COUNTER);

    data.isConstructToken = Enums::ConstrInOption::CREATE_EMPTY_TOKEN == option
        || Enums::ConstrInOption::CREATE_DUP_TOKEN == option
        || Enums::ConstrInOption::FINISH_WITH_EMPTY_TOKEN_SEP == option
        || Enums::ConstrInOption::FINISH_WITH_DUP_TOKEN_SEP == option;
    data.isDuplicateTokenMemory = (Enums::ConstrInOption::CREATE_DUP_TOKEN == option
                                   || Enums::ConstrInOption::FINISH_WITH_DUP_TOKEN_SEP == option)
        && !cudaSimulationParameters.cellFunctionConstructorOffspringTokenSuppressMemoryCopy;
    data.isFinishConstruction = Enums::ConstrInOption::FINISH_NO_SEP == option
        || Enums::ConstrInOption::FINISH_WITH_SEP == option
        || Enums::ConstrInOption::FINISH_WITH_EMPTY_TOKEN_SEP == option
        || Enums::ConstrInOption::FINISH_WITH_DUP_TOKEN_SEP == option;
    data.isSeparateConstruction = Enums::ConstrInOption::FINISH_WITH_SEP == option
        || Enums::ConstrInOption::FINISH_WITH_EMPTY_TOKEN_SEP == option
        || Enums::ConstrInOption::FINISH_WITH_DUP_TOKEN_SEP == option;

    data.angleAlignment = static_cast<unsigned char>(memory[Enums::Constr::IN_ANGLE_ALIGNMENT]);

    data.uniformDist = static_cast<Enums::ConstrInUniformDist::Type>(
                           static_cast<unsigned char>(token->memory[Enums::Constr::IN_UNIFORM_DIST])
                           % Enums::ConstrInUniformDist::_COUNTER)
            == Enums::ConstrInUniformDist::YES
        ? true
        : false;

    data.angle = memory[Enums::Constr::INOUT_ANGLE];
    data.distance = memory[Enums::Constr::IN_DIST];
    data.maxConnections = memory[Enums::Constr::IN_CELL_MAX_CONNECTIONS];
    data.branchNumber = memory[Enums::Constr::IN_CELL_BRANCH_NO];
    data.metaData = memory[Enums::Constr::IN_CELL_METADATA];
    data.cellFunctionType = memory[Enums::Constr::IN_CELL_FUNCTION];
}

__inline__ __device__ Cell* ConstructorFunction::getFirstCellOfConstructionSite(Token* token)
{
    Cell* result = nullptr;
    auto const& cell = token->cell;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto const& connectingCell = cell->connections[i].cell;
        if (connectingCell->tokenBlocked) {
            result = connectingCell;
        }
    }
    return result;
}

__inline__ __device__ void
ConstructorFunction::startNewConstruction(Token* token, SimulationData& data, ConstructionData& constructionData)
{
    auto const& cell = token->cell;
    auto const adaptMaxConnections = isAdaptMaxConnections(constructionData);

    if (!isConnectable(cell->numConnections, cell->maxConnections, adaptMaxConnections)) {
        token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::ERROR_CONNECTION;
        return;
    }

    auto const anglesForNewConnection = calcAnglesForNewConnection(data, cell, QuantityConverter::convertDataToAngle(constructionData.angle));

    auto const distance = QuantityConverter::convertDataToDistance(token->memory[Enums::Constr::IN_DIST]);
    auto const relPosOfNewCellDelta = Math::unitVectorOfAngle(anglesForNewConnection.angleForCell)
        * cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance;
    float2 posOfNewCell = /*separation
        ? cell->relPos + relPosOfNewCellDelta + Math::unitVectorOfAngle(newCellAngle) * distance : */
        cell->absPos + relPosOfNewCellDelta;

    constructionData.isConstructToken = false;  //not supported
    auto energyForNewEntities = adaptEnergies(token, constructionData);

    if (!energyForNewEntities.energyAvailable) {
        token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::ERROR_NO_ENERGY;
        return;
    }

    Cell* newCell;
    constructCell(data, token, posOfNewCell, energyForNewEntities.cell, constructionData, newCell);

    if (!newCell->tryLock()) {
        token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::ERROR_LOCK;
        return;
    }


    if (!constructionData.isFinishConstruction || !constructionData.isSeparateConstruction) {
        CellConnectionProcessor::addConnectionsForConstructor(
            data,
            cell,
            newCell,
            anglesForNewConnection.angleFromPreviousConnection,
            0,
            cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance);
    }
    if (constructionData.isFinishConstruction) {
        newCell->tokenBlocked = false;
    }
    if (AdaptMaxConnections::Yes == adaptMaxConnections) {
        cell->maxConnections = cell->numConnections;
        newCell->maxConnections = newCell->numConnections;
    }

    newCell->releaseLock();

    token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::SUCCESS;
    token->memory[Enums::Constr::INOUT_ANGLE] = 0;
}

__inline__ __device__ void ConstructorFunction::continueConstruction(
    Token* token,
    SimulationData& data,
    ConstructionData const& constructionData,
    Cell* firstConstructedCell)
{
    auto cell = token->cell;
    auto posDelta = firstConstructedCell->absPos - cell->absPos;
    data.cellMap.mapDisplacementCorrection(posDelta);

    auto desiredDistance = QuantityConverter::convertDataToDistance(constructionData.distance);
    posDelta =
        Math::normalized(posDelta) * (cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance - desiredDistance);
/*
    printf("distance: %f\n", desiredDistance);
*/

    if (Math::length(posDelta) <= cudaSimulationParameters.cellMinDistance
        || cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance - desiredDistance < 0) {
        token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::ERROR_DIST;
        return;
    }
    auto adaptMaxConnections = isAdaptMaxConnections(constructionData);
    if (AdaptMaxConnections::No == adaptMaxConnections && 1 == constructionData.maxConnections) {
        token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::ERROR_CONNECTION;
        return;
    }

    auto energyForNewEntities = adaptEnergies(token, constructionData);
    if (!energyForNewEntities.energyAvailable) {
        token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::ERROR_NO_ENERGY;
        return;
    }

    Cell* newCell;
    auto posOfNewCell = cell->absPos + posDelta;
    constructCell(data, token, posOfNewCell, energyForNewEntities.cell, constructionData, newCell);
    firstConstructedCell->tokenBlocked = false;

    if (!newCell->tryLock()) {
        cell->energy +=
            energyForNewEntities.token;  //token could not be constructed anymore => transfer energy back to cell
        token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::ERROR_LOCK;
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
    CellConnectionProcessor::delConnectionsForConstructor(cell, firstConstructedCell);
    if (!constructionData.isFinishConstruction || !constructionData.isSeparateConstruction) {
        CellConnectionProcessor::addConnectionsForConstructor(
            data,
            cell,
            newCell,
            angleFromPreviousForCell,
            0,
            cudaSimulationParameters.cellFunctionConstructorOffspringCellDistance);
    }
    auto angleFromPreviousForNewCell = QuantityConverter::convertDataToAngle(constructionData.angle) + 180.0f;
    CellConnectionProcessor::addConnectionsForConstructor(
        data,
        newCell,
        firstConstructedCell,
        angleFromPreviousForNewCell,
        angleFromPreviousForFirstConstructedCell,
        desiredDistance/*,
        constructionData.angleAlignment*/);

    if (constructionData.isFinishConstruction) {
        newCell->tokenBlocked = false;
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
        data.cellMap.mapDisplacementCorrection(otherPosDelta);
        Math::normalize(otherPosDelta);
        if (Math::dot(posDelta, otherPosDelta) < 0.1) {
            continue;
        }
        if (otherCell->tryLock()) {
            if (isConnectable(newCell->numConnections, newCell->maxConnections, adaptMaxConnections)
                && isConnectable(otherCell->numConnections, otherCell->maxConnections, adaptMaxConnections)) {

                auto distance = constructionData.uniformDist ? desiredDistance : Math::length(otherPosDelta);
                CellConnectionProcessor::addConnectionsForConstructor(
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

    token->memory[Enums::Constr::OUTPUT] = Enums::ConstrOut::SUCCESS;
}

__inline__ __device__ void ConstructorFunction::constructCell(
    SimulationData& data,
    Token* token,
    float2 const& posOfNewCell,
    float const energyOfNewCell,
    ConstructionData const& constructionData,
    Cell*& result)
{
    EntityFactory factory;
    factory.init(&data);
    result = factory.createCell();
    result->energy = energyOfNewCell;
    result->absPos = posOfNewCell;
    data.cellMap.mapPosCorrection(result->absPos);
    result->maxConnections = getMaxConnections(constructionData);
    result->numConnections = 0;
    result->branchNumber = static_cast<unsigned char>(constructionData.branchNumber)
        % cudaSimulationParameters.cellMaxTokenBranchNumber;
    result->tokenBlocked = true;
    result->cellFunctionType = constructionData.cellFunctionType;
    result->numStaticBytes = static_cast<unsigned char>(token->memory[Enums::Constr::IN_CELL_FUNCTION_DATA])
        % (MAX_CELL_STATIC_BYTES + 1);
    auto offset = result->numStaticBytes + 1;
    result->numMutableBytes =
        static_cast<unsigned char>(
            token->memory[(Enums::Constr::IN_CELL_FUNCTION_DATA + offset) % MAX_TOKEN_MEM_SIZE])
        % (MAX_CELL_MUTABLE_BYTES + 1);
    result->metadata.color = constructionData.metaData;

    for (int i = 0; i < result->numStaticBytes; ++i) {
        result->staticData[i] = token->memory[(Enums::Constr::IN_CELL_FUNCTION_DATA + i + 1) % MAX_TOKEN_MEM_SIZE];
    }
    for (int i = 0; i <= result->numMutableBytes; ++i) {
        result->mutableData[i] =
            token->memory[(Enums::Constr::IN_CELL_FUNCTION_DATA + offset + i + 1) % MAX_TOKEN_MEM_SIZE];
    }
}

__inline__ __device__ auto ConstructorFunction::isAdaptMaxConnections(ConstructionData const& data)
    -> AdaptMaxConnections
{
    return 0 == getMaxConnections(data) ? AdaptMaxConnections::Yes : AdaptMaxConnections::No;
}

__inline__ __device__ int ConstructorFunction::getMaxConnections(ConstructionData const& data)
{
    return static_cast<unsigned char>(data.maxConnections) % (cudaSimulationParameters.cellMaxBonds + 1);
}

__inline__ __device__ bool
ConstructorFunction::isConnectable(int numConnections, int maxConnections, AdaptMaxConnections adaptMaxConnections)
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

__inline__ __device__ auto ConstructorFunction::calcAnglesForNewConnection(
    SimulationData& data,
    Cell* cell,
    float angleDeviation)
    -> AnglesForNewConnection
{
    if (0 == cell->numConnections) {
        return AnglesForNewConnection{0, 0};
    }
    auto displacement = cell->connections[0].cell->absPos - cell->absPos;
    data.cellMap.mapDisplacementCorrection(displacement);
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

__inline__ __device__ auto ConstructorFunction::adaptEnergies(Token* token, ConstructionData const& data)
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

__inline__ __device__ Token* ConstructorFunction::constructToken(
    SimulationData& data,
    Cell* cell,
    Token* token,
    Cell* sourceCell,
    float energy,
    bool duplicateMemory)
{
    EntityFactory factory;
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

__inline__ __device__ void ConstructorFunction::mutateToken(Token* token, SimulationData& data)
{
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorTokenDataMutationProb) {
        auto index = data.numberGen.random(255) % cudaSimulationParameters.tokenMemorySize;
        token->memory[index] = data.numberGen.random(255);
    }
}


/*
__inline__ __device__ void ConstructorFunction::mutateConstructionData(
    SimulationData& data,
    ConstructionData& constructionData)
{
    if (data.numberGen.random() < cudaSimulationParameters.cellFunctionConstructorCellPropertyMutationProb) {
        constructionData.constrIn = static_cast<Enums::ConstrIn::Type>(
            static_cast<unsigned char>(data.numberGen.random(255)) % Enums::ConstrIn::_COUNTER);
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
