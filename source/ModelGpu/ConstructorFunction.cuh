#pragma once
#include "ModelBasic/ElementaryTypes.h"
#include "SimulationData.cuh"
#include "Math.cuh"
#include "QuantityConverter.cuh"

class ConstructorFunction
{
public:
    __inline__ __device__ static void processing(Token* token);

private:
    __inline__ __device__ static bool checkDistance(float distance);
    __inline__ __device__ static Cell* getConstructionSite(Cell* cell);

    __inline__ __device__ static void continueConstruction(Token* token, Cell* constructionCell);
    __inline__ __device__ static void startNewConstruction();
    
    __inline__ __device__ static void removeConnection(Cell* cell, Cell* otherCell);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__inline__ __device__ void ConstructorFunction::processing(Token * token)
{
    auto const command = token->memory[Enums::Constr::IN] % Enums::ConstrIn::_COUNTER;

    if (Enums::ConstrIn::DO_NOTHING == command) {
        token->memory[Enums::Constr::OUT] = Enums::ConstrOut::SUCCESS;
        return;
    }

    auto const distance = QuantityConverter::convertDataToDistance(token->memory[Enums::Constr::IN_DIST]);
    if (!checkDistance(distance)) {
        token->memory[Enums::Constr::OUT] = Enums::ConstrOut::ERROR_DIST;
    }

    auto const& cell = token->cell;
    auto const constructionSite = getConstructionSite(cell);

    if (constructionSite) {
        continueConstruction(token, constructionSite);
    }
    else {
        startNewConstruction();
    }
}

__inline__ __device__ bool ConstructorFunction::checkDistance(float distance)
{
    return distance > cudaSimulationParameters.cellMaxDistance;
}

__inline__ __device__ Cell * ConstructorFunction::getConstructionSite(Cell * cell)
{
    Cell* result = nullptr;
    for (int i = 0; i < cell->numConnections; ++i) {
        auto const& connectingCell = cell->connections[i];
        if (connectingCell->tokenBlocked) {
            result = connectingCell;
        }
    }
    return result;
}

__inline__ __device__ void ConstructorFunction::continueConstruction(Token * token, Cell * constructionCell)
{
    auto const& cell = token->cell;
    removeConnection(cell, constructionCell);


//    auto const option = token->memory[Enums::Constr::IN_OPTION] % Enums::ConstrInOption::_COUNTER;
}

__inline__ __device__ void ConstructorFunction::startNewConstruction()
{
}

__inline__ __device__ void ConstructorFunction::removeConnection(Cell* cell, Cell* otherCell)
{
    auto remove = [](Cell* cell, Cell* connectionToRemove) {
        bool connectionFound = false;
        for (int i = 0; i < cell->numConnections; ++i) {
            auto& connectingCell = cell->connections[i];
            if (!connectionFound) {
                if (connectingCell == connectionToRemove) {
                    connectionFound = true;
                }
            }
            else {
                cell->connections[i - 1] = connectingCell;
            }
        }
        --cell->numConnections;
    };

    remove(cell, otherCell);
    remove(otherCell, cell);
}

