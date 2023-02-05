#pragma once

#include "Base.cuh"
#include "Definitions.cuh"

struct AddConnectionOperation {
    bool addTokens;
    Cell* cell;
    Cell* otherCell;
};

struct DelAllConnectionsOperation
{
};

struct DelConnectionOperation
{
    Cell* connectedCell;
};

struct DelCellOperation
{
};

union StructureOperationData
{
    AddConnectionOperation addConnection;
    DelAllConnectionsOperation delAllConnections;
    DelConnectionOperation delConnection;
    DelCellOperation delCell;
};

struct StructuralOperation
{
    enum class Type : int
    {
        None,
        AddConnections,
        DelAllConnections,
        DelConnection,
        DelCell,
    };
    Type type;
    StructureOperationData data;
    int nextOperationIndex; //linked list, = -1 end
};

struct CellFunctionOperation
{
    Cell* cell;
};
