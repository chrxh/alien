#pragma once

#include "Base.cuh"
#include "Definitions.cuh"

struct AddConnectionPairOperation {
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
    uint64_t cellIndex;
};

union StructureOperationData
{
    AddConnectionPairOperation addConnection;
    DelAllConnectionsOperation delAllConnections;
    DelConnectionOperation delConnection;
    DelCellOperation delCell;
};

struct StructuralOperation
{
    enum class Type : int
    {
        AddConnectionPair,
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
