#pragma once

#include "Base.cuh"
#include "Definitions.cuh"

struct AddConnectionPairOperation
{
    bool addTokens;
    Object* object;
    Object* otherObject;
};

struct DelConnectionOperation
{
    Object* connectedObject;
};

struct DelCellOperation
{
    uint64_t cellIndex;
};

union StructureOperationData
{
    AddConnectionPairOperation addConnection;
    DelConnectionOperation delConnection;
    DelCellOperation delCell;
};

struct StructuralOperation
{
    enum class Type : int
    {
        AddConnectionPair,
        DelConnection,
        DelCell,
    };
    Type type;
    StructureOperationData data;
    int nextOperationIndex;  //linked list, = -1 end
};

struct CellTypeOperation
{
    Object* object;
};
