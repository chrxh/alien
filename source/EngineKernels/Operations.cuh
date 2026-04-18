#pragma once

#include "Base.cuh"

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

struct DelObjectOperation
{
    uint64_t objectIndex;
};

union StructureOperationData
{
    AddConnectionPairOperation addConnection;
    DelConnectionOperation delConnection;
    DelObjectOperation delObject;
};

struct StructuralOperation
{
    enum class Type : int
    {
        AddConnectionPair,
        DelConnection,
        DelObject,
    };
    Type type;
    StructureOperationData data;
    int nextOperationIndex;  //linked list, = -1 end
};

struct CellTypeOperation
{
    Object* object;
};
