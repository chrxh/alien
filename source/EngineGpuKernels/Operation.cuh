#pragma once

#include "Base.cuh"
#include "Definitions.cuh"

struct AddConnectionOperation {
    Cell* cell;
    Cell* otherCell;
};

struct DelOperation
{
    enum class Type
    {
        DelConnections,
        DelCell
    };
    Type type;
    Cell* cell;
    int cellIndex;
};