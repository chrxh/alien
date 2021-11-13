#pragma once

#include "Base.cuh"
#include "Definitions.cuh"

struct AddConnectionOperation {
    Cell* cell;
    Cell* otherCell;
};

struct DelConnectionsOperation
{
    Cell* cell;
};

struct DelCellOperation
{
    Cell* cell;
    int cellIndex;
};

struct DelCellAndConnectionOperations
{
    Cell* cell;
    int cellIndex;
};

union OperationData
{
    AddConnectionOperation addConnectionOperation;
    DelConnectionsOperation delConnectionsOperation;
    DelCellOperation delCellOperation;
    DelCellAndConnectionOperations delCellAndConnectionOperation;
};

struct Operation
{
    enum class Type
    {
        AddConnections,
        DelConnections,
        DelCell,
        DelCellAndConnections,
    };
    Type type;
    OperationData data;
};
