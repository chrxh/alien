#pragma once
#include "Base/Definitions.h"

struct CellPreviewDescription
{
    RealVector2D pos;
    int executionOrderNumber = 0;
    int color = 0;
    int nodeIndex = 0;
    bool multipleConstructor = false;

    enum class NodePos
    {
        Start,
        Intermediate,
        End
    } nodePos = NodePos::Intermediate;
};

struct ConnectionPreviewDescription
{
    RealVector2D cell1;
    RealVector2D cell2;
    bool arrowToCell1 = false;
    bool arrowToCell2 = false;
};

struct UndefinedCellPreviewDescription
{
    RealVector2D pos;
};

struct PreviewDescription
{
    std::vector<CellPreviewDescription> cells;
    std::vector<ConnectionPreviewDescription> connections;
    std::vector<UndefinedCellPreviewDescription> undefinedCells;
};
