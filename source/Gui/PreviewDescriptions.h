#pragma once
#include "Base/Definitions.h"

struct CellPreviewDescription
{
    RealVector2D pos;
    int executionOrderNumber = 0;
    int color = 0;
    bool selected = false;
};

struct ConnectionPreviewDescription
{
    int cellIndex1;
    int cellIndex2;
};

struct PreviewDescription
{
    std::vector<CellPreviewDescription> cells;
    std::vector<ConnectionPreviewDescription> connections;
};
