#pragma once
#include "Base/Definitions.h"

struct CellPreviewDescription
{
    RealVector2D pos;
    int color;
};

struct ConnectionPreviewDescription
{
    RealVector2D cell1;
    RealVector2D cell2;
};

struct PreviewDescription
{
    std::vector<CellPreviewDescription> cells;
    std::vector<ConnectionPreviewDescription> connections;
};
