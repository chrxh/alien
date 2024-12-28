#pragma once
#include "Base/Definitions.h"

struct CellPreviewDescription
{
    RealVector2D pos;
    int color = 0;
    int nodeIndex = 0;
    bool partStart = false;
    bool partEnd = false;
    bool multipleConstructor = false;
    bool selfReplicator = false;
};

struct ConnectionPreviewDescription
{
    RealVector2D cell1;
    RealVector2D cell2;
    bool arrowToCell1 = false;
    bool arrowToCell2 = false;
};

struct SymbolPreviewDescription
{
    enum class Type
    {
        Dot,
        Infinity
    } type;
    RealVector2D pos;
};

struct PreviewDescription
{
    std::vector<CellPreviewDescription> cells;
    std::vector<ConnectionPreviewDescription> connections;
    std::vector<SymbolPreviewDescription> symbols;
};
