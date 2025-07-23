#pragma once

#include <vector>

#include "Base/Definitions.h"
#include "Base/Macros.h"

struct CellPreviewDescription
{
    auto operator<=>(CellPreviewDescription const&) const = default;
    
    MEMBER(CellPreviewDescription, RealVector2D, pos, {});
    MEMBER(CellPreviewDescription, int, color, 0);
    MEMBER(CellPreviewDescription, int, nodeIndex, 0);
};

struct ConnectionPreviewDescription
{
    auto operator<=>(ConnectionPreviewDescription const&) const = default;
    
    MEMBER(ConnectionPreviewDescription, RealVector2D, cell1, {});
    MEMBER(ConnectionPreviewDescription, RealVector2D, cell2, {});
    MEMBER(ConnectionPreviewDescription, bool, arrowToCell1, false);
    MEMBER(ConnectionPreviewDescription, bool, arrowToCell2, false);
};

struct PreviewDescription
{
    auto operator<=>(PreviewDescription const&) const = default;
    
    MEMBER(PreviewDescription, std::vector<CellPreviewDescription>, cells, {});
    MEMBER(PreviewDescription, std::vector<ConnectionPreviewDescription>, connections, {});
};
