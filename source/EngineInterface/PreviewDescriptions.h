#pragma once

#include <vector>

#include "Base/Definitions.h"
#include "Base/Macros.h"

struct SignalRestrictionPreviewDescription
{
    auto operator<=>(SignalRestrictionPreviewDescription const&) const = default;

    MEMBER(SignalRestrictionPreviewDescription, float, startAngle, 0);
    MEMBER(SignalRestrictionPreviewDescription, float, endAngle, 0);
};

struct CellPreviewDescription
{
    auto operator<=>(CellPreviewDescription const&) const = default;
    
    MEMBER(CellPreviewDescription, RealVector2D, pos, {});
    MEMBER(CellPreviewDescription, int, color, 0);
    MEMBER(CellPreviewDescription, int, geneIndex, 0);
    MEMBER(CellPreviewDescription, int, nodeIndex, 0);
    MEMBER(CellPreviewDescription, std::optional<SignalRestrictionPreviewDescription>, signalRestriction, std::nullopt);
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
