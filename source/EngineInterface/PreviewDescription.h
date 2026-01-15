#pragma once

#include <vector>

#include <Base/Definitions.h>
#include <Base/Macros.h>

#include "CellTypeConstants.h"

struct SignalRestrictionPreviewDescription
{
    auto operator<=>(SignalRestrictionPreviewDescription const&) const = default;

    MEMBER(SignalRestrictionPreviewDescription, float, startAngle, 0);
    MEMBER(SignalRestrictionPreviewDescription, float, endAngle, 0);
};

struct SignalPreviewDescription
{
    SignalPreviewDescription();
    auto operator<=>(SignalPreviewDescription const&) const = default;

    MEMBER(SignalPreviewDescription, std::vector<float>, channels, {});
};

struct CellPreviewDescription
{
    auto operator<=>(CellPreviewDescription const&) const = default;

    MEMBER(CellPreviewDescription, uint64_t, id, 0);
    MEMBER(CellPreviewDescription, RealVector2D, pos, {});
    MEMBER(CellPreviewDescription, int, color, 0);  // -1 if cell is not in ready state
    MEMBER(CellPreviewDescription, int, geneIndex, 0);
    MEMBER(CellPreviewDescription, int, nodeIndex, 0);
    MEMBER(CellPreviewDescription, std::optional<SignalPreviewDescription>, signal, std::nullopt);
    MEMBER(CellPreviewDescription, std::optional<SignalRestrictionPreviewDescription>, signalRestriction, std::nullopt);
    MEMBER(CellPreviewDescription, SignalState, signalState, 0);
    MEMBER(CellPreviewDescription, std::optional<int>, constructorGeneIndex, std::nullopt);
};

struct ConnectionPreviewDescription
{
    auto operator<=>(ConnectionPreviewDescription const&) const = default;

    MEMBER(ConnectionPreviewDescription, RealVector2D, object1, {});
    MEMBER(ConnectionPreviewDescription, RealVector2D, object2, {});
    MEMBER(ConnectionPreviewDescription, bool, arrowToObject1, false);
    MEMBER(ConnectionPreviewDescription, bool, arrowToObject2, false);
};

struct PreviewDescription
{
    auto operator<=>(PreviewDescription const&) const = default;

    MEMBER(PreviewDescription, std::vector<CellPreviewDescription>, objects, {});
    MEMBER(PreviewDescription, std::vector<ConnectionPreviewDescription>, connections, {});
};
