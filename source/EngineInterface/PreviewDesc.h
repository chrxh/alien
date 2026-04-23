#pragma once

#include <vector>

#include <Base/Definitions.h>
#include <Base/Macros.h>

#include "CellTypeConstants.h"

struct SignalPreviewDesc
{
    SignalPreviewDesc();
    auto operator<=>(SignalPreviewDesc const&) const = default;

    MEMBER(SignalPreviewDesc, std::vector<float>, channels, {});
};

struct CellPreviewDesc
{
    auto operator<=>(CellPreviewDesc const&) const = default;

    MEMBER(CellPreviewDesc, uint64_t, id, 0);
    MEMBER(CellPreviewDesc, RealVector2D, pos, {});
    MEMBER(CellPreviewDesc, int, color, 0);  // -1 if cell is not in ready state
    MEMBER(CellPreviewDesc, int, geneIndex, 0);
    MEMBER(CellPreviewDesc, int, nodeIndex, 0);
    MEMBER(CellPreviewDesc, int, cellType, 0);
    MEMBER(CellPreviewDesc, SignalPreviewDesc, signal, SignalPreviewDesc());
    MEMBER(CellPreviewDesc, std::optional<int>, constructorGeneIndex, std::nullopt);
};

struct ConnectionPreviewDesc
{
    auto operator<=>(ConnectionPreviewDesc const&) const = default;

    MEMBER(ConnectionPreviewDesc, RealVector2D, object1, {});
    MEMBER(ConnectionPreviewDesc, RealVector2D, object2, {});
    MEMBER(ConnectionPreviewDesc, float, connectionWeightToObject1, 0.0f);
    MEMBER(ConnectionPreviewDesc, float, connectionWeightToObject2, 0.0f);
};

struct PreviewDesc
{
    auto operator<=>(PreviewDesc const&) const = default;

    MEMBER(PreviewDesc, std::vector<CellPreviewDesc>, objects, {});
    MEMBER(PreviewDesc, std::vector<ConnectionPreviewDesc>, connections, {});
};
