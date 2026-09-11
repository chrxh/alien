#pragma once

#include <optional>
#include <vector>

#include <imgui.h>

#include <EngineInterface/PreviewDesc.h>

#include "Definitions.h"

enum class PreviewCellLabel
{
    None,
    NodeIndex,
    CellType
};

// Mapping between the world of a preview and the view it is drawn into
struct PreviewViewport
{
    MEMBER(PreviewViewport, RealVector2D, worldCenter, {});
    MEMBER(PreviewViewport, float, zoom, 20.0f);
    MEMBER(PreviewViewport, RealVector2D, viewStartPos, {});
    MEMBER(PreviewViewport, RealVector2D, viewSize, {});
};

struct PreviewRenderParameters
{
    MEMBER(PreviewRenderParameters, bool, showFrontMarker, false);
    MEMBER(PreviewRenderParameters, bool, showGeneReferences, false);
    MEMBER(PreviewRenderParameters, PreviewCellLabel, cellLabel, PreviewCellLabel::None);
    MEMBER(PreviewRenderParameters, std::optional<int>, selectedGeneIndex, std::nullopt);
    MEMBER(PreviewRenderParameters, std::optional<int>, selectedNodeIndex, std::nullopt);
    MEMBER(PreviewRenderParameters, std::optional<uint64_t>, selectedCellId, std::nullopt);
};

// Draws the cell graph of previews into an ImGui draw list
class PreviewDescRenderer
{
public:
    void draw(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, PreviewRenderParameters const& parameters);

    // Draws the previews as a collage of tiles separated by thin lines. All creatures share one zoom so that their relative sizes are preserved,
    // the largest creature is placed in the first tile.
    void drawCollage(ImDrawList* drawList, std::vector<PreviewDesc> const& previews, RealVector2D const& viewStartPos, RealVector2D const& viewSize);

    static RealVector2D mapWorldToViewPosition(RealVector2D const& worldPos, PreviewViewport const& viewport);
    static RealVector2D mapViewToWorldPosition(RealVector2D const& viewPos, PreviewViewport const& viewport);

    // Distance from the origin within which all cells of the preview are located
    static float calcContentRadius(PreviewDesc const& desc);

    // Zoom level at which the given content radius fits into the given view size
    static float calcZoomToFitContent(float contentRadius, RealVector2D const& viewSize);

private:
    void drawFrontMarker(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport);
    void drawSelection(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, PreviewRenderParameters const& parameters) const;
    void drawCells(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, PreviewRenderParameters const& parameters) const;
    void drawCellLabels(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, PreviewRenderParameters const& parameters) const;
    void drawConnections(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport) const;
    void drawGeneReferences(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport) const;

    std::optional<float> _lastFrontAngleRadius;
};
