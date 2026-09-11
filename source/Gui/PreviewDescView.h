#pragma once

#include <optional>
#include <vector>

#include <imgui.h>

#include <EngineInterface/PreviewDesc.h>

#include "Definitions.h"
#include "PreviewViewport.h"

// Draws the cell graph of previews into an ImGui draw list
class _PreviewDescView
{
public:
    static PreviewDescView create();

    enum class CellLabel
    {
        None,
        NodeIndex,
        CellType
    };

    struct RenderParameters
    {
        bool showFrontMarker = false;
        bool showGeneReferences = false;
        CellLabel cellLabel = CellLabel::None;
        std::optional<int> selectedGeneIndex;
        std::optional<int> selectedNodeIndex;
        std::optional<uint64_t> selectedCellId;
    };

    void draw(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, RenderParameters const& parameters);

    void drawCollage(ImDrawList* drawList, std::vector<PreviewDesc> const& previews, RealVector2D const& viewStartPos, RealVector2D const& viewSize);

private:
    _PreviewDescView() = default;

    void drawFrontMarker(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport);
    void drawSelection(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, RenderParameters const& parameters) const;
    void drawCells(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, RenderParameters const& parameters) const;
    void drawCellLabels(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, RenderParameters const& parameters) const;
    void drawConnections(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport) const;
    void drawGeneReferences(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport) const;

    std::optional<float> _lastFrontAngleRadius;
};
