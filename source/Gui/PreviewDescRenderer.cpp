#include "PreviewDescRenderer.h"

#include <algorithm>
#include <ranges>

#include <Base/Math.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr ZoomLevelForGeneReferences = 16.0f;
    auto constexpr ZoomLevelForCellLabels = 32.0f;
    auto constexpr ZoomLevelForConnections = 8.0f;
    auto constexpr MaxCellLabelTextSize = 16.0f;

    auto constexpr SignalStrengthWhiteness = 0.2f;
    auto constexpr SignalStrengthEnlargement = 0.5f;

    auto constexpr CollageTileMargin = 5.0f;
}

void PreviewDescRenderer::draw(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, PreviewRenderParameters const& parameters)
{
    if (parameters._showFrontMarker) {
        drawFrontMarker(drawList, desc, viewport);
    }
    drawSelection(drawList, desc, viewport, parameters);
    drawCells(drawList, desc, viewport, parameters);
    drawCellLabels(drawList, desc, viewport, parameters);
    drawConnections(drawList, desc, viewport);
    if (parameters._showGeneReferences) {
        drawGeneReferences(drawList, desc, viewport);
    }
}

namespace
{
    struct PreviewTile
    {
        RealVector2D startPos;
        RealVector2D size;
        int row = 0;
    };

    // Arranges the tiles of the collage in the grid that provides the largest tiles, filling the last row centered
    std::vector<PreviewTile> calcTiles(int numTiles, RealVector2D const& viewStartPos, RealVector2D const& viewSize)
    {
        auto numColumns = 1;
        auto largestTileExtent = 0.0f;
        for (auto columns : std::views::iota(1, numTiles + 1)) {
            auto rows = (numTiles + columns - 1) / columns;
            auto tileExtent = std::min(viewSize.x / toFloat(columns), viewSize.y / toFloat(rows));
            if (tileExtent > largestTileExtent) {
                largestTileExtent = tileExtent;
                numColumns = columns;
            }
        }
        auto numRows = (numTiles + numColumns - 1) / numColumns;
        RealVector2D tileSize{viewSize.x / toFloat(numColumns), viewSize.y / toFloat(numRows)};

        std::vector<PreviewTile> result;
        for (auto index : std::views::iota(0, numTiles)) {
            auto row = index / numColumns;
            auto column = index % numColumns;
            auto tilesInRow = std::min(numColumns, numTiles - row * numColumns);
            auto rowOffsetX = (viewSize.x - toFloat(tilesInRow) * tileSize.x) / 2;
            result.emplace_back(
                RealVector2D{viewStartPos.x + rowOffsetX + toFloat(column) * tileSize.x, viewStartPos.y + toFloat(row) * tileSize.y}, tileSize, row);
        }
        return result;
    }

    void drawTileSeparators(ImDrawList* drawList, std::vector<PreviewTile> const& tiles, RealVector2D const& viewStartPos, RealVector2D const& viewSize)
    {
        auto thickness = scale(1.0f);
        for (auto const& [tile, nextTile] : std::views::zip(tiles, tiles | std::views::drop(1))) {
            if (tile.row == nextTile.row) {
                auto x = tile.startPos.x + tile.size.x;
                drawList->AddLine({x, tile.startPos.y}, {x, tile.startPos.y + tile.size.y}, Const::GenomePreviewSeparatorColor, thickness);
            } else {
                auto y = tile.startPos.y + tile.size.y;
                drawList->AddLine({viewStartPos.x, y}, {viewStartPos.x + viewSize.x, y}, Const::GenomePreviewSeparatorColor, thickness);
            }
        }
    }
}

void PreviewDescRenderer::drawCollage(
    ImDrawList* drawList,
    std::vector<PreviewDesc> const& previews,
    RealVector2D const& viewStartPos,
    RealVector2D const& viewSize)
{
    if (previews.empty()) {
        return;
    }

    std::vector<PreviewDesc const*> sortedPreviews;
    for (auto const& preview : previews) {
        sortedPreviews.emplace_back(&preview);
    }
    std::ranges::sort(sortedPreviews, std::greater{}, [](auto const* preview) { return calcContentRadius(*preview); });

    auto tiles = calcTiles(toInt(sortedPreviews.size()), viewStartPos, viewSize);
    drawTileSeparators(drawList, tiles, viewStartPos, viewSize);

    auto margin = scale(CollageTileMargin);
    auto contentSize = tiles.front().size - RealVector2D{margin * 2, margin * 2};
    auto zoom = calcZoomToFitContent(calcContentRadius(*sortedPreviews.front()), contentSize);

    for (auto const& [preview, tile] : std::views::zip(sortedPreviews, tiles)) {
        auto viewport = PreviewViewport().zoom(zoom).viewStartPos(tile.startPos).viewSize(tile.size);
        draw(drawList, *preview, viewport, PreviewRenderParameters());
    }
}

RealVector2D PreviewDescRenderer::mapWorldToViewPosition(RealVector2D const& worldPos, PreviewViewport const& viewport)
{
    auto scaleFactor = scale(viewport._zoom);
    return {
        (worldPos.x - viewport._worldCenter.x) * scaleFactor + viewport._viewSize.x / 2 + viewport._viewStartPos.x,
        (worldPos.y - viewport._worldCenter.y) * scaleFactor + viewport._viewSize.y / 2 + viewport._viewStartPos.y};
}

RealVector2D PreviewDescRenderer::mapViewToWorldPosition(RealVector2D const& viewPos, PreviewViewport const& viewport)
{
    auto scaleFactor = scale(viewport._zoom);
    return {
        (viewPos.x - viewport._viewStartPos.x - viewport._viewSize.x / 2) / scaleFactor + viewport._worldCenter.x,
        (viewPos.y - viewport._viewStartPos.y - viewport._viewSize.y / 2) / scaleFactor + viewport._worldCenter.y};
}

float PreviewDescRenderer::calcContentRadius(PreviewDesc const& desc)
{
    auto maxDistance = 0.0f;
    for (auto const& cell : desc._cells) {
        maxDistance = std::max(maxDistance, Math::length(cell._pos));
    }
    return std::max(maxDistance + 1.0f, 3.0f);
}

float PreviewDescRenderer::calcZoomToFitContent(float contentRadius, RealVector2D const& viewSize)
{
    auto viewExtent = std::min(viewSize.x, viewSize.y);
    return scaleInverse(viewExtent / (2.0f * contentRadius));
}

void PreviewDescRenderer::drawFrontMarker(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport)
{
    // The radius follows the extent of the creature, but small changes are ignored to keep the marker calm
    auto radius = calcContentRadius(desc) * 1.25f;
    if (_lastFrontAngleRadius.has_value() && _lastFrontAngleRadius.value() / (radius + 1.0f) < 1.25f
        && _lastFrontAngleRadius.value() / (radius + 1.0f) > 0.75f) {
        radius = _lastFrontAngleRadius.value();
    }
    _lastFrontAngleRadius = radius;

    radius *= scale(viewport._zoom);

    auto center = mapWorldToViewPosition({0, 0}, viewport);
    drawList->AddCircle({center.x, center.y}, radius, ImColor::HSV(0, 0, 0.2f), 64);

    auto textSize = scale(12.0f);

    auto const visualFrontAngle = 0;
    auto frontStartPos = center + Math::unitVectorOfAngle(visualFrontAngle) * (radius - textSize / 2);
    auto frontEndPos = center + Math::unitVectorOfAngle(visualFrontAngle) * (radius + textSize / 2);
    drawList->AddLine({frontStartPos.x, frontStartPos.y}, {frontEndPos.x, frontEndPos.y}, ImColor::HSV(0, 0, 0.4f));

    AlienGui::RotateStart(drawList);
    auto textPos = center + Math::unitVectorOfAngle(visualFrontAngle) * (radius + textSize);
    AlienGui::AddTextWithSubpixelAccuracy(
        drawList, ImGui::GetFont(), textSize, {textPos.x - textSize, textPos.y - textSize / 2}, ImColor::HSV(0, 0, 0.4f), "Front");
    AlienGui::RotateEnd(visualFrontAngle, drawList);
}

void PreviewDescRenderer::drawSelection(
    ImDrawList* drawList,
    PreviewDesc const& desc,
    PreviewViewport const& viewport,
    PreviewRenderParameters const& parameters) const
{
    auto const cellSize = scale(viewport._zoom);
    auto const& selectedGene = parameters._selectedGeneIndex;
    auto const& selectedNode = parameters._selectedNodeIndex;
    auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;

    // Draw selected gene
    auto selectedGeneColor = ImColor::HSV(0.66f, 0.5f, 0.1f);
    for (auto const& cell : desc._cells) {
        auto cellPos = mapWorldToViewPosition(cell._pos, viewport);
        if (selectedGene.has_value() && cell._geneIndex == selectedGene.value()) {
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.6f, selectedGeneColor);
        }
    }

    // Draw selected nodes
    for (auto const& cell : desc._cells) {
        auto cellPos = mapWorldToViewPosition(cell._pos, viewport);
        if (selectedGene.has_value() && selectedNode.has_value() && cell._geneIndex == selectedGene.value() && cell._nodeIndex == selectedNode.value()) {
            ImU32 color;
            if (cell._inactive) {
                float h, s, v;
                AlienGui::ConvertRGBtoHSV(Const::GenomePreviewInactiveColor, h, s, v);
                color = ImColor::HSV(h, s, v * 0.7f);
            } else {
                float h, s, v;
                AlienGui::ConvertRGBtoHSV(customizationColors.values[cell._color].toRgbColor(), h, s, v);
                color = ImColor::HSV(h, 0.5f, 0.4f);
            }
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.4f, color);
        }
    }
}

void PreviewDescRenderer::drawCells(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport, PreviewRenderParameters const& parameters)
    const
{
    auto const cellSize = scale(viewport._zoom);
    auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;

    for (auto const& cell : desc._cells) {
        auto cellPos = mapWorldToViewPosition(cell._pos, viewport);
        float h, s, v;
        uint32_t color = customizationColors.values[cell._color].toRgbColor();
        if (cell._inactive) {
            color = Const::GenomePreviewInactiveColor;
        }
        AlienGui::ConvertRGBtoHSV(color, h, s, v);

        auto signalStrength = viewport._zoom > ZoomLevelForConnections ? toFloat(cell._highlightIntensity) / 255.0f : 0.0f;
        auto whiteness = signalStrength * SignalStrengthWhiteness;

        auto cellRadiusFactor = (viewport._zoom > ZoomLevelForConnections ? 0.15f : 0.5f) * (1.0f + signalStrength * SignalStrengthEnlargement);
        drawList->AddCircleFilled(
            {cellPos.x, cellPos.y},
            std::max(1.0f, cellSize * cellRadiusFactor),
            ImColor::HSV(h, s * 1.2f * (1.0f - whiteness), v * (1.0f - whiteness) + whiteness));

        if (parameters._selectedCellId.has_value() && parameters._selectedCellId.value() == cell._id) {
            if (viewport._zoom > ZoomLevelForGeneReferences) {
                drawList->AddCircle({cellPos.x, cellPos.y}, cellSize * 0.15f, ImColor::HSV(0, 0, 1, 0.7f), 0, 2.0f);
            }
        }
    }
}

void PreviewDescRenderer::drawCellLabels(
    ImDrawList* drawList,
    PreviewDesc const& desc,
    PreviewViewport const& viewport,
    PreviewRenderParameters const& parameters) const
{
    if (parameters._cellLabel == PreviewCellLabel::None || viewport._zoom <= ZoomLevelForCellLabels) {
        return;
    }

    auto const cellSize = scale(viewport._zoom);
    auto font = StyleRepository::get().getSmallBoldFont();
    for (auto const& cell : desc._cells) {
        auto cellPos = mapWorldToViewPosition(cell._pos, viewport);
        auto text = parameters._cellLabel == PreviewCellLabel::NodeIndex ? std::to_string(cell._nodeIndex) : Const::CellTypeStrings.at(cell._cellType);
        auto fontSize = std::min(cellSize * 0.18f, MaxCellLabelTextSize);
        auto textSize = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, text.c_str());
        AlienGui::AddTextWithSubpixelAccuracy(
            drawList, font, fontSize, {cellPos.x - textSize.x / 2 + 1, cellPos.y - textSize.y / 2 + 1}, ImColor::HSV(0, 0, 0, 0.7f), text.c_str());
        AlienGui::AddTextWithSubpixelAccuracy(
            drawList, font, fontSize, {cellPos.x - textSize.x / 2, cellPos.y - textSize.y / 2}, ImColor::HSV(0, 0, 1.0f, 0.7f), text.c_str());
    }
}

void PreviewDescRenderer::drawConnections(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport) const
{
    if (viewport._zoom <= ZoomLevelForConnections) {
        return;
    }

    auto const lineThickness = scale(1.0f);
    auto const cellSize = scale(viewport._zoom);
    for (auto const& connection : desc._connections) {
        auto cellPos1 = mapWorldToViewPosition(connection._cell1, viewport);
        auto cellPos2 = mapWorldToViewPosition(connection._cell2, viewport);
        auto connectionColor = connection._inactive ? Const::GenomePreviewInactiveColor : Const::GenomePreviewConnectionColor;

        auto direction = cellPos1 - cellPos2;

        Math::normalize(direction);
        auto connectionStartPos = cellPos1 - direction * cellSize * 0.15f;
        auto connectionEndPos = cellPos2 + direction * cellSize * 0.15f;
        drawList->AddLine({connectionStartPos.x, connectionStartPos.y}, {connectionEndPos.x, connectionEndPos.y}, connectionColor, lineThickness);

        if (connection._connectionWeightToObject1 != 0.0f) {
            auto arrowScale = std::min(std::abs(connection._connectionWeightToObject1), 1.0f);
            auto arrowPartDirection1 = RealVector2D{-direction.x + direction.y, -direction.x - direction.y};
            auto arrowPartStart1 = connectionStartPos + arrowPartDirection1 * cellSize / 8 * arrowScale;
            drawList->AddLine({arrowPartStart1.x, arrowPartStart1.y}, {connectionStartPos.x, connectionStartPos.y}, connectionColor, lineThickness);

            auto arrowPartDirection2 = RealVector2D{-direction.x - direction.y, direction.x - direction.y};
            auto arrowPartStart2 = connectionStartPos + arrowPartDirection2 * cellSize / 8 * arrowScale;
            drawList->AddLine({arrowPartStart2.x, arrowPartStart2.y}, {connectionStartPos.x, connectionStartPos.y}, connectionColor, lineThickness);
        }

        if (connection._connectionWeightToObject2 != 0.0f) {
            auto arrowScale = std::min(std::abs(connection._connectionWeightToObject2), 1.0f);
            auto arrowPartDirection1 = RealVector2D{direction.x - direction.y, direction.x + direction.y};
            auto arrowPartStart1 = connectionEndPos + arrowPartDirection1 * cellSize / 8 * arrowScale;
            drawList->AddLine({arrowPartStart1.x, arrowPartStart1.y}, {connectionEndPos.x, connectionEndPos.y}, connectionColor, lineThickness);

            auto arrowPartDirection2 = RealVector2D{direction.x + direction.y, -direction.x + direction.y};
            auto arrowPartStart2 = connectionEndPos + arrowPartDirection2 * cellSize / 8 * arrowScale;
            drawList->AddLine({arrowPartStart2.x, arrowPartStart2.y}, {connectionEndPos.x, connectionEndPos.y}, connectionColor, lineThickness);
        }
    }
}

void PreviewDescRenderer::drawGeneReferences(ImDrawList* drawList, PreviewDesc const& desc, PreviewViewport const& viewport) const
{
    if (viewport._zoom <= ZoomLevelForGeneReferences) {
        return;
    }

    auto const cellSize = scale(viewport._zoom);
    auto font = StyleRepository::get().getSmallBoldFont();
    for (auto const& cell : desc._cells) {
        if (!cell._constructorGeneIndex.has_value()) {
            continue;
        }
        auto cellPos = mapWorldToViewPosition(cell._pos, viewport);
        auto text = std::to_string(cell._constructorGeneIndex.value());
        auto textLength = toFloat(text.size());
        auto truncatedSize = std::min(scale(30.0f), cellSize);
        drawList->AddRectFilled(
            {cellPos.x + truncatedSize * 0.2f, cellPos.y + truncatedSize * 0.1f},
            {cellPos.x + truncatedSize * 0.32f * textLength + truncatedSize * 0.4f, cellPos.y + truncatedSize * 0.8f},
            Const::GenomePreviewGeneRefBackgroundColor1);
        drawList->AddRect(
            {cellPos.x + truncatedSize * 0.2f, cellPos.y + truncatedSize * 0.1f},
            {cellPos.x + truncatedSize * 0.32f * textLength + truncatedSize * 0.4f, cellPos.y + truncatedSize * 0.8f},
            Const::GenomePreviewGeneRefBackgroundColor2);
        AlienGui::AddTextWithSubpixelAccuracy(
            drawList,
            font,
            truncatedSize / 1.5f,
            {cellPos.x + truncatedSize * 0.3f, cellPos.y + truncatedSize * 0.1f},
            Const::GenomePreviewLinkToGeneTextColor,
            text.c_str());
    }
}
