#include "CreaturePreviewWidget.h"

#include <cmath>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/Math.h"
#include "Base/StringHelper.h"

#include "EngineInterface/Colors.h"
#include "EngineInterface/PreviewDescriptionConverterService.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr ZoomLevelForLabels = 16.0f;
    auto constexpr ZoomLevelForConnections = 8.0f;

}

CreaturePreviewWidget _CreaturePreviewWidget::create(
    GenomeTabEditData const& editData,
    GeneIndicesForSubGenome const& geneIndices,
    GenomeDescriptionWithStartGeneIndex const& genomeWithStartIndex)
{
    return CreaturePreviewWidget(new _CreaturePreviewWidget(editData, geneIndices, genomeWithStartIndex));
}

void _CreaturePreviewWidget::process(CollectionDescription&& phenotype)
{
    auto geneStartIndex = _genomeWithStartIndex.startIndex;
    auto conversionResult = PreviewDescriptionConverterService::get().convert(_editData->genome, std::move(phenotype), geneStartIndex, _visualFrontAngle);
    auto& desc = conversionResult.description;
    _visualFrontAngle = conversionResult.visualFrontAngle;

    auto windowSize = ImGui::GetWindowSize();
    if (ImGui::BeginChild("outerPreview", ImVec2(0, 0), 0, ImGuiWindowFlags_AlwaysHorizontalScrollbar)) {

        auto const cellSize = scale(_zoom);

        RealVector2D upperLeft;
        RealVector2D lowerRight;
        for (auto const& cell : desc._cells) {
            if (cell._pos.x < upperLeft.x) {
                upperLeft.x = cell._pos.x;
            }
            if (cell._pos.y < upperLeft.y) {
                upperLeft.y = cell._pos.y;
            }
            if (cell._pos.x > lowerRight.x) {
                lowerRight.x = cell._pos.x;
            }
            if (cell._pos.y > lowerRight.y) {
                lowerRight.y = cell._pos.y;
            }
        }
        RealVector2D previewSize = RealVector2D{200.0f, 200.0f} * cellSize;
        if (!_windowSizeFromPreviousFrame.has_value()) {
            float centerX = (previewSize.x - windowSize.x + scale(40.0f)) / 2.0f;
            float centerY = (previewSize.y - windowSize.y + scale(70.0f)) / 2.0f;
            ImGui::SetScrollX(centerX);
            ImGui::SetScrollY(centerY);
        } else {
            auto deltaX = windowSize.x - _windowSizeFromPreviousFrame->x;
            auto deltaY = windowSize.y - _windowSizeFromPreviousFrame->y;
            auto scrollX = ImGui::GetScrollX();
            auto scrollY = ImGui::GetScrollY();
            ImGui::SetScrollX(scrollX - deltaX / 2);
            ImGui::SetScrollY(scrollY - deltaY / 2);
        }
        _windowSizeFromPreviousFrame = RealVector2D{windowSize.x, windowSize.y};

        if (_zoomFromPreviousFrame.has_value() && _zoomFromPreviousFrame != _zoom) {
            auto zoomFactor = _zoom / _zoomFromPreviousFrame.value();
            if (zoomFactor > 1.0f) {
                ImGui::SetScrollX(ImGui::GetScrollX() * zoomFactor + windowSize.x / 4);
                ImGui::SetScrollY(ImGui::GetScrollY() * zoomFactor + windowSize.y / 4);
            } else {
                ImGui::SetScrollX((ImGui::GetScrollX() - windowSize.x / 4) * zoomFactor);
                ImGui::SetScrollY((ImGui::GetScrollY() - windowSize.y / 4) * zoomFactor);
            }
        }
        _zoomFromPreviousFrame = _zoom;

        processContent(conversionResult);
    }
    ImGui::EndChild();

    // Action buttons
    ImGui::SetCursorPos({ImGui::GetScrollX() + scale(10), ImGui::GetScrollY() + windowSize.y - scale(40)});
    processActionButtons();
}

uint64_t _CreaturePreviewWidget::getCreatureId() const
{
    return _creatureId;
}

void _CreaturePreviewWidget::setCreatureId(uint64_t value)
{
    _creatureId = value;
}

GeneIndicesForSubGenome const& _CreaturePreviewWidget::getGeneIndices() const
{
    return _geneIndices;
}

void _CreaturePreviewWidget::setGeneIndices(GeneIndicesForSubGenome const& value)
{
    _geneIndices = value;
}

GenomeDescriptionWithStartGeneIndex const& _CreaturePreviewWidget::getGenomeWithStartIndex() const
{
    return _genomeWithStartIndex;
}

void _CreaturePreviewWidget::setGenomeWithStartIndex(GenomeDescriptionWithStartGeneIndex const& value)
{
    _genomeWithStartIndex = value;
}

void _CreaturePreviewWidget::resetVisualFrontAngle()
{
    _visualFrontAngle.reset();
}

_CreaturePreviewWidget::_CreaturePreviewWidget(
    GenomeTabEditData const& editData,
    GeneIndicesForSubGenome const& geneIndices,
    GenomeDescriptionWithStartGeneIndex const& genomeWithStartIndex)
    : _editData(editData)
    , _geneIndices(geneIndices)
    , _genomeWithStartIndex(genomeWithStartIndex)
{}

void _CreaturePreviewWidget::processContent(ConversionResult const& conversionResult)
{
    auto const LineThickness = scale(2.0f);
    auto const cellSize = scale(_zoom);
    auto const previewSize = RealVector2D{200.0f, 200.0f} * cellSize;
    auto const previewCenter = previewSize / 2.0f;
    auto const& desc = conversionResult.description;
    auto& selectedGene = _editData->selectedGeneIndex;
    auto selectedNode = _editData->getSelectedNodeIndex();
    auto drawList = ImGui::GetWindowDrawList();
    auto const& style = StyleRepository::get();

    auto windowPos = ImGui::GetWindowPos();
    auto windowSize = ImGui::GetWindowSize();
    auto mousePos = ImGui::GetMousePos();
    auto clickedOnPreviewWindow = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mousePos.x >= windowPos.x && mousePos.y >= windowPos.y
        && mousePos.x <= windowPos.x + windowSize.x && mousePos.y <= windowPos.y + windowSize.y;

    if (ImGui::BeginChild("innerPreview", ImVec2(previewSize.x, previewSize.y), 0, ImGuiWindowFlags_NoScrollbar)) {

        auto windowPos = ImGui::GetWindowPos();
        RealVector2D offset{windowPos.x + cellSize, windowPos.y + cellSize};

        // Draw front circle
        {
            auto maxDistance = 0.0f;
            for (auto const& cell : desc._cells) {
                maxDistance = std::max(maxDistance, Math::length(cell._pos));
            }
            auto center = offset + previewCenter;
            auto radius = (maxDistance + 1.0f) * cellSize;
            radius = std::max(radius, cellSize * 3.0f);
            drawList->AddCircle({center.x, center.y}, radius, ImColor::HSV(0, 0, 0.2f), 64);

            auto textSize = scale(12.0f);

            auto frontStartPos = center + Math::unitVectorOfAngle(conversionResult.frontAngle) * (radius - textSize / 2);
            auto frontEndPos = center + Math::unitVectorOfAngle(conversionResult.frontAngle) * (radius + textSize / 2);
            drawList->AddLine({frontStartPos.x, frontStartPos.y}, {frontEndPos.x, frontEndPos.y}, ImColor::HSV(0, 0, 0.4f));

            AlienGui::RotateStart(drawList);
            auto textPos = center + Math::unitVectorOfAngle(conversionResult.frontAngle) * (radius + textSize);
            drawList->AddText(nullptr, textSize, {textPos.x - textSize + 0.5f, textPos.y - textSize / 2 + 0.5f}, ImColor::HSV(0, 0, 0.4f), "Front");
            AlienGui::RotateEnd(conversionResult.frontAngle, drawList);
        }

        // Draw selected gene
        auto selectedGeneColor = ImColor::HSV(0, 0, 0.15f);
        for (auto const& cell : desc._cells) {
            auto cellPos = cell._pos * cellSize + offset + previewCenter;
            if (selectedGene.has_value() && cell._geneIndex == selectedGene.value()) {
                drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.6f, selectedGeneColor);
            }
        }

        // Draw cells and selected cells
        for (auto const& cell : desc._cells) {
            auto cellPos = cell._pos * cellSize + offset + previewCenter;
            float h, s, v;
            AlienGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell._color], h, s, v);

            auto cellRadiusFactor = _zoom > ZoomLevelForConnections ? 0.25f : 0.5f;
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * cellRadiusFactor, ImColor::HSV(h, s * 1.2f, v * 1.0f));

            if (selectedGene.has_value() && selectedNode.has_value() && cell._geneIndex == selectedGene.value() && cell._nodeIndex == selectedNode.value()) {
                if (_zoom > ZoomLevelForLabels) {
                    drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor(1.0f, 1.0f, 1.0f));
                } else {
                    drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor::HSV(h, s * 0.8f, v * 1.2f));
                }
            }

            if (clickedOnPreviewWindow) {
                if (mousePos.x >= cellPos.x - cellSize / 2 && mousePos.y >= cellPos.y - cellSize / 2 && mousePos.x <= cellPos.x + cellSize / 2
                    && mousePos.y <= cellPos.y + cellSize / 2) {
                    selectedGene = cell._geneIndex;
                    selectedNode = cell._nodeIndex;
                }
            }
        }

        // Draw signal restrictions
        if (_zoom > ZoomLevelForConnections) {
            for (auto const& cell : desc._cells) {
                auto cellPos = cell._pos * cellSize + offset + previewCenter;
                auto constexpr cellRadiusFactor = 0.3f;
                float radius = cellSize * cellRadiusFactor;
                if (!cell._signalRestriction.has_value()) {
                    drawList->AddCircleFilled({cellPos.x, cellPos.y}, radius, ImColor::HSV(0, 0, 1.0f, 0.2f));
                } else {
                    auto startAngle = cell._signalRestriction->_startAngle;
                    auto endAngle = cell._signalRestriction->_endAngle;

                    // With the following code to draw a filled arc (pie segment) between startAngle and endAngle:
                    const int numSegments = 32;  // Increase for smoother arc
                    float radius = cellSize * cellRadiusFactor;
                    float startRad = startAngle * Const::DegToRad;
                    float endRad = endAngle * Const::DegToRad;
                    if (startRad > endRad) {
                        endRad += 2 * Const::Pi;  // If the angle wraps around, we need to adjust the end angle
                    }

                    float angleStep = (endRad - startRad) / numSegments;

                    std::vector<ImVec2> arcPoints;
                    arcPoints.push_back(ImVec2(cellPos.x, cellPos.y));  // Center
                    for (int i = 0; i <= numSegments; ++i) {
                        float angle = startRad + i * angleStep;
                        arcPoints.push_back(ImVec2(cellPos.x + radius * sinf(angle), cellPos.y - radius * cosf(angle)));
                    }

                    // Draw filled polygon (pie segment)
                    drawList->AddConvexPolyFilled(arcPoints.data(), arcPoints.size(), ImColor::HSV(0, 0, 1.0f, 0.2f));
                }
            }
        }

        // Draw cell connections
        if (_zoom > ZoomLevelForConnections) {
            for (auto const& connection : desc._connections) {
                auto cellPos1 = connection._cell1 * cellSize + offset + previewCenter;
                auto cellPos2 = connection._cell2 * cellSize + offset + previewCenter;

                auto direction = cellPos1 - cellPos2;

                Math::normalize(direction);
                auto connectionStartPos = cellPos1 - direction * cellSize / 4;
                auto connectionEndPos = cellPos2 + direction * cellSize / 4;
                drawList->AddLine(
                    {connectionStartPos.x, connectionStartPos.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);

                if (connection._arrowToCell1) {
                    auto arrowPartDirection1 = RealVector2D{-direction.x + direction.y, -direction.x - direction.y};
                    auto arrowPartStart1 = connectionStartPos + arrowPartDirection1 * cellSize / 8;
                    drawList->AddLine(
                        {arrowPartStart1.x, arrowPartStart1.y},
                        {connectionStartPos.x, connectionStartPos.y},
                        Const::GenomePreviewConnectionColor,
                        LineThickness);

                    auto arrowPartDirection2 = RealVector2D{-direction.x - direction.y, direction.x - direction.y};
                    auto arrowPartStart2 = connectionStartPos + arrowPartDirection2 * cellSize / 8;
                    drawList->AddLine(
                        {arrowPartStart2.x, arrowPartStart2.y},
                        {connectionStartPos.x, connectionStartPos.y},
                        Const::GenomePreviewConnectionColor,
                        LineThickness);
                }

                if (connection._arrowToCell2) {
                    auto arrowPartDirection1 = RealVector2D{direction.x - direction.y, direction.x + direction.y};
                    auto arrowPartStart1 = connectionEndPos + arrowPartDirection1 * cellSize / 8;
                    drawList->AddLine(
                        {arrowPartStart1.x, arrowPartStart1.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);

                    auto arrowPartDirection2 = RealVector2D{direction.x + direction.y, -direction.x + direction.y};
                    auto arrowPartStart2 = connectionEndPos + arrowPartDirection2 * cellSize / 8;
                    drawList->AddLine(
                        {arrowPartStart2.x, arrowPartStart2.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);
                }
            }
        }

        // Draw gene references
        if (_zoom > ZoomLevelForLabels) {
            for (auto const& cell : desc._cells) {
                if (cell._constructorGeneIndex.has_value()) {
                    auto cellPos = cell._pos * cellSize + offset + previewCenter;
                    auto text = std::to_string(cell._constructorGeneIndex.value() + 1);
                    auto textLength = text.size();
                    drawList->AddRectFilled(
                        {cellPos.x + cellSize * 0.3f, cellPos.y + cellSize * 0.2f},
                        {cellPos.x + cellSize * 0.32f * textLength + cellSize * 0.5f, cellPos.y + cellSize * 0.9f},
                        Const::GenomePreviewLinkToGeneBackgroundColor1);
                    drawList->AddRect(
                        {cellPos.x + cellSize * 0.3f, cellPos.y + cellSize * 0.2f},
                        {cellPos.x + cellSize * 0.32f * textLength + cellSize * 0.5f, cellPos.y + cellSize * 0.9f},
                        Const::GenomePreviewLinkToGeneBackgroundColor2);
                    drawList->AddText(
                        style.getSmallBoldFont(),
                        cellSize / 1.5f,
                        {cellPos.x + cellSize * 0.4f + 0.5f, cellPos.y + cellSize * 0.2f + 0.5f},
                        Const::GenomePreviewLinkToGeneTextColor,
                        text.c_str());
                }
            }
        }
    }
    ImGui::EndChild();

    _editData->setSelectedNodeIndex(selectedNode);
}

void _CreaturePreviewWidget::processActionButtons()
{
    if (ImGui::BeginChild("##buttons", ImVec2(scale(105), scale(30)), 0)) {
        ImGui::SetCursorPos({0, 0});
        ImGui::PushID(1);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_SEARCH_PLUS))) {
            _zoom *= 1.5f;
        }
        ImGui::PopID();
        ImGui::SameLine();
        ImGui::PushID(2);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_SEARCH_MINUS))) {
            _zoom /= 1.5f;
        }
        ImGui::PopID();
        //ImGui::SameLine();
        //AlienGui::VerticalSeparator(23);
        //ImGui::SameLine();
        //ImGui::PushID(3);
        //if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_FORWARD).highlighted(_settings->maxSpeed))) {
        //    _settings->maxSpeed = !_settings->maxSpeed;
        //}
        //ImGui::PopID();
    }
    ImGui::EndChild();
}
