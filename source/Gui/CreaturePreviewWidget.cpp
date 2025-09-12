#include "CreaturePreviewWidget.h"

#include <cmath>

#include <boost/range/adaptor/sliced.hpp>
#include <boost/algorithm/string/join.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/Math.h"
#include "Base/StringHelper.h"

#include "EngineInterface/Colors.h"
#include "EngineInterface/PreviewDescriptionConverterService.h"
#include "EngineInterface/SpaceCalculator.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "StyleRepository.h"
#include "SimulationScrollbars.h"

namespace
{
    auto constexpr ZoomLevelForLabels = 16.0f;
    auto constexpr ZoomLevelForConnections = 8.0f;
}

CreaturePreviewWidget _CreaturePreviewWidget::create(
    GenomeTabEditData const& editData,
    GeneIndicesForSubGenome const& geneIndices,
    SubGenomeDescription const& genomeWithStartIndex)
{
    return CreaturePreviewWidget(new _CreaturePreviewWidget(editData, geneIndices, genomeWithStartIndex));
}

void _CreaturePreviewWidget::process(Description&& phenotype, float width)
{
    GenomeDescriptionEditService::get().removeSeedFromPhenotype(phenotype);

    auto geneStartIndex = _subGenome.startIndex;

    auto conversionResult =
        PreviewDescriptionConverterService::get().convertToPreviewDescription(_editData->genome, geneStartIndex, std::move(phenotype), _visualFrontAngle);
    _visualFrontAngle = conversionResult.visualFrontAngle;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImColor(0.0f, 0.0f, 0.106f).Value);

    if (ImGui::BeginChild("CellGraphWidget", ImVec2(width, 0), 0, ImGuiWindowFlags_NoScrollbar)) {

        processCellGraph(conversionResult);

        ImGui::SetCursorPos({ImGui::GetScrollX() + scale(10.0f), ImGui::GetScrollY() + ImGui::GetWindowSize().y - scale(40.0f)});
        processActionButtons();

        RealVector2D windowPos{ImGui::GetWindowPos().x, ImGui::GetWindowPos().y};
        RealVector2D windowSize{ImGui::GetWindowWidth(), ImGui::GetWindowHeight()};

        RealRect worldRect{{-100.0f, -100.0f}, {100.0f, 100.0f}};
        RealRect visibleWorldRect{
            mapViewToWorldPosition(windowPos, windowSize, windowPos),
            mapViewToWorldPosition(windowPos + windowSize, windowSize, windowPos),
        };
        RealRect viewRect{windowPos, windowPos + windowSize};
        _scrollbars->process(_worldCenter, worldRect, visibleWorldRect, viewRect);

        processTitle();
    }
    ImGui::EndChild();

    ImGui::PopStyleColor();
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

SubGenomeDescription const& _CreaturePreviewWidget::getGenomeWithStartIndex() const
{
    return _subGenome;
}

void _CreaturePreviewWidget::setGenomeWithStartIndex(SubGenomeDescription const& value)
{
    _subGenome = value;
}

void _CreaturePreviewWidget::resetVisualFrontAngle()
{
    _visualFrontAngle.reset();
}

_CreaturePreviewWidget::_CreaturePreviewWidget(
    GenomeTabEditData const& editData,
    GeneIndicesForSubGenome const& geneIndices,
    SubGenomeDescription const& genomeWithStartIndex)
    : _editData(editData)
    , _geneIndices(geneIndices)
    , _subGenome(genomeWithStartIndex)
{
    _scrollbars = std::make_shared<_SimulationScrollbars>(false);
}

void _CreaturePreviewWidget::processCellGraph(ConversionResult const& conversionResult)
{
    auto const LineThickness = scale(2.0f);
    auto const cellSize = scale(_zoom);
    auto const& desc = conversionResult.description;
    auto& selectedGene = _editData->selectedGeneIndex;
    auto selectedNode = _editData->getSelectedNodeIndex();
    auto drawList = ImGui::GetWindowDrawList();
    auto& style = StyleRepository::get();

    RealVector2D windowPos{ImGui::GetWindowPos().x, ImGui::GetWindowPos().y};
    RealVector2D windowSize{ImGui::GetWindowWidth(), ImGui::GetWindowHeight()};
    auto mousePos = ImGui::GetMousePos();
    auto clickedOnPreviewWindow = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mousePos.x >= windowPos.x && mousePos.y >= windowPos.y
        && mousePos.x <= windowPos.x + windowSize.x && mousePos.y <= windowPos.y + windowSize.y;

    // Draw front circle
    {
        auto maxDistance = 0.0f;
        for (auto const& cell : desc._cells) {
            maxDistance = std::max(maxDistance, Math::length(cell._pos));
        }
        auto center = mapWorldToViewPosition({0, 0}, windowSize, windowPos);
        
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
        auto cellPos = mapWorldToViewPosition(cell._pos, windowSize, windowPos);
        if (selectedGene.has_value() && cell._geneIndex == selectedGene.value()) {
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.6f, selectedGeneColor);
        }
    }

    // Draw cells and selected cells
    for (auto const& cell : desc._cells) {
        auto cellPos = mapWorldToViewPosition(cell._pos, windowSize, windowPos);
        float h, s, v;
        AlienGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell._color], h, s, v);

        auto cellRadiusFactor = _zoom > ZoomLevelForConnections ? 0.25f : 0.5f;
        drawList->AddCircleFilled({cellPos.x, cellPos.y}, std::max(1.0f, cellSize * cellRadiusFactor), ImColor::HSV(h, s * 1.2f, v * 1.0f));

        if (selectedGene.has_value() && selectedNode.has_value() && cell._geneIndex == selectedGene.value() && cell._nodeIndex == selectedNode.value()) {
            if (_zoom > ZoomLevelForLabels) {
                drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor(1.0f, 1.0f, 1.0f));
            } else {
                drawList->AddCircle({cellPos.x, cellPos.y}, std::max(1.0f, cellSize / 2), ImColor::HSV(h, s * 0.8f, v * 1.2f));
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
            auto cellPos = mapWorldToViewPosition(cell._pos, windowSize, windowPos);
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
            auto cellPos1 = mapWorldToViewPosition(connection._cell1, windowSize, windowPos);
            auto cellPos2 = mapWorldToViewPosition(connection._cell2, windowSize, windowPos);

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
                auto cellPos = mapWorldToViewPosition(cell._pos, windowSize, windowPos);
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

void _CreaturePreviewWidget::processTitle()
{
    ImGui::SetCursorPos({scale(7.0f), scale(7.0f)});
    std::vector<std::string> geneIndexStrings;
    auto geneIndices = getGeneIndices();
    geneIndexStrings.emplace_back(std::to_string(geneIndices.front() + 1) + " (start)");
    for (auto const& geneIndex : geneIndices | boost::adaptors::sliced(1, geneIndices.size())) {
        geneIndexStrings.emplace_back(std::to_string(geneIndex + 1));
    }
    auto title = "Genes: " + boost::join(geneIndexStrings, ", ");
    if (_subGenome.trimmed) {
        title += "  -- trimmed";
    }
    AlienGui::Text(title.c_str());
}

RealVector2D _CreaturePreviewWidget::mapWorldToViewPosition(RealVector2D const& worldPos, RealVector2D const& viewSize, RealVector2D const& viewStartPos) const
{
    auto scaleFactor = scale(_zoom);
    return {
        (worldPos.x - _worldCenter.x) * scaleFactor + viewSize.x / 2 + viewStartPos.x,
        (worldPos.y - _worldCenter.y) * scaleFactor + viewSize.y / 2 + viewStartPos.y};
}

RealVector2D _CreaturePreviewWidget::mapViewToWorldPosition(RealVector2D const& viewPos, RealVector2D const& viewSize, RealVector2D const& viewStartPos) const
{
    auto scaleFactor = scale(_zoom);
    return {
        (viewPos.x - viewStartPos.x - viewSize.x / 2) / scaleFactor + _worldCenter.x,
        (viewPos.y - viewStartPos.y - viewSize.y / 2) / scaleFactor + _worldCenter.y
    };
}
