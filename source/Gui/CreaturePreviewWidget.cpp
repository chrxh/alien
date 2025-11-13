#include "CreaturePreviewWidget.h"

#include <cmath>
#include <ranges>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/sliced.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Math.h>
#include <Base/StringHelper.h>

#include <EngineInterface/Colors.h>
#include <EngineInterface/PreviewDescriptionConverterService.h>
#include <EngineInterface/SpaceCalculator.h>

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "SimulationScrollbars.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr ZoomLevelForLabels = 16.0f;
    auto constexpr ZoomLevelForConnections = 8.0f;
    auto constexpr SignalTextWidth = 25.0f;
}


CreaturePreviewWidget
_CreaturePreviewWidget::create(GenomeTabEditData const& editData, GeneIndicesForSubGenome const& geneIndices, SubGenomeDescription const& genomeWithStartIndex)
{
    return CreaturePreviewWidget(new _CreaturePreviewWidget(editData, geneIndices, genomeWithStartIndex));
}

void _CreaturePreviewWidget::process(bool& phenotypeChanged, Description& phenotype, float width)
{
    auto phenotypeWithoutSeed = phenotype;
    GenomeDescriptionEditService::get().removeSeedFromPhenotype(phenotypeWithoutSeed);

    auto geneStartIndex = _subGenome.startIndex;

    auto conversionResult = PreviewDescriptionConverterService::get().convertToPreviewDescription(
        _editData->genome, geneStartIndex, std::move(phenotypeWithoutSeed), _visualFrontAngle);
    _visualFrontAngle = conversionResult.visualFrontAngle;

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImColor(0.0f, 0.0f, 0.106f).Value);

    if (ImGui::BeginChild("CellGraphWidget", ImVec2(width, 0), 0, ImGuiWindowFlags_NoScrollbar)) {
        processMouseNavigation();
        processCellGraphAndSelection(conversionResult);
        processSignalEditor(phenotypeChanged, phenotype, conversionResult);
        processActionButtons();
        processScrollbars();
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

void _CreaturePreviewWidget::processMouseNavigation()
{
    if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
        RealVector2D windowSize{ImGui::GetWindowWidth(), ImGui::GetWindowHeight()};
        RealVector2D windowPos{ImGui::GetWindowPos().x, ImGui::GetWindowPos().y};
        RealVector2D mousePos = {ImGui::GetMousePos().x, ImGui::GetMousePos().y};

        if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle)) {
            _worldPosForPanning = mapViewToWorldPosition(mousePos, windowSize, windowPos);
        }
        if (ImGui::IsMouseDown(ImGuiMouseButton_Middle) && _worldPosForPanning.has_value()) {
            moveCenter(_worldPosForPanning.value(), mousePos, windowSize, windowPos);
        }
        if (ImGui::GetIO().MouseWheel > 0) {
            auto worldPos = mapViewToWorldPosition(mousePos, windowSize, windowPos);
            _zoom *= sqrt(1.5f);
            moveCenter(worldPos, mousePos, windowSize, windowPos);
        }
        if (ImGui::GetIO().MouseWheel < 0) {
            auto worldPos = mapViewToWorldPosition(mousePos, windowSize, windowPos);
            _zoom /= sqrt(1.5f);
            moveCenter(worldPos, mousePos, windowSize, windowPos);
        }
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Middle)) {
        _worldPosForPanning.reset();
    }
}

void _CreaturePreviewWidget::processCellGraphAndSelection(ConversionResult const& conversionResult)
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
    auto clickedOnPreviewWindow = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);

    // Clear selection if another node has been selected outside of this widget or if cell id does not exist in preview
    auto selectedCellIdExists = false;
    for (auto const& cell : desc._cells) {
        if (_selectedCellIdFromPreview.has_value() && _selectedCellIdFromPreview.value() == cell._id) {
            selectedCellIdExists = true;
            break;
        }
    }
    if (!selectedCellIdExists || _selectedNodeFromPreview != selectedNode) {
        _selectedCellIdFromPreview.reset();
        _selectedNodeFromPreview.reset();
    }

    // Draw front circle
    {
        auto maxDistance = 0.0f;
        for (auto const& cell : desc._cells) {
            maxDistance = std::max(maxDistance, Math::length(cell._pos));
        }
        auto radius = (maxDistance + 1.0f);
        radius = std::max(radius, 3.0f) * 1.25f;
        if (_lastFrontAngleRadius.has_value() && _lastFrontAngleRadius.value() / (radius + 1.0f) < 1.25f
            && _lastFrontAngleRadius.value() / (radius + 1.0f) > 0.75f) {
            radius = _lastFrontAngleRadius.value();
        }
        _lastFrontAngleRadius = radius;

        radius *= cellSize;

        auto center = mapWorldToViewPosition({0, 0}, windowSize, windowPos);
        drawList->AddCircle({center.x, center.y}, radius, ImColor::HSV(0, 0, 0.2f), 64);

        auto textSize = scale(12.0f);

        auto frontStartPos = center + Math::unitVectorOfAngle(conversionResult.frontAngle) * (radius - textSize / 2);
        auto frontEndPos = center + Math::unitVectorOfAngle(conversionResult.frontAngle) * (radius + textSize / 2);
        drawList->AddLine({frontStartPos.x, frontStartPos.y}, {frontEndPos.x, frontEndPos.y}, ImColor::HSV(0, 0, 0.4f));

        AlienGui::RotateStart(drawList);
        auto textPos = center + Math::unitVectorOfAngle(conversionResult.frontAngle) * (radius + textSize);
        AlienGui::AddTextWithSubpixelAccuracy(
            drawList, ImGui::GetFont(), textSize, {textPos.x - textSize, textPos.y - textSize / 2}, ImColor::HSV(0, 0, 0.4f), "Front");
        AlienGui::RotateEnd(conversionResult.frontAngle, drawList);
    }

    // Draw selected gene
    auto selectedGeneColor = ImColor::HSV(0.66f, 0.5f, 0.15f);
    for (auto const& cell : desc._cells) {
        auto cellPos = mapWorldToViewPosition(cell._pos, windowSize, windowPos);
        if (selectedGene.has_value() && cell._geneIndex == selectedGene.value()) {
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.6f, selectedGeneColor);
        }
    }

    // Draw selected nodes
    for (auto const& cell : desc._cells) {
        auto cellPos = mapWorldToViewPosition(cell._pos, windowSize, windowPos);
        if (selectedGene.has_value() && selectedNode.has_value() && cell._geneIndex == selectedGene.value() && cell._nodeIndex == selectedNode.value()) {
            float h, s, v;
            AlienGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell._color], h, s, v);
            s = 0.5f;
            v = 0.3f;
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.6f, ImColor::HSV(h, s, v));
        }
    }

    // Draw signal restrictions
    if (_zoom > ZoomLevelForConnections) {
        for (auto const& cell : desc._cells) {
            auto cellPos = mapWorldToViewPosition(cell._pos, windowSize, windowPos);
            float radius = cellSize * 0.33f;

            if (!cell._signalRestriction.has_value()) {
                drawList->AddCircleFilled({cellPos.x, cellPos.y}, radius, ImColor::HSV(0, 0, 1.0f, 0.2f));
            } else {
                auto startAngle = Math::getNormalizedAngle(cell._signalRestriction->_startAngle, 0);
                auto endAngle = Math::getNormalizedAngle(cell._signalRestriction->_endAngle, 0);

                // Draw filled ring sector (annular sector) between startAngle and endAngle
                const int numSegments = 32;  // Increase for smoother arc
                float startRad = Math::getNormalizedAngle(startAngle * Const::DegToRad, 0);
                float endRad = Math::getNormalizedAngle(endAngle * Const::DegToRad, 0);
                if (startRad > endRad) {
                    endRad += 2 * Const::Pi;  // If the angle wraps around, we need to adjust the end angle
                }
                float angleStep = (endRad - startRad) / numSegments;

                std::vector<ImVec2> ringPoints;
                ringPoints.reserve((numSegments + 1) * 2);

                // Outer arc (from start to end)
                for (int i = 0; i <= numSegments; ++i) {
                    float angle = startRad + i * angleStep;
                    ringPoints.push_back(ImVec2(cellPos.x + radius * sinf(angle), cellPos.y - radius * cosf(angle)));
                }
                ringPoints.push_back(ImVec2(cellPos.x, cellPos.y));

                // Draw filled polygon (ring sector)
                drawList->AddConcavePolyFilled(ringPoints.data(), ringPoints.size(), ImColor::HSV(0, 0, 1.0f, 0.2f));
            }
        }
    }


    // Draw cells and selected cells
    for (auto const& cell : desc._cells) {
        auto cellPos = mapWorldToViewPosition(cell._pos, windowSize, windowPos);
        float h, s, v;
        AlienGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell._color], h, s, v);

        auto cellRadiusFactor = _zoom > ZoomLevelForConnections ? 0.25f : 0.5f;
        drawList->AddCircleFilled({cellPos.x, cellPos.y}, std::max(1.0f, cellSize * cellRadiusFactor), ImColor::HSV(h, s * 1.2f, v * 1.0f));

        if (_selectedCellIdFromPreview.has_value() && _selectedCellIdFromPreview.value() == cell._id) {
            if (_zoom > ZoomLevelForLabels) {
                drawList->AddCircle({cellPos.x, cellPos.y}, cellSize * 0.25f, ImColor::HSV(0, 0, 1, 0.7f), 0, 2.0f/*cellSize * 0.05f*/);
            }
        }

        if (clickedOnPreviewWindow) {
            if (mousePos.x >= cellPos.x - cellSize / 2 && mousePos.y >= cellPos.y - cellSize / 2 && mousePos.x <= cellPos.x + cellSize / 2
                && mousePos.y <= cellPos.y + cellSize / 2) {
                selectedGene = cell._geneIndex;
                selectedNode = cell._nodeIndex;
                _selectedNodeFromPreview = selectedNode;
                _selectedCellIdFromPreview = cell._id;

            }
        }
    }

    // Draw signals
    if (_zoom > ZoomLevelForConnections) {
        for (auto const& cell : desc._cells) {
            auto cellPos = mapWorldToViewPosition(cell._pos, windowSize, windowPos);
            auto constexpr cellRadiusFactor = 0.3f;
            float radius = cellSize * cellRadiusFactor;

            if (cell._signalState == SignalState_Active) {
                drawList->AddCircleFilled({cellPos.x, cellPos.y}, radius * 0.65f, ImColor::HSV(0, 0, 1.0f, 1.0f));
                drawList->AddCircle({cellPos.x, cellPos.y}, radius * 0.65f, ImColor::HSV(0, 0, 0.2f, 0.8f));
            }
            if (cell._signalState == SignalState_Fading) {
                drawList->AddCircleFilled({cellPos.x, cellPos.y}, radius * 0.35f, ImColor::HSV(0, 0, 1.0f, 0.5f));
                drawList->AddCircle({cellPos.x, cellPos.y}, radius * 0.35f, ImColor::HSV(0, 0, 0.2f, 0.5f));
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
            auto connectionStartPos = cellPos1 - direction * cellSize * 0.25f;
            auto connectionEndPos = cellPos2 + direction * cellSize * 0.25f;
            drawList->AddLine(
                {connectionStartPos.x, connectionStartPos.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);

            if (connection._arrowToCell1) {
                auto arrowPartDirection1 = RealVector2D{-direction.x + direction.y, -direction.x - direction.y};
                auto arrowPartStart1 = connectionStartPos + arrowPartDirection1 * cellSize / 8;
                drawList->AddLine(
                    {arrowPartStart1.x, arrowPartStart1.y}, {connectionStartPos.x, connectionStartPos.y}, Const::GenomePreviewConnectionColor, LineThickness);

                auto arrowPartDirection2 = RealVector2D{-direction.x - direction.y, direction.x - direction.y};
                auto arrowPartStart2 = connectionStartPos + arrowPartDirection2 * cellSize / 8;
                drawList->AddLine(
                    {arrowPartStart2.x, arrowPartStart2.y}, {connectionStartPos.x, connectionStartPos.y}, Const::GenomePreviewConnectionColor, LineThickness);
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
                auto textLength = toFloat(text.size());
                auto truncatedSize = std::min(scale(30.0f), cellSize);
                drawList->AddRectFilled(
                    {cellPos.x + truncatedSize * 0.2f, cellPos.y + truncatedSize * 0.1f},
                    {cellPos.x + truncatedSize * 0.32f * textLength + truncatedSize * 0.4f, cellPos.y + truncatedSize * 0.8f},
                    Const::GenomePreviewLinkToGeneBackgroundColor1);
                drawList->AddRect(
                    {cellPos.x + truncatedSize * 0.2f, cellPos.y + truncatedSize * 0.1f},
                    {cellPos.x + truncatedSize * 0.32f * textLength + truncatedSize * 0.4f, cellPos.y + truncatedSize * 0.8f},
                    Const::GenomePreviewLinkToGeneBackgroundColor2);
                AlienGui::AddTextWithSubpixelAccuracy(
                    drawList,
                    style.getSmallBoldFont(),
                    truncatedSize / 1.5f,
                    {cellPos.x + truncatedSize * 0.3f, cellPos.y + truncatedSize * 0.1f},
                    Const::GenomePreviewLinkToGeneTextColor,
                    text.c_str());
            }
        }
    }

    _editData->setSelectedNodeIndex(selectedNode);
}

void _CreaturePreviewWidget::processSignalEditor(bool& phenotypeChanged, Description& phenotype, ConversionResult const& conversionResult)
{
    if (_editData->run || !_editData->detailSimulation || !_selectedCellIdFromPreview.has_value()) {
        return;
    }
    std::optional<CellPreviewDescription> selectedCell;
    for (auto const& cell : conversionResult.description._cells) {
        if (cell._id == _selectedCellIdFromPreview.value()) {
            selectedCell = cell;
            break;
        }
    }
    CHECK(selectedCell.has_value());


    ImGui::SetCursorPos({ImGui::GetScrollX() + ImGui::GetWindowWidth() - scale(220.0f), ImGui::GetScrollY() + scale(13.0f)});
    auto height = selectedCell->_signalState == SignalState_Active ? scale(168.0f) : scale(67.0f);
    if (ImGui::BeginChild("signalEditor", ImVec2(scale(190), height), ImGuiChildFlags_FrameStyle)) {

        AlienGui::Group(AlienGui::GroupParameters().text("Signal editor").highlighted(true));
        int signalState = selectedCell->_signalState; 
        phenotypeChanged |=
            AlienGui::Switcher(AlienGui::SwitcherParameters().name("").values({"No signal", "Fading signal", "Signal"}).textWidth(0), signalState);
        selectedCell->_signalState = static_cast<uint8_t>(signalState);

        if (selectedCell->_signalState == SignalState_Active) {

            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));  // Transparent background
            ImGuiStyle& style = ImGui::GetStyle();
            auto originalGrabMinSize = style.GrabMinSize;
            style.GrabMinSize = scale(8.0f);

            if (!selectedCell->_signal.has_value()) {
                selectedCell->_signal = SignalPreviewDescription();
            }
            auto& channels = selectedCell->_signal->_channels;
            int index = 0;
            if (ImGui::BeginChild("1", ImVec2(scale(85), scale(0)))) {
                for (auto& channel : channels | std::views::take(4)) {
                    phenotypeChanged |= AlienGui::SliderFloat(
                        AlienGui::SliderFloatParameters().name("#" + std::to_string(index)).format("%.3f").textWidth(SignalTextWidth).min(-2.0f).max(2.0f), &channel);
                    ++index;
                }
            }
            ImGui::EndChild();
            ImGui::SameLine();
            if (ImGui::BeginChild("2", ImVec2(scale(85), scale(0)))) {
                for (auto& channel : channels | std::views::drop(4)) {
                    phenotypeChanged |= AlienGui::SliderFloat(
                        AlienGui::SliderFloatParameters().name("#" + std::to_string(index)).format("%.3f").textWidth(SignalTextWidth).min(-2.0f).max(2.0f),
                        &channel);
                    ++index;
                }
            }
            ImGui::EndChild();

            style.GrabMinSize = originalGrabMinSize; 
            ImGui::PopStyleColor();
        }

        if (phenotypeChanged) {
            updatePhenotype(phenotype, selectedCell.value());
        }
    }
    ImGui::EndChild();
}

void _CreaturePreviewWidget::processActionButtons()
{
    ImGui::SetCursorPos({ImGui::GetScrollX() + scale(10.0f), ImGui::GetScrollY() + ImGui::GetWindowHeight() - scale(40.0f)});
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
    }
    ImGui::EndChild();
}

void _CreaturePreviewWidget::processScrollbars()
{
    RealVector2D windowSize{ImGui::GetWindowWidth(), ImGui::GetWindowHeight()};
    RealVector2D windowPos{ImGui::GetWindowPos().x, ImGui::GetWindowPos().y};

    RealRect worldRect{{-100.0f, -100.0f}, {100.0f, 100.0f}};
    RealRect visibleWorldRect{
        mapViewToWorldPosition(windowPos, windowSize, windowPos),
        mapViewToWorldPosition(windowPos + windowSize, windowSize, windowPos),
    };
    RealRect viewRect{windowPos, windowPos + windowSize};
    _scrollbars->process(_worldCenter, worldRect, visibleWorldRect, viewRect);
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
        (viewPos.y - viewStartPos.y - viewSize.y / 2) / scaleFactor + _worldCenter.y};
}

void _CreaturePreviewWidget::moveCenter(
    RealVector2D const& startWorldPosition,
    RealVector2D const& endViewPos,
    RealVector2D const& viewSize,
    RealVector2D const& viewStartPos)
{
    auto deltaViewPos = endViewPos - viewStartPos - viewSize / 2.0f;
    auto deltaWorldPos = deltaViewPos / _zoom;
    _worldCenter = startWorldPosition - deltaWorldPos;
}

void _CreaturePreviewWidget::updatePhenotype(Description& phenotype, CellPreviewDescription const& editedCell) const
{
    phenotype.forEachCell([&](CellDescription& cell) {
        if (cell._id == editedCell._id) {
            cell._signalState = editedCell._signalState;
            if (editedCell._signalState == SignalState_Active) {
                auto signalDesc = SignalDescription().channels(editedCell._signal.value()._channels);
                cell._signal = signalDesc;
            } else {
                cell._signal.reset();
            }
        }
    });
}
