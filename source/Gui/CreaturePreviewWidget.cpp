#include "CreaturePreviewWidget.h"

#include <cmath>
#include <ranges>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/sliced.hpp>

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/Math.h>
#include <Base/StringHelper.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/Colors.h>
#include <EngineInterface/PreviewDescConverterService.h>
#include <EngineInterface/SpaceCalculator.h>

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeWindowEditData.h"
#include "SimulationScrollbars.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr ZoomLevelForGeneReferences = 16.0f;
    auto constexpr ZoomLevelForNodeIndices = 32.0f;
    auto constexpr ZoomLevelForConnections = 8.0f;
    auto constexpr SignalTextWidth = 40.0f;
}


CreaturePreviewWidget _CreaturePreviewWidget::create(
    GenomeWindowEditData const& genomeEditData,
    GenomeTabEditData const& editData,
    GeneIndicesForSubGenome const& geneIndices,
    SubGenomeDesc const& genomeWithStartIndex)
{
    return CreaturePreviewWidget(new _CreaturePreviewWidget(genomeEditData, editData, geneIndices, genomeWithStartIndex));
}

void _CreaturePreviewWidget::process(bool& phenotypeChanged, Desc& phenotype, float width)
{
    auto phenotypeWithoutSeed = phenotype;
    GenomeDescEditService::get().removeSeedFromPhenotype(phenotypeWithoutSeed);

    auto geneStartIndex = _subGenome.startIndex;

    auto conversionResult =
        PreviewDescConverterService::get().convertToPreviewDesc(_editData->genome, geneStartIndex, std::move(phenotypeWithoutSeed));

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

SubGenomeDesc const& _CreaturePreviewWidget::getGenomeWithStartIndex() const
{
    return _subGenome;
}

void _CreaturePreviewWidget::setGenomeWithStartIndex(SubGenomeDesc const& value)
{
    _subGenome = value;
}

_CreaturePreviewWidget::_CreaturePreviewWidget(
    GenomeWindowEditData const& genomeEditData,
    GenomeTabEditData const& editData,
    GeneIndicesForSubGenome const& geneIndices,
    SubGenomeDesc const& genomeWithStartIndex)
    : _genomeEditData(genomeEditData)
    , _editData(editData)
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
    auto const LineThickness = scale(1.0f);
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
    for (auto const& object : desc._objects) {
        if (_selectedCellIdFromPreview.has_value() && _selectedCellIdFromPreview.value() == object._id) {
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
        for (auto const& object : desc._objects) {
            maxDistance = std::max(maxDistance, Math::length(object._pos));
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

    // Draw selected gene
    auto selectedGeneColor = ImColor::HSV(0.66f, 0.5f, 0.1f);
    for (auto const& object : desc._objects) {
        auto cellPos = mapWorldToViewPosition(object._pos, windowSize, windowPos);
        if (selectedGene.has_value() && object._geneIndex == selectedGene.value()) {
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.6f, selectedGeneColor);
        }
    }

    // Draw selected nodes
    for (auto const& object : desc._objects) {
        auto cellPos = mapWorldToViewPosition(object._pos, windowSize, windowPos);
        if (selectedGene.has_value() && selectedNode.has_value() && object._geneIndex == selectedGene.value() && object._nodeIndex == selectedNode.value()) {
            float h, s, v;
            AlienGui::ConvertRGBtoHSV(Const::IndividualObjectColors[object._color], h, s, v);
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.4f, ImColor::HSV(h, 0.5f, 0.4f));
        }
    }

    // Draw cells and selected cells
    for (auto const& object : desc._objects) {
        auto cellPos = mapWorldToViewPosition(object._pos, windowSize, windowPos);
        float h, s, v;
        uint32_t color = object._color != -1 ? Const::IndividualObjectColors[object._color] : 0x707070;
        AlienGui::ConvertRGBtoHSV(color, h, s, v);

        auto cellRadiusFactor = _zoom > ZoomLevelForConnections ? 0.15f : 0.5f;
        drawList->AddCircleFilled({cellPos.x, cellPos.y}, std::max(1.0f, cellSize * cellRadiusFactor), ImColor::HSV(h, s * 1.2f, v * 1.0f));

        if (_selectedCellIdFromPreview.has_value() && _selectedCellIdFromPreview.value() == object._id) {
            if (_zoom > ZoomLevelForGeneReferences) {
                drawList->AddCircle({cellPos.x, cellPos.y}, cellSize * 0.15f, ImColor::HSV(0, 0, 1, 0.7f), 0, 2.0f /*cellSize * 0.05f*/);
            }
        }

        if (clickedOnPreviewWindow) {
            if (mousePos.x >= cellPos.x - cellSize / 2 && mousePos.y >= cellPos.y - cellSize / 2 && mousePos.x <= cellPos.x + cellSize / 2
                && mousePos.y <= cellPos.y + cellSize / 2) {
                selectedGene = object._geneIndex;
                selectedNode = object._nodeIndex;
                _selectedNodeFromPreview = selectedNode;
                _selectedCellIdFromPreview = object._id;
            }
        }
    }

    // Draw node indices or cell functions
    if (_zoom > ZoomLevelForNodeIndices) {
        for (auto const& object : desc._objects) {
            auto cellPos = mapWorldToViewPosition(object._pos, windowSize, windowPos);
            std::string text;
            if (_genomeEditData->showNodeIndex) {
                text = std::to_string(object._nodeIndex + 1);
            } else {
                text = Const::CellTypeStrings.at(object._cellType);
            }
            auto fontSize = cellSize * 0.18f;
            auto textWidth = fontSize * toFloat(text.size()) * 0.55f;
            AlienGui::AddTextWithSubpixelAccuracy(
                drawList,
                style.getSmallBoldFont(),
                fontSize,
                {cellPos.x - textWidth / 2, cellPos.y - fontSize / 2},
                ImColor::HSV(0, 0, 1.0f, 0.7f),
                text.c_str());
        }
    }

    // Draw signals
    if (_zoom > ZoomLevelForConnections && _editData->detailSimulation) {
        for (auto const& object : desc._objects) {
            auto cellPos = mapWorldToViewPosition(object._pos, windowSize, windowPos);
            auto constexpr cellRadiusFactor = 0.11f;
            float radius = cellSize * cellRadiusFactor;

            auto signalStrength = 0.0f;
            for (auto const& ch : object._signal._channels) {
                signalStrength += std::abs(ch);
            }
            signalStrength = std::min(1.0f, static_cast<float>(sqrt(sqrt(signalStrength)) / 2));
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, radius, ImColor::HSV(0, 0, 1.0f, signalStrength));
            drawList->AddCircle({cellPos.x, cellPos.y}, radius, ImColor::HSV(0, 0, 0.2f, signalStrength));
        }
    }

    // Draw cell connections and connection weights
    if (_zoom > ZoomLevelForConnections) {
        for (auto const& connection : desc._connections) {
            auto cellPos1 = mapWorldToViewPosition(connection._object1, windowSize, windowPos);
            auto cellPos2 = mapWorldToViewPosition(connection._object2, windowSize, windowPos);

            auto direction = cellPos1 - cellPos2;

            Math::normalize(direction);
            auto connectionStartPos = cellPos1 - direction * cellSize * 0.15f;
            auto connectionEndPos = cellPos2 + direction * cellSize * 0.15f;
            drawList->AddLine(
                {connectionStartPos.x, connectionStartPos.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);

            if (connection._connectionWeightToObject1 != 0.0f) {
                auto arrowScale = std::min(std::abs(connection._connectionWeightToObject1), 1.0f);
                auto arrowPartDirection1 = RealVector2D{-direction.x + direction.y, -direction.x - direction.y};
                auto arrowPartStart1 = connectionStartPos + arrowPartDirection1 * cellSize / 8 * arrowScale;
                drawList->AddLine(
                    {arrowPartStart1.x, arrowPartStart1.y}, {connectionStartPos.x, connectionStartPos.y}, Const::GenomePreviewConnectionColor, LineThickness);

                auto arrowPartDirection2 = RealVector2D{-direction.x - direction.y, direction.x - direction.y};
                auto arrowPartStart2 = connectionStartPos + arrowPartDirection2 * cellSize / 8 * arrowScale;
                drawList->AddLine(
                    {arrowPartStart2.x, arrowPartStart2.y}, {connectionStartPos.x, connectionStartPos.y}, Const::GenomePreviewConnectionColor, LineThickness);
            }

            if (connection._connectionWeightToObject2 != 0.0f) {
                auto arrowScale = std::min(std::abs(connection._connectionWeightToObject2), 1.0f);
                auto arrowPartDirection1 = RealVector2D{direction.x - direction.y, direction.x + direction.y};
                auto arrowPartStart1 = connectionEndPos + arrowPartDirection1 * cellSize / 8 * arrowScale;
                drawList->AddLine(
                    {arrowPartStart1.x, arrowPartStart1.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);

                auto arrowPartDirection2 = RealVector2D{direction.x + direction.y, -direction.x + direction.y};
                auto arrowPartStart2 = connectionEndPos + arrowPartDirection2 * cellSize / 8 * arrowScale;
                drawList->AddLine(
                    {arrowPartStart2.x, arrowPartStart2.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);
            }
        }
    }

    // Draw gene references
    if (_zoom > ZoomLevelForGeneReferences) {
        for (auto const& object : desc._objects) {
            if (object._constructorGeneIndex.has_value()) {
                auto cellPos = mapWorldToViewPosition(object._pos, windowSize, windowPos);
                auto text = std::to_string(object._constructorGeneIndex.value() + 1);
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

void _CreaturePreviewWidget::processSignalEditor(bool& phenotypeChanged, Desc& phenotype, ConversionResult const& conversionResult)
{
    auto width = _editData->detailSimulation && _selectedCellIdFromPreview.has_value() ? scale(410) : scale(250);
    auto height = _editData->detailSimulation && _selectedCellIdFromPreview.has_value() ? scale(149.0f) : scale(67.0f);
    auto contentAvailable = ImGui::GetContentRegionAvail();
    if (contentAvailable.x < scale(480.0f) || contentAvailable.y < scale(250.0f)) {
        return;
    }
    ImGui::SetCursorPos({ImGui::GetScrollX() + ImGui::GetWindowWidth() - width - scale(30.0f), ImGui::GetScrollY() + scale(13.0f)});
    if (ImGui::BeginChild("signalEditor", ImVec2(width, height), ImGuiChildFlags_FrameStyle)) {

        AlienGui::Group(AlienGui::GroupParameters().text("Signal editor"));

        if (_editData->detailSimulation && _selectedCellIdFromPreview.has_value()) {
            std::optional<CellPreviewDesc> selectedCell;
            for (auto const& object : conversionResult.description._objects) {
                if (object._id == _selectedCellIdFromPreview.value()) {
                    selectedCell = object;
                    break;
                }
            }
            CHECK(selectedCell.has_value());

            ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));  // Transparent background
            ImGuiStyle& style = ImGui::GetStyle();
            auto originalGrabMinSize = style.GrabMinSize;
            style.GrabMinSize = scale(8.0f);

            auto& channels = selectedCell->_signal._channels;
            int index = 0;
            for (int i = 0; i < NEURONS_PER_CELL / 4; ++i) {
                ImGui::PushID(i);
                if (ImGui::BeginChild("", ImVec2(scale(95), scale(0)))) {
                    for (int j = 0; j < 4; ++j) {
                        auto& channel = channels.at(i * 4 + j);
                        phenotypeChanged |= AlienGui::SliderFloat(
                            AlienGui::SliderFloatParameters().name("#" + std::to_string(index)).format("%.3f").textWidth(SignalTextWidth).min(-2.0f).max(2.0f),
                            &channel);
                        ++index;
                    }
                }
                ImGui::EndChild();
                ImGui::PopID();
                if (i < NEURONS_PER_CELL / 4 - 1) {
                    ImGui::SameLine();
                }
            }

            style.GrabMinSize = originalGrabMinSize;
            ImGui::PopStyleColor();

            if (phenotypeChanged) {
                updatePhenotype(phenotype, selectedCell.value());
            }
        } else {
            if (!_editData->detailSimulation) {
                AlienGui::Text("Detailed simulation mode disabled");
            } else if (!_selectedCellIdFromPreview.has_value()) {
                AlienGui::Text("No cell selected");
            }
        }
    }
    ImGui::EndChild();
}

void _CreaturePreviewWidget::processActionButtons()
{
    ImGui::SetCursorPos({ImGui::GetScrollX() + scale(10.0f), ImGui::GetScrollY() + ImGui::GetWindowHeight() - scale(40.0f)});
    if (ImGui::BeginChild("##buttons", ImVec2(scale(160), scale(30)), 0)) {
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
        ImGui::SameLine();
        ImGui::PushID(3);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(_genomeEditData->showNodeIndex ? "123" : "Abc"))) {
            _genomeEditData->showNodeIndex = !_genomeEditData->showNodeIndex;
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

void _CreaturePreviewWidget::updatePhenotype(Desc& phenotype, CellPreviewDesc const& editedCell) const
{
    for (auto& object : phenotype._objects) {
        if (object._id == editedCell._id) {
            object.getCellRef()._signal = SignalDesc().channels(editedCell._signal._channels);
        }
    }
}
