#include "CreaturePreviewWidget.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/Math.h"
#include "Base/StringHelper.h"

#include "EngineInterface/Colors.h"
#include "EngineInterface/PreviewDescriptionConverterService.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "StyleRepository.h"

CreaturePreviewWidget _CreaturePreviewWidget::create(
    GenomeTabEditData const& editData,
    GeneIndicesForSubGenome const& geneIndices,
    GenomeDescriptionWithStartGeneIndex const& genomeWithStartIndex)
{
    return CreaturePreviewWidget(new _CreaturePreviewWidget(editData, geneIndices, genomeWithStartIndex));
}

bool _CreaturePreviewWidget::process(CollectionDescription&& phenotype)
{
    auto geneStartIndex = _genomeWithStartIndex.startIndex;
    auto conversionResult =
        PreviewDescriptionConverterService::get().convert(_editData->genome, std::move(phenotype), geneStartIndex, _visualFrontAngle);
    auto& desc = conversionResult.description;
    _visualFrontAngle = conversionResult.visualFrontAngle;

    auto constexpr ZoomLevelForLabels = 16.0f;
    auto constexpr ZoomLevelForConnections = 8.0f;
    auto const LineThickness = scale(2.0f);

    auto result = false;

    auto& selectedGene = _editData->selectedGeneIndex;
    auto selectedNode = _editData->getSelectedNodeIndex();

    auto windowSize = ImGui::GetWindowSize();
    if (ImGui::BeginChild("outerPreview", ImVec2(0, 0), 0, ImGuiWindowFlags_AlwaysHorizontalScrollbar)) {

        auto const& style = StyleRepository::get();
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        auto const cellSize = scale(_zoom);

        //auto drawTextWithShadow = [&drawList, &cellSize](std::string const& text, float posX, float posY) {
        //    drawList->AddText(
        //        StyleRepository::get().getLargeFont(), cellSize / 2, {posX + 1.0f, posY + 1.0f}, Const::ExecutionNumberOverlayShadowColor, text.c_str());
        //    drawList->AddText(StyleRepository::get().getLargeFont(), cellSize / 2, {posX, posY}, Const::ExecutionNumberOverlayColor, text.c_str());
        //};

        //auto color = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
        auto windowPos = ImGui::GetWindowPos();

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
        RealVector2D previewSize = RealVector2D{200.0f, 200.0f} * cellSize;  //(lowerRight - upperLeft) * cellSize + RealVector2D(cellSize, cellSize) * 2;
        //ImGui::SetCursorPos({(previewSize.x - 1) / 2, (previewSize.y - 1) / 2});
        if (!_lastWindowSize.has_value()) {
            //float centerX = ImGui::GetScrollMaxX() / 2.0f;
            //float centerY = ImGui::GetScrollMaxY() / 2.0f;
            float centerX = (previewSize.x - windowSize.x + scale(40.0f)) / 2.0f;
            float centerY = (previewSize.y - windowSize.y + scale(70.0f)) / 2.0f;
            ImGui::SetScrollX(centerX);
            ImGui::SetScrollY(centerY);
        } else {
            auto deltaX = windowSize.x - _lastWindowSize->x;
            auto deltaY = windowSize.y - _lastWindowSize->y;
            auto scrollX = ImGui::GetScrollX();
            auto scrollY = ImGui::GetScrollY();
            ImGui::SetScrollX(scrollX - deltaX / 2);
            ImGui::SetScrollY(scrollY - deltaY / 2);
        }
        _lastWindowSize = RealVector2D{windowSize.x, windowSize.y};

        auto mousePos = ImGui::GetMousePos();
        auto clickedOnPreviewWindow = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mousePos.x >= windowPos.x && mousePos.y >= windowPos.y
            && mousePos.x <= windowPos.x + windowSize.x && mousePos.y <= windowPos.y + windowSize.y;
        //ImGui::SetCursorPos({std::max(0.0f, windowSize.x - previewSize.x) / 2, std::max(0.0f, windowSize.y - previewSize.y) / 2});
        if (ImGui::BeginChild("innerPreview", ImVec2(previewSize.x, previewSize.y), 0, ImGuiWindowFlags_NoScrollbar)) {

            auto windowPos = ImGui::GetWindowPos();
            RealVector2D offset{windowPos.x + cellSize, windowPos.y + cellSize};

            // Draw selected gene
            auto selectedGeneColor = ImColor::HSV(0, 0, 0.15f);
            for (auto const& cell : desc._cells) {
                auto cellPos = (cell._pos + RealVector2D{100.0f, 100.0f}) * cellSize + offset;
                if (selectedGene.has_value() && cell._geneIndex == selectedGene.value()) {
                    drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.6f, selectedGeneColor);
                }
            }

            // Draw cells and selected cells
            for (auto const& cell : desc._cells) {
                auto cellPos = (cell._pos + RealVector2D{100.0f, 100.0f}) * cellSize + offset;
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
                        result = true;
                    }
                }
            }

            // Draw signal restrictions
            if (_zoom > ZoomLevelForConnections) {
                for (auto const& cell : desc._cells) {
                    auto cellPos = (cell._pos + RealVector2D{100.0f, 100.0f}) * cellSize + offset;
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

            // Draw symbols
            //for (auto const& symbol : desc.symbols) {
            //    auto pos = (symbol.pos - upperLeft) * cellSize + offset;
            //    switch (symbol.type) {
            //    case SymbolPreviewDescription::Type::Dot: {
            //        auto cellRadiusFactor = zoom > ZoomLevelForConnections ? 0.15f : 0.35f;
            //        drawList->AddCircleFilled({pos.x, pos.y}, cellSize * cellRadiusFactor, Const::GenomePreviewDotSymbolColor);
            //    } break;
            //    case SymbolPreviewDescription::Type::Infinity: {
            //        if (zoom > ZoomLevelForConnections) {
            //            drawList->AddText(
            //                StyleRepository::get().getIconFont(),
            //                cellSize / 2,
            //                {pos.x - cellSize * 0.4f, pos.y - cellSize * 0.2f},
            //                Const::GenomePreviewInfinitySymbolColor,
            //                ICON_FA_INFINITY);
            //        }
            //    } break;
            //    }
            //}

            // Draw cell connections
            if (_zoom > ZoomLevelForConnections) {
                for (auto const& connection : desc._connections) {
                    auto cellPos1 = (connection._cell1 + RealVector2D{100.0f, 100.0f}) * cellSize + offset;
                    auto cellPos2 = (connection._cell2 + RealVector2D{100.0f, 100.0f}) * cellSize + offset;

                    auto direction = cellPos1 - cellPos2;

                    Math::normalize(direction);
                    auto connectionStartPos = cellPos1 - direction * cellSize / 4;
                    auto connectionEndPos = cellPos2 + direction * cellSize / 4;
                    drawList->AddLine(
                        {connectionStartPos.x, connectionStartPos.y},
                        {connectionEndPos.x, connectionEndPos.y},
                        Const::GenomePreviewConnectionColor,
                        LineThickness);

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
                            {arrowPartStart1.x, arrowPartStart1.y},
                            {connectionEndPos.x, connectionEndPos.y},
                            Const::GenomePreviewConnectionColor,
                            LineThickness);

                        auto arrowPartDirection2 = RealVector2D{direction.x + direction.y, -direction.x + direction.y};
                        auto arrowPartStart2 = connectionEndPos + arrowPartDirection2 * cellSize / 8;
                        drawList->AddLine(
                            {arrowPartStart2.x, arrowPartStart2.y},
                            {connectionEndPos.x, connectionEndPos.y},
                            Const::GenomePreviewConnectionColor,
                            LineThickness);
                    }
                }
            }

            // Draw gene references
            if (_zoom > ZoomLevelForLabels) {
                for (auto const& cell : desc._cells) {
                    if (cell._constructorGeneIndex.has_value()) {
                        auto cellPos = (cell._pos + RealVector2D{100.0f, 100.0f}) * cellSize + offset;
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
                            Const::GenomePreviewLinkToGeneTextColor, text.c_str());
                    }
                }
            }


            // Draw cell infos (start/end marks and multiple constructor marks)
            //if (zoom > ZoomLevelForLabels) {
            //    for (auto const& cell : desc._cells) {
            //        auto cellPos = (cell._pos - upperLeft) * cellSize + offset;
            //        auto length = cellSize / 4;
            //        if (cell.partStart !=  cell.partEnd) {
            //            drawList->AddTriangleFilled(
            //                {cellPos.x + length, cellPos.y},
            //                {cellPos.x + length * 2, cellPos.y - length / 2},
            //                {cellPos.x + length * 2, cellPos.y + length / 2},
            //                cell.partStart ? Const::GenomePreviewStartColor : Const::GenomePreviewEndColor);
            //        }
            //        if (cell.partStart && cell.partEnd) {
            //            drawList->AddTriangleFilled(
            //                {cellPos.x + length, cellPos.y - length},
            //                {cellPos.x + length * 2, cellPos.y - length * 3  / 2},
            //                {cellPos.x + length * 2, cellPos.y - length / 2},
            //                Const::GenomePreviewStartColor);
            //            drawList->AddTriangleFilled(
            //                {cellPos.x + length, cellPos.y + length},
            //                {cellPos.x + length * 2, cellPos.y + length / 2},
            //                {cellPos.x + length * 2, cellPos.y + length * 3 / 2},
            //                Const::GenomePreviewEndColor);
            //        }
            //        if (cell.multipleConstructor) {
            //            drawList->AddLine(
            //                {cellPos.x + length, cellPos.y + length},
            //                {cellPos.x + length * 2, cellPos.y + length},
            //                Const::GenomePreviewMultipleConstructorColor,
            //                LineThickness);
            //            drawList->AddLine(
            //                {cellPos.x + length * 1.5f, cellPos.y + length / 2},
            //                {cellPos.x + length * 1.5f, cellPos.y + length * 1.5f},
            //                Const::GenomePreviewMultipleConstructorColor,
            //                LineThickness);
            //        }
            //        if (cell.selfReplicator) {
            //            drawList->AddText(
            //                StyleRepository::get().getIconFont(),
            //                cellSize / 4,
            //                {cellPos.x - length * 2, cellPos.y + length},
            //                Const::GenomePreviewSelfReplicatorColor,
            //                ICON_FA_CLONE);
            //        }
            //    }
            //}
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();

    // Action buttons
    ImGui::SetCursorPos({ImGui::GetScrollX() + scale(10), ImGui::GetScrollY() + windowSize.y - scale(40)});
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

    _editData->setSelectedNodeIndex(selectedNode);
    return result;
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
