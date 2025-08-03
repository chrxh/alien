#include "PreviewDescriptionWidget.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/Math.h"
#include "Base/StringHelper.h"

#include "EngineInterface/Colors.h"

#include "AlienGui.h"
#include "PreviewDescriptionWidgetSettings.h"
#include "StyleRepository.h"

PreviewDescriptionWidget _PreviewDescriptionWidget::create(PreviewDescriptionSettings const& settings)
{
    return PreviewDescriptionWidget(new _PreviewDescriptionWidget(settings));
}

bool _PreviewDescriptionWidget::process(int tps, PreviewDescription const& desc)
{
    auto constexpr ZoomLevelForLabels = 16.0f;
    auto constexpr ZoomLevelForConnections = 8.0f;
    auto const LineThickness = scale(2.0f);

    auto result = false;

    auto windowSize = ImGui::GetWindowSize();
    if (ImGui::BeginChild("outerPreview", ImVec2(0, 0), 0, ImGuiWindowFlags_AlwaysHorizontalScrollbar)) {

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        auto const cellSize = scale(_zoom);

        auto drawTextWithShadow = [&drawList, &cellSize](std::string const& text, float posX, float posY) {
            drawList->AddText(
                StyleRepository::get().getLargeFont(), cellSize / 2, {posX + 1.0f, posY + 1.0f}, Const::ExecutionNumberOverlayShadowColor, text.c_str());
            drawList->AddText(StyleRepository::get().getLargeFont(), cellSize / 2, {posX, posY}, Const::ExecutionNumberOverlayColor, text.c_str());
        };

        auto color = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
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

        auto mousePos = ImGui::GetMousePos();
        auto clickedOnPreviewWindow = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mousePos.x >= windowPos.x && mousePos.y >= windowPos.y
            && mousePos.x <= windowPos.x + windowSize.x && mousePos.y <= windowPos.y + windowSize.y;
        //ImGui::SetCursorPos({std::max(0.0f, windowSize.x - previewSize.x) / 2, std::max(0.0f, windowSize.y - previewSize.y) / 2});
        if (ImGui::BeginChild("innerPreview", ImVec2(previewSize.x, previewSize.y), 0, ImGuiWindowFlags_NoScrollbar)) {

            auto windowPos = ImGui::GetWindowPos();
            RealVector2D offset{windowPos.x + cellSize, windowPos.y + cellSize};

            //ImGui::SetCursorPos({(previewSize.x - 1) / 2, (previewSize.y - 1) / 2});

            // Draw selected gene
            auto selectedGeneColor = ImColor::HSV(0, 0, 0.15f);
            for (auto const& cell : desc._cells) {
                auto cellPos = (cell._pos - upperLeft) * cellSize + offset;
                if (_selectedGene.has_value() && cell._geneIndex == _selectedGene.value()) {
                    drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * 0.6f, selectedGeneColor);
                }
            }

            // Draw cells and selected cells
            for (auto const& cell : desc._cells) {
                auto cellPos = (cell._pos - upperLeft) * cellSize + offset;
                float h, s, v;
                AlienGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell._color], h, s, v);

                auto cellRadiusFactor = _zoom > ZoomLevelForConnections ? 0.25f : 0.5f;
                drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * cellRadiusFactor, ImColor::HSV(h, s * 1.2f, v * 1.0f));

                if (_selectedGene.has_value() && _selectedNode.has_value() && cell._geneIndex == _selectedGene.value() && cell._nodeIndex == _selectedNode.value()) {
                    if (_zoom > ZoomLevelForLabels) {
                        drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor(1.0f, 1.0f, 1.0f));
                    } else {
                        drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor::HSV(h, s * 0.8f, v * 1.2f));
                    }
                }

                if (clickedOnPreviewWindow) {
                    if (mousePos.x >= cellPos.x - cellSize / 2 && mousePos.y >= cellPos.y - cellSize / 2 && mousePos.x <= cellPos.x + cellSize / 2
                        && mousePos.y <= cellPos.y + cellSize / 2) {
                        _selectedGene = cell._geneIndex;
                        _selectedNode = cell._nodeIndex;
                        result = true;
                    }
                }
            }

            // Draw signal restrictions
            if (_zoom > ZoomLevelForConnections) {
                for (auto const& cell : desc._cells) {
                    auto cellPos = (cell._pos - upperLeft) * cellSize + offset;
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
                    auto cellPos1 = (connection._cell1 - upperLeft) * cellSize + offset;
                    auto cellPos2 = (connection._cell2 - upperLeft) * cellSize + offset;

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

    // Draw timestep in bottom right corner of "preview" child window
    ImGui::SetCursorPos({ImGui::GetScrollX() + windowSize.x - scale(100), ImGui::GetScrollY() + windowSize.y - scale(40)});
    if (ImGui::BeginChild("##TPS", ImVec2(scale(100), scale(30)), false)) {
        AlienGui::Text("TPS: " + StringHelper::format(tps));
    }
    ImGui::EndChild();

    // Action buttons
    ImGui::SetCursorPos({ImGui::GetScrollX() + scale(10), ImGui::GetScrollY() + windowSize.y - scale(40)});
    if (ImGui::BeginChild("##buttons", ImVec2(scale(105), scale(30)), false)) {
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
        AlienGui::VerticalSeparator(23);
        ImGui::SameLine();
        ImGui::PushID(3);
        if (AlienGui::ActionButton(AlienGui::ActionButtonParameters().buttonText(ICON_FA_FORWARD).highlighted(_settings->maxSpeed))) {
            _settings->maxSpeed = !_settings->maxSpeed;
        }
        ImGui::PopID();
    }
    ImGui::EndChild();
    return result;
}

std::optional<int> _PreviewDescriptionWidget::getSelectedGene() const
{
    return _selectedGene;
}

void _PreviewDescriptionWidget::setSelectedGene(std::optional<int> selectedGene)
{
    _selectedGene = selectedGene;
}

std::optional<int> _PreviewDescriptionWidget::getSelectedNode() const
{
    return _selectedNode;
}

void _PreviewDescriptionWidget::setSelectedNode(std::optional<int> selectedNode)
{
    _selectedNode = selectedNode;
}

_PreviewDescriptionWidget::_PreviewDescriptionWidget(PreviewDescriptionSettings const& settings)
    : _settings(settings)
{}
