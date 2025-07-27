#include "PreviewDescriptionWidget.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/Math.h"
#include "EngineInterface/Colors.h"

#include "AlienGui.h"
#include "StyleRepository.h"

PreviewDescriptionWidget _PreviewDescriptionWidget::create()
{
    return PreviewDescriptionWidget(new _PreviewDescriptionWidget());
}

bool _PreviewDescriptionWidget::process(PreviewDescription const& desc)
{
    auto constexpr ZoomLevelForLabels = 16.0f;
    auto constexpr ZoomLevelForConnections = 8.0f;
    auto const LineThickness = StyleRepository::get().scale(2.0f);

    auto result = false;

    if (ImGui::BeginChild("preview", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar)) {

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        auto const cellSize = StyleRepository::get().scale(_zoom);

        auto drawTextWithShadow = [&drawList, &cellSize](std::string const& text, float posX, float posY) {
            drawList->AddText(
                StyleRepository::get().getLargeFont(), cellSize / 2, {posX + 1.0f, posY + 1.0f}, Const::ExecutionNumberOverlayShadowColor, text.c_str());
            drawList->AddText(StyleRepository::get().getLargeFont(), cellSize / 2, {posX, posY}, Const::ExecutionNumberOverlayColor, text.c_str());
        };

        auto color = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
        auto windowPos = ImGui::GetWindowPos();
        auto windowSize = ImGui::GetWindowSize();

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
        RealVector2D previewSize = (lowerRight - upperLeft) * cellSize + RealVector2D(cellSize, cellSize) * 2;

        auto mousePos = ImGui::GetMousePos();
        auto clickedOnPreviewWindow = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mousePos.x >= windowPos.x && mousePos.y >= windowPos.y
            && mousePos.x <= windowPos.x + windowSize.x && mousePos.y <= windowPos.y + windowSize.y;
        ImGui::SetCursorPos({std::max(0.0f, windowSize.x - previewSize.x) / 2, std::max(0.0f, windowSize.y - previewSize.y) / 2});
        if (ImGui::BeginChild("##genome", ImVec2(previewSize.x, previewSize.y), false, ImGuiWindowFlags_HorizontalScrollbar)) {

            auto windowPos = ImGui::GetWindowPos();
            RealVector2D offset{windowPos.x + cellSize, windowPos.y + cellSize};

            ImGui::SetCursorPos({previewSize.x - 1, previewSize.y - 1});

            //draw cells
            for (auto const& cell : desc._cells) {
                auto cellPos = (cell._pos - upperLeft) * cellSize + offset;
                float h, s, v;
                AlienGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell._color], h, s, v);

                auto cellRadiusFactor = _zoom > ZoomLevelForConnections ? 0.25f : 0.5f;
                drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * cellRadiusFactor, ImColor::HSV(h, s * 1.2f, v * 1.0f));

                if (_selectedNode && cell._nodeIndex == *_selectedNode) {
                    if (_zoom > ZoomLevelForLabels) {
                        drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor(1.0f, 1.0f, 1.0f));
                    } else {
                        drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor::HSV(h, s * 0.8f, v * 1.2f));
                    }
                }

                if (clickedOnPreviewWindow) {
                    if (mousePos.x >= cellPos.x - cellSize / 2 && mousePos.y >= cellPos.y - cellSize / 2 && mousePos.x <= cellPos.x + cellSize / 2
                        && mousePos.y <= cellPos.y + cellSize / 2) {
                        _selectedNode = cell._nodeIndex;
                        result = true;
                    }
                }
            }

            //draw symbols
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

            //draw cell connections
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

            //draw cell infos (start/end marks and multiple constructor marks)
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

        //zoom buttons
        ImGui::SetCursorPos({ImGui::GetScrollX() + StyleRepository::get().scale(10), ImGui::GetScrollY() + windowSize.y - StyleRepository::get().scale(40)});
        if (ImGui::BeginChild("##buttons", ImVec2(StyleRepository::get().scale(100), StyleRepository::get().scale(30)), false)) {
            ImGui::SetCursorPos({0, 0});
            ImGui::PushStyleColor(ImGuiCol_Button, color);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, color);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, color);
            ImGui::PushID(1);
            if (ImGui::Button(ICON_FA_SEARCH_PLUS)) {
                _zoom *= 1.5f;
            }
            ImGui::PopID();
            ImGui::SameLine();
            ImGui::PushID(2);
            if (ImGui::Button(ICON_FA_SEARCH_MINUS)) {
                _zoom /= 1.5f;
            }
            ImGui::PopID();
            ImGui::PopStyleColor(3);
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();
    return result;
}

float _PreviewDescriptionWidget::getZoom() const
{
    return _zoom;
}

void _PreviewDescriptionWidget::setZoom(float zoom)
{
    _zoom = zoom;
}

std::optional<int> _PreviewDescriptionWidget::getSelectedNode() const
{
    return _selectedNode;
}

void _PreviewDescriptionWidget::setSelectedNode(std::optional<int> selectedNode)
{
    _selectedNode = selectedNode;
}

_PreviewDescriptionWidget::_PreviewDescriptionWidget()
{
}