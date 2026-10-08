#include "GeneGraphWidget.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <ranges>

#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <Base/StringHelper.h>

#include <Data/GenomeDescAccessService.h>

#include "AlienGui.h"
#include "GenomeIssueDescription.h"
#include "GenomeTabEditData.h"
#include "StyleService.h"

namespace
{
    auto constexpr BoxWidth = 96.0f;
    auto constexpr BoxPadding = 4.0f;
    auto constexpr BoxRounding = 3.0f;
    auto constexpr SelectionGap = 2.5f;
    auto constexpr SelectionThickness = 1.5f;
    auto constexpr HorizontalGap = 16.0f;
    auto constexpr VerticalGap = 32.0f;
    auto constexpr Margin = 10.0f;
    auto constexpr BendSize = 14.0f;
    auto constexpr ArrowLength = 10.0f;
    auto constexpr ArrowWidth = 9.0f;
    auto constexpr DashLength = 4.0f;
    auto constexpr EdgeThickness = 1.3f;
    auto constexpr HighlightedEdgeThickness = 2.4f;
    auto constexpr EdgeHoverDistance = 5.0f;
    auto constexpr NumCurveSegments = 32;

    struct GeneEdge
    {
        GenomeDescAccessService::GeneConstruction construction;
        std::vector<GenomeIssue> issues;
    };

    struct GraphLayout
    {
        std::vector<int> layers;
        std::vector<ImVec2> boxOffsets;
        ImVec2 boxSize;
        float boxAreaWidth = 0;
        ImVec2 canvasSize;
    };

    struct BezierCurve
    {
        ImVec2 start;
        ImVec2 control1;
        ImVec2 control2;
        ImVec2 end;
    };

    bool isConstructedByNode(GeneEdge const& edge, int nodeIndex)
    {
        auto const& nodeIndices = edge.construction.constructorNodeIndices;
        return std::ranges::find(nodeIndices, nodeIndex) != nodeIndices.end();
    }

    bool turnsOffConstructorOrSeparation(GenomeIssue const& issue, GeneEdge const& edge)
    {
        if (issue.geneIndex != edge.construction.constructingGeneIndex) {
            return false;
        }
        switch (issue.type) {
        case GenomeIssueType::CycleAvoidingRootGene:
        case GenomeIssueType::TooManyGenesWithSeparation:
        case GenomeIssueType::ConstructsRemovedGene:
            return issue.nodeIndex.has_value() && isConstructedByNode(edge, issue.nodeIndex.value());
        default:
            return std::ranges::any_of(issue.cutOffNodeIndices, [&](int nodeIndex) { return isConstructedByNode(edge, nodeIndex); });
        }
    }

    std::vector<GeneEdge> collectEdges(GenomeDesc const& genome, std::vector<GenomeIssue> const& issues)
    {
        std::vector<GeneEdge> result;
        for (auto const& construction : GenomeDescAccessService::get().getGeneConstructions(genome)) {
            GeneEdge edge{.construction = construction};
            std::ranges::copy_if(issues, std::back_inserter(edge.issues), [&](auto const& issue) { return turnsOffConstructorOrSeparation(issue, edge); });
            result.emplace_back(edge);
        }
        return result;
    }

    std::vector<int> calcLayers(GenomeDesc const& genome)
    {
        auto distances = GenomeDescAccessService::get().calcDistancesFromRootGene(genome);
        auto maxDistance = std::ranges::max(distances | std::views::transform([](auto const& distance) { return distance.value_or(0); }));
        auto unreachableLayer = maxDistance + 1;
        auto result = distances | std::views::transform([&](auto const& distance) { return distance.value_or(unreachableLayer); });
        return std::vector(result.begin(), result.end());
    }

    std::map<int, float> calcBarycentersOfConstructingGenes(
        std::vector<int> const& row,
        int rowIndex,
        std::vector<GeneEdge> const& edges,
        std::vector<int> const& layers,
        std::vector<float> const& boxCenters)
    {
        std::map<int, float> result;
        for (auto geneIndex : row) {
            auto sum = 0.0f;
            auto count = 0;
            for (auto const& edge : edges) {
                auto const& construction = edge.construction;
                if (construction.constructedGeneIndex == geneIndex && layers.at(construction.constructingGeneIndex) < rowIndex) {
                    sum += boxCenters.at(construction.constructingGeneIndex);
                    ++count;
                }
            }
            result.emplace(geneIndex, count > 0 ? sum / toFloat(count) : std::numeric_limits<float>::max());
        }
        return result;
    }

    GraphLayout calcLayout(GenomeDesc const& genome, std::vector<GeneEdge> const& edges)
    {
        auto numGenes = toInt(genome._genes.size());
        GraphLayout result;
        result.layers = calcLayers(genome);
        result.boxSize = {scale(BoxWidth), ImGui::GetFontSize() * 2 + scale(BoxPadding) * 2};
        result.boxOffsets.resize(numGenes);

        std::vector<std::vector<int>> rows(std::ranges::max(result.layers) + 1);
        for (auto const& [geneIndex, layer] : result.layers | boost::adaptors::indexed(0)) {
            rows.at(layer).emplace_back(toInt(geneIndex));
        }
        auto calcRowWidth = [&](int numBoxes) { return toFloat(numBoxes) * result.boxSize.x + toFloat(numBoxes - 1) * scale(HorizontalGap); };
        auto maxRowWidth = std::ranges::max(rows | std::views::transform([&](auto const& row) { return calcRowWidth(toInt(row.size())); }));

        std::vector<float> boxCenters(numGenes, 0.0f);
        for (auto const& [rowIndex, row] : rows | boost::adaptors::indexed(0)) {
            auto barycenters = calcBarycentersOfConstructingGenes(row, toInt(rowIndex), edges, result.layers, boxCenters);
            auto sortedRow = row;
            std::ranges::sort(sortedRow, [&](int lhs, int rhs) { return std::pair(barycenters.at(lhs), lhs) < std::pair(barycenters.at(rhs), rhs); });

            auto startX = scale(Margin) + (maxRowWidth - calcRowWidth(toInt(sortedRow.size()))) / 2;
            auto y = scale(Margin) + toFloat(rowIndex) * (result.boxSize.y + scale(VerticalGap));
            for (auto const& [position, geneIndex] : sortedRow | boost::adaptors::indexed(0)) {
                auto x = startX + toFloat(position) * (result.boxSize.x + scale(HorizontalGap));
                result.boxOffsets.at(geneIndex) = {x, y};
                boxCenters.at(geneIndex) = x + result.boxSize.x / 2;
            }
        }

        auto spaceForCurvesAroundRightSide = scale(BendSize) * 3;
        auto spaceForCurvesBelowLastRow = scale(BendSize) * 2;
        result.boxAreaWidth = scale(Margin) * 2 + maxRowWidth;
        result.canvasSize = {
            result.boxAreaWidth + spaceForCurvesAroundRightSide,
            scale(Margin) * 2 + toFloat(rows.size()) * (result.boxSize.y + scale(VerticalGap)) - scale(VerticalGap) + spaceForCurvesBelowLastRow};
        return result;
    }

    ImVec2 calcCenteringOffset(ImVec2 const& availableSize, GraphLayout const& layout)
    {
        auto offsetX = std::clamp((availableSize.x - layout.boxAreaWidth) / 2, 0.0f, std::max(0.0f, availableSize.x - layout.canvasSize.x));
        auto offsetY = std::max(0.0f, (availableSize.y - layout.canvasSize.y) / 2);
        return {offsetX, offsetY};
    }

    BezierCurve calcSelfLoopCurve(ImVec2 const& box, ImVec2 const& boxSize)
    {
        auto bend = scale(BendSize);
        auto right = box.x + boxSize.x;
        return BezierCurve{
            .start = {right, box.y + boxSize.y * 0.3f},
            .control1 = {right + bend * 2, box.y - bend / 2},
            .control2 = {right + bend * 2, box.y + boxSize.y + bend / 2},
            .end = {right, box.y + boxSize.y * 0.7f}};
    }

    BezierCurve calcDownwardCurve(ImVec2 const& sourceBox, ImVec2 const& targetBox, ImVec2 const& boxSize)
    {
        ImVec2 start{sourceBox.x + boxSize.x / 2, sourceBox.y + boxSize.y};
        ImVec2 end{targetBox.x + boxSize.x / 2, targetBox.y};
        auto verticalBend = (end.y - start.y) / 2;
        return BezierCurve{.start = start, .control1 = {start.x, start.y + verticalBend}, .control2 = {end.x, end.y - verticalBend}, .end = end};
    }

    BezierCurve calcCurveBelowRow(ImVec2 const& sourceBox, ImVec2 const& targetBox, ImVec2 const& boxSize)
    {
        auto bend = scale(BendSize);
        ImVec2 start{sourceBox.x + boxSize.x / 2, sourceBox.y + boxSize.y};
        ImVec2 end{targetBox.x + boxSize.x / 2, targetBox.y + boxSize.y};
        return BezierCurve{.start = start, .control1 = {start.x, start.y + bend * 2}, .control2 = {end.x, end.y + bend * 2}, .end = end};
    }

    BezierCurve calcCurveAroundRightSide(ImVec2 const& sourceBox, ImVec2 const& targetBox, ImVec2 const& boxSize)
    {
        ImVec2 start{sourceBox.x + boxSize.x, sourceBox.y + boxSize.y / 2};
        ImVec2 end{targetBox.x + boxSize.x, targetBox.y + boxSize.y / 2};
        auto bendX = std::max(start.x, end.x) + scale(BendSize) * 2;
        return BezierCurve{.start = start, .control1 = {bendX, start.y}, .control2 = {bendX, end.y}, .end = end};
    }

    BezierCurve calcCurve(GeneEdge const& edge, GraphLayout const& layout, ImVec2 const& origin)
    {
        auto const& construction = edge.construction;
        auto const& sourceOffset = layout.boxOffsets.at(construction.constructingGeneIndex);
        auto const& targetOffset = layout.boxOffsets.at(construction.constructedGeneIndex);
        ImVec2 sourceBox{origin.x + sourceOffset.x, origin.y + sourceOffset.y};
        ImVec2 targetBox{origin.x + targetOffset.x, origin.y + targetOffset.y};

        if (construction.constructingGeneIndex == construction.constructedGeneIndex) {
            return calcSelfLoopCurve(sourceBox, layout.boxSize);
        }
        auto sourceLayer = layout.layers.at(construction.constructingGeneIndex);
        auto targetLayer = layout.layers.at(construction.constructedGeneIndex);
        if (targetLayer > sourceLayer) {
            return calcDownwardCurve(sourceBox, targetBox, layout.boxSize);
        }
        if (targetLayer == sourceLayer) {
            return calcCurveBelowRow(sourceBox, targetBox, layout.boxSize);
        }
        return calcCurveAroundRightSide(sourceBox, targetBox, layout.boxSize);
    }

    std::vector<ImVec2> sampleCurve(BezierCurve const& curve)
    {
        std::vector<ImVec2> result;
        for (auto segment : std::views::iota(0, NumCurveSegments + 1)) {
            result.emplace_back(ImBezierCubicCalc(curve.start, curve.control1, curve.control2, curve.end, toFloat(segment) / NumCurveSegments));
        }
        return result;
    }

    float calcDistance(ImVec2 const& point1, ImVec2 const& point2)
    {
        return std::hypot(point1.x - point2.x, point1.y - point2.y);
    }

    float calcDistanceToPolyline(ImVec2 const& point, std::vector<ImVec2> const& polyline)
    {
        auto result = std::numeric_limits<float>::max();
        for (auto const& [start, end] : std::views::zip(polyline, polyline | std::views::drop(1))) {
            result = std::min(result, calcDistance(point, ImLineClosestPoint(start, end, point)));
        }
        return result;
    }

    void drawPolyline(ImDrawList* drawList, std::vector<ImVec2> const& polyline, ImU32 color, float thickness, bool dashed)
    {
        if (!dashed) {
            drawList->AddPolyline(polyline.data(), toInt(polyline.size()), color, ImDrawFlags_None, thickness);
            return;
        }
        auto dashLength = scale(DashLength);
        auto distance = 0.0f;
        for (auto const& [start, end] : std::views::zip(polyline, polyline | std::views::drop(1))) {
            if (std::fmod(distance, dashLength * 2) < dashLength) {
                drawList->AddLine(start, end, color, thickness);
            }
            distance += calcDistance(start, end);
        }
    }

    void drawArrow(ImDrawList* drawList, std::vector<ImVec2> polyline, ImU32 color, float thickness, bool dashed)
    {
        auto tip = polyline.back();
        auto arrowLength = scale(ArrowLength);
        auto reversedPolyline = polyline | std::views::reverse;
        auto arrowStart = std::ranges::find_if(reversedPolyline, [&](ImVec2 const& point) { return calcDistance(point, tip) >= arrowLength; });
        if (arrowStart == reversedPolyline.end()) {
            drawPolyline(drawList, polyline, color, thickness, dashed);
            return;
        }

        auto distance = calcDistance(*arrowStart, tip);
        auto directionX = (tip.x - arrowStart->x) / distance;
        auto directionY = (tip.y - arrowStart->y) / distance;
        ImVec2 base{tip.x - directionX * arrowLength, tip.y - directionY * arrowLength};
        polyline.erase(arrowStart.base(), polyline.end());
        polyline.emplace_back(base);
        drawPolyline(drawList, polyline, color, thickness, dashed);

        auto halfWidth = scale(ArrowWidth) / 2;
        drawList->AddTriangleFilled(
            tip, {base.x - directionY * halfWidth, base.y + directionX * halfWidth}, {base.x + directionY * halfWidth, base.y - directionX * halfWidth}, color);
    }

    void removeLastUtf8Character(std::string& text)
    {
        auto isContinuationByte = [](char byte) { return (static_cast<unsigned char>(byte) & 0xc0) == 0x80; };
        while (!text.empty() && isContinuationByte(text.back())) {
            text.pop_back();
        }
        if (!text.empty()) {
            text.pop_back();
        }
    }

    std::string truncateToWidth(std::string text, float maxWidth)
    {
        if (ImGui::CalcTextSize(text.c_str()).x <= maxWidth) {
            return text;
        }
        while (!text.empty() && ImGui::CalcTextSize((text + "...").c_str()).x > maxWidth) {
            removeLastUtf8Character(text);
        }
        return text + "...";
    }

    std::string getGeneTooltip(GenomeDesc const& genome, std::vector<GenomeIssue> const& issues, int geneIndex)
    {
        auto const& gene = genome._genes.at(geneIndex);
        auto result = "Gene " + std::to_string(geneIndex);
        if (!gene._name.empty()) {
            result += " (" + gene._name + ")";
        }
        if (geneIndex == 0) {
            result += ", root gene";
        }
        auto geneIssues = GenomeIssue::filterByGene(issues, geneIndex);
        if (!geneIssues.empty()) {
            result += "\n\n" + GenomeIssueDescription::getTooltip(geneIssues, genome);
        }
        return result;
    }

    std::string getEdgeTooltip(GenomeDesc const& genome, GeneEdge const& edge)
    {
        auto const& construction = edge.construction;
        auto result = "Gene " + std::to_string(construction.constructingGeneIndex) + " constructs gene " + std::to_string(construction.constructedGeneIndex)
            + (construction.constructorNodeIndices.size() == 1 ? "\nConstructor in node " : "\nConstructors in nodes ")
            + StringHelper::join(construction.constructorNodeIndices) + (construction.anyWithSeparation ? ", with separation" : "");
        if (!edge.issues.empty()) {
            result += "\n\n" + GenomeIssueDescription::getTooltip(edge.issues, genome);
        }
        return result;
    }

    ImRect getBoxRect(int geneIndex, GraphLayout const& layout, ImVec2 const& origin)
    {
        auto const& offset = layout.boxOffsets.at(geneIndex);
        ImVec2 boxMin{origin.x + offset.x, origin.y + offset.y};
        return ImRect(boxMin, {boxMin.x + layout.boxSize.x, boxMin.y + layout.boxSize.y});
    }

    std::optional<int> findGeneAt(ImVec2 const& position, GraphLayout const& layout, ImVec2 const& origin)
    {
        for (auto geneIndex : std::views::iota(0, toInt(layout.boxOffsets.size()))) {
            if (getBoxRect(geneIndex, layout, origin).Contains(position)) {
                return geneIndex;
            }
        }
        return std::nullopt;
    }

    std::optional<int> findEdgeNear(ImVec2 const& position, std::vector<std::vector<ImVec2>> const& polylines)
    {
        std::optional<int> result;
        auto minDistance = scale(EdgeHoverDistance);
        for (auto const& [edgeIndex, polyline] : polylines | boost::adaptors::indexed(0)) {
            auto distance = calcDistanceToPolyline(position, polyline);
            if (distance < minDistance) {
                minDistance = distance;
                result = toInt(edgeIndex);
            }
        }
        return result;
    }

    void drawEdges(
        ImDrawList* drawList,
        GenomeTabEditData const& editData,
        std::vector<GeneEdge> const& edges,
        std::vector<std::vector<ImVec2>> const& polylines,
        std::optional<int> const& hoveredEdgeIndex)
    {
        auto removedGeneIndices = GenomeIssue::getRemovedGeneIndices(editData->genomeIssues);
        auto selectedNodeIndex = editData->isNodeLevelSelected() ? editData->getSelectedNodeIndex() : std::nullopt;

        for (auto const& [edgeIndex, edge] : edges | boost::adaptors::indexed(0)) {
            auto const& construction = edge.construction;
            auto mostRelevantIssue = GenomeIssueDescription::findMostRelevantIssue(edge.issues);
            auto isSelected = editData->selectedGeneIndex == construction.constructingGeneIndex && selectedNodeIndex.has_value()
                && isConstructedByNode(edge, selectedNodeIndex.value());
            auto isRemoved = removedGeneIndices.contains(construction.constructingGeneIndex) || removedGeneIndices.contains(construction.constructedGeneIndex);

            ImColor color = Const::TextDimColor;
            if (mostRelevantIssue.has_value()) {
                color = GenomeIssueDescription::getColor(mostRelevantIssue.value());
            } else if (isSelected) {
                color = Const::AccentColor;
            } else if (isRemoved) {
                color = Const::TextDecentColor;
            }
            auto isHighlighted = isSelected || hoveredEdgeIndex == edgeIndex;
            drawArrow(
                drawList, polylines.at(edgeIndex), color, scale(isHighlighted ? HighlightedEdgeThickness : EdgeThickness), construction.anyWithSeparation);
        }
    }

    void drawSelectionOutline(ImDrawList* drawList, ImRect const& boxRect)
    {
        auto gap = scale(SelectionGap);
        drawList->AddRect(
            {boxRect.Min.x - gap, boxRect.Min.y - gap},
            {boxRect.Max.x + gap, boxRect.Max.y + gap},
            Const::AccentColor,
            scale(BoxRounding) + gap,
            ImDrawFlags_None,
            scale(SelectionThickness));
    }

    void drawGenes(
        ImDrawList* drawList,
        GenomeTabEditData const& editData,
        GraphLayout const& layout,
        ImVec2 const& origin,
        std::optional<int> const& hoveredGeneIndex)
    {
        auto const& genome = editData->genome;
        auto removedGeneIndices = GenomeIssue::getRemovedGeneIndices(editData->genomeIssues);
        auto padding = scale(BoxPadding);

        for (auto const& [geneIndex, gene] : genome._genes | boost::adaptors::indexed(0)) {
            auto mostRelevantIssue = GenomeIssueDescription::findMostRelevantIssue(GenomeIssue::filterByGene(editData->genomeIssues, toInt(geneIndex)));
            auto isRemoved = removedGeneIndices.contains(toInt(geneIndex));
            auto isHovered = hoveredGeneIndex == geneIndex;
            auto boxRect = getBoxRect(toInt(geneIndex), layout, origin);

            ImColor fillColor = Const::RaisedColor;
            if (geneIndex == 0) {
                fillColor = isHovered ? Const::HeaderSelectedHoveredColor : Const::HeaderColor;
            } else if (isHovered) {
                fillColor = Const::TreeNodeHighHoveredColor;
            }
            ImColor borderColor = Const::LineColor;
            ImColor labelColor = Const::TextBaseColor;
            if (isRemoved) {
                borderColor = Const::WarningColor;
                labelColor = Const::WarningColor;
            } else if (mostRelevantIssue.has_value()) {
                borderColor = GenomeIssueDescription::getColor(mostRelevantIssue.value());
                labelColor = borderColor;
            }
            auto borderThickness = scale(isRemoved || mostRelevantIssue.has_value() ? 1.5f : 1.0f);
            drawList->AddRectFilled(boxRect.Min, boxRect.Max, fillColor, scale(BoxRounding));
            drawList->AddRect(boxRect.Min, boxRect.Max, borderColor, scale(BoxRounding), ImDrawFlags_None, borderThickness);
            if (editData->selectedGeneIndex == geneIndex) {
                drawSelectionOutline(drawList, boxRect);
            }

            auto label = "Gene " + std::to_string(geneIndex);
            drawList->AddText({boxRect.Min.x + padding, boxRect.Min.y + padding}, labelColor, label.c_str());
            if (!gene._name.empty()) {
                drawList->AddText(
                    {boxRect.Min.x + padding, boxRect.Min.y + padding + ImGui::GetFontSize()},
                    Const::TextDecentColor,
                    truncateToWidth(gene._name, layout.boxSize.x - padding * 2).c_str());
            }
        }
    }

    void selectConstructingNode(GenomeTabEditData const& editData, GeneEdge const& edge)
    {
        auto mostRelevantIssue = GenomeIssueDescription::findMostRelevantIssue(edge.issues);
        auto hasIssueAtConstructor =
            mostRelevantIssue.has_value() && mostRelevantIssue->nodeIndex.has_value() && isConstructedByNode(edge, mostRelevantIssue->nodeIndex.value());
        auto nodeIndex = hasIssueAtConstructor ? mostRelevantIssue->nodeIndex.value() : edge.construction.constructorNodeIndices.front();
        editData->selectNode(edge.construction.constructingGeneIndex, nodeIndex);
    }

    void processGraph(GenomeTabEditData const& editData)
    {
        auto const& genome = editData->genome;
        auto edges = collectEdges(genome, editData->genomeIssues);
        auto layout = calcLayout(genome, edges);

        auto centeringOffset = calcCenteringOffset(ImGui::GetContentRegionAvail(), layout);
        ImGui::SetCursorPos({ImGui::GetCursorPosX() + centeringOffset.x, ImGui::GetCursorPosY() + centeringOffset.y});
        auto origin = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##canvas", layout.canvasSize);
        auto isCanvasHovered = ImGui::IsItemHovered();
        auto isCanvasClicked = ImGui::IsItemClicked();

        std::vector<std::vector<ImVec2>> polylines;
        for (auto const& edge : edges) {
            polylines.emplace_back(sampleCurve(calcCurve(edge, layout, origin)));
        }

        std::optional<int> hoveredGeneIndex;
        std::optional<int> hoveredEdgeIndex;
        if (isCanvasHovered) {
            auto mousePos = ImGui::GetMousePos();
            hoveredGeneIndex = findGeneAt(mousePos, layout, origin);
            if (!hoveredGeneIndex.has_value()) {
                hoveredEdgeIndex = findEdgeNear(mousePos, polylines);
            }
        }

        auto drawList = ImGui::GetWindowDrawList();
        drawEdges(drawList, editData, edges, polylines, hoveredEdgeIndex);
        drawGenes(drawList, editData, layout, origin, hoveredGeneIndex);

        if (isCanvasClicked) {
            if (hoveredGeneIndex.has_value()) {
                editData->selectGene(hoveredGeneIndex.value());
            } else if (hoveredEdgeIndex.has_value()) {
                selectConstructingNode(editData, edges.at(hoveredEdgeIndex.value()));
            }
        }
        if (hoveredGeneIndex.has_value()) {
            AlienGui::Tooltip(getGeneTooltip(genome, editData->genomeIssues, hoveredGeneIndex.value()));
        } else if (hoveredEdgeIndex.has_value()) {
            AlienGui::Tooltip(getEdgeTooltip(genome, edges.at(hoveredEdgeIndex.value())));
        }
    }
}

void GeneGraphWidget::process(GenomeTabEditData const& editData)
{
    if (ImGui::BeginChild("GeneGraph", ImVec2(0, 0), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
        if (!editData->genome._genes.empty()) {
            processGraph(editData);
        }
    }
    ImGui::EndChild();
}
