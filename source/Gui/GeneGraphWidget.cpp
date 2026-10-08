#include "GeneGraphWidget.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <ranges>
#include <set>

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>
#include <imgui_internal.h>

#include <Fonts/IconsFontAwesome5.h>

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
    auto constexpr ArrowSize = 7.0f;
    auto constexpr DashLength = 4.0f;
    auto constexpr EdgeThickness = 1.3f;
    auto constexpr HighlightedEdgeThickness = 2.4f;
    auto constexpr EdgeHoverDistance = 5.0f;
    auto constexpr NumCurveSegments = 32;

    // All constructors of a gene that construct the same gene
    struct GeneEdge
    {
        int sourceGeneIndex = 0;
        int targetGeneIndex = 0;
        std::vector<int> nodeIndices;
        bool separation = false;
        std::vector<GenomeIssue> issues;
    };

    struct GraphLayout
    {
        std::vector<int> layers;
        std::vector<ImVec2> boxPositions;  // Upper left corners relative to the canvas origin
        ImVec2 boxSize;
        float boxAreaWidth = 0;  // Without the space for the arrows bending around the right side
        ImVec2 canvasSize;
    };

    struct BezierCurve
    {
        ImVec2 start;
        ImVec2 control1;
        ImVec2 control2;
        ImVec2 end;
    };

    bool containsNode(GeneEdge const& edge, int nodeIndex)
    {
        return std::ranges::find(edge.nodeIndices, nodeIndex) != edge.nodeIndices.end();
    }

    // True if the issue turns off a constructor of the edge or its separation
    bool affectsEdge(GenomeIssue const& issue, GeneEdge const& edge)
    {
        if (issue.geneIndex != edge.sourceGeneIndex) {
            return false;
        }
        switch (issue.type) {
        case GenomeIssueType::CycleAvoidingRootGene:
        case GenomeIssueType::TooManyGenesWithSeparation:
        case GenomeIssueType::ConstructsRemovedGene:
            return issue.nodeIndex.has_value() && containsNode(edge, issue.nodeIndex.value());
        default:
            return std::ranges::any_of(issue.voidedNodeIndices, [&](int nodeIndex) { return containsNode(edge, nodeIndex); });
        }
    }

    std::vector<GeneEdge> collectEdges(GenomeDesc const& genome, std::vector<GenomeIssue> const& issues)
    {
        auto numGenes = toInt(genome._genes.size());
        std::map<std::pair<int, int>, GeneEdge> edgeByGenePair;
        for (auto const& [geneIndex, gene] : genome._genes | boost::adaptors::indexed(0)) {
            for (auto const& [nodeIndex, node] : gene._nodes | boost::adaptors::indexed(0)) {
                if (!node._constructor.has_value() || node._constructor->_geneIndex < 0 || node._constructor->_geneIndex >= numGenes) {
                    continue;
                }
                auto sourceGeneIndex = toInt(geneIndex);
                auto targetGeneIndex = node._constructor->_geneIndex;
                auto& edge =
                    edgeByGenePair
                        .try_emplace({sourceGeneIndex, targetGeneIndex}, GeneEdge{.sourceGeneIndex = sourceGeneIndex, .targetGeneIndex = targetGeneIndex})
                        .first->second;
                edge.nodeIndices.emplace_back(toInt(nodeIndex));
                edge.separation = edge.separation || node._constructor->_separation;
            }
        }

        std::vector<GeneEdge> result;
        for (auto& edge : edgeByGenePair | std::views::values) {
            std::ranges::copy_if(issues, std::back_inserter(edge.issues), [&](auto const& issue) { return affectsEdge(issue, edge); });
            result.emplace_back(edge);
        }
        return result;
    }

    // Breadth-first search from the root gene; genes that are not reachable form an additional last layer
    std::vector<int> calcLayers(int numGenes, std::vector<GeneEdge> const& edges)
    {
        std::vector<int> result(numGenes, -1);
        result.front() = 0;
        std::deque<int> genesToScan = {0};
        while (!genesToScan.empty()) {
            auto geneIndex = genesToScan.front();
            genesToScan.pop_front();
            for (auto const& edge : edges) {
                if (edge.sourceGeneIndex == geneIndex && result.at(edge.targetGeneIndex) < 0) {
                    result.at(edge.targetGeneIndex) = result.at(geneIndex) + 1;
                    genesToScan.emplace_back(edge.targetGeneIndex);
                }
            }
        }
        auto unreachableLayer = std::ranges::max(result) + 1;
        for (auto& layer : result) {
            if (layer < 0) {
                layer = unreachableLayer;
            }
        }
        return result;
    }

    // Every layer forms a row; a gene is placed below the average position of the genes constructing it, which keeps most arrows short
    GraphLayout calcLayout(int numGenes, std::vector<GeneEdge> const& edges)
    {
        GraphLayout result;
        result.layers = calcLayers(numGenes, edges);
        result.boxSize = {scale(BoxWidth), ImGui::GetFontSize() * 2 + scale(BoxPadding) * 2};
        result.boxPositions.resize(numGenes);

        std::vector<std::vector<int>> rows(std::ranges::max(result.layers) + 1);
        for (auto const& [geneIndex, layer] : result.layers | boost::adaptors::indexed(0)) {
            rows.at(layer).emplace_back(toInt(geneIndex));
        }
        auto calcRowWidth = [&](int numBoxes) { return toFloat(numBoxes) * result.boxSize.x + toFloat(numBoxes - 1) * scale(HorizontalGap); };
        auto maxRowWidth = std::ranges::max(rows | std::views::transform([&](auto const& row) { return calcRowWidth(toInt(row.size())); }));

        std::vector<float> boxCenters(numGenes, 0.0f);
        for (auto rowIndex : std::views::iota(0, toInt(rows.size()))) {
            auto& row = rows.at(rowIndex);
            std::map<int, float> barycenters;
            for (auto geneIndex : row) {
                auto sum = 0.0f;
                auto count = 0;
                for (auto const& edge : edges) {
                    if (edge.targetGeneIndex == geneIndex && result.layers.at(edge.sourceGeneIndex) < rowIndex) {
                        sum += boxCenters.at(edge.sourceGeneIndex);
                        ++count;
                    }
                }
                barycenters.emplace(geneIndex, count > 0 ? sum / toFloat(count) : std::numeric_limits<float>::max());
            }
            std::ranges::sort(row, [&](int lhs, int rhs) { return std::pair(barycenters.at(lhs), lhs) < std::pair(barycenters.at(rhs), rhs); });

            auto startX = scale(Margin) + (maxRowWidth - calcRowWidth(toInt(row.size()))) / 2;
            auto y = scale(Margin) + toFloat(rowIndex) * (result.boxSize.y + scale(VerticalGap));
            for (auto const& [position, geneIndex] : row | boost::adaptors::indexed(0)) {
                auto x = startX + toFloat(position) * (result.boxSize.x + scale(HorizontalGap));
                result.boxPositions.at(geneIndex) = {x, y};
                boxCenters.at(geneIndex) = x + result.boxSize.x / 2;
            }
        }

        // Extra space for the arrows bending around the right side and below the last row
        result.boxAreaWidth = scale(Margin) * 2 + maxRowWidth;
        result.canvasSize = {
            result.boxAreaWidth + scale(BendSize) * 3,
            scale(Margin) * 2 + toFloat(rows.size()) * (result.boxSize.y + scale(VerticalGap)) - scale(VerticalGap) + scale(BendSize) * 2};
        return result;
    }

    BezierCurve calcCurve(GeneEdge const& edge, GraphLayout const& layout, ImVec2 const& origin)
    {
        auto const& boxSize = layout.boxSize;
        auto const& sourcePosition = layout.boxPositions.at(edge.sourceGeneIndex);
        auto const& targetPosition = layout.boxPositions.at(edge.targetGeneIndex);
        ImVec2 source{origin.x + sourcePosition.x, origin.y + sourcePosition.y};
        ImVec2 target{origin.x + targetPosition.x, origin.y + targetPosition.y};
        auto bend = scale(BendSize);

        if (edge.sourceGeneIndex == edge.targetGeneIndex) {
            auto right = source.x + boxSize.x;
            return BezierCurve{
                .start = {right, source.y + boxSize.y * 0.3f},
                .control1 = {right + bend * 2, source.y - bend / 2},
                .control2 = {right + bend * 2, source.y + boxSize.y + bend / 2},
                .end = {right, source.y + boxSize.y * 0.7f}};
        }

        auto sourceLayer = layout.layers.at(edge.sourceGeneIndex);
        auto targetLayer = layout.layers.at(edge.targetGeneIndex);
        if (targetLayer > sourceLayer) {
            ImVec2 start{source.x + boxSize.x / 2, source.y + boxSize.y};
            ImVec2 end{target.x + boxSize.x / 2, target.y};
            auto verticalBend = (end.y - start.y) / 2;
            return BezierCurve{.start = start, .control1 = {start.x, start.y + verticalBend}, .control2 = {end.x, end.y - verticalBend}, .end = end};
        }
        if (targetLayer == sourceLayer) {
            // Arrows within a row pass below it
            ImVec2 start{source.x + boxSize.x / 2, source.y + boxSize.y};
            ImVec2 end{target.x + boxSize.x / 2, target.y + boxSize.y};
            return BezierCurve{.start = start, .control1 = {start.x, start.y + bend * 2}, .control2 = {end.x, end.y + bend * 2}, .end = end};
        }

        // Arrows leading upwards bend around the right side
        ImVec2 start{source.x + boxSize.x, source.y + boxSize.y / 2};
        ImVec2 end{target.x + boxSize.x, target.y + boxSize.y / 2};
        auto bendX = std::max(start.x, end.x) + bend * 2;
        return BezierCurve{.start = start, .control1 = {bendX, start.y}, .control2 = {bendX, end.y}, .end = end};
    }

    std::vector<ImVec2> sampleCurve(BezierCurve const& curve)
    {
        std::vector<ImVec2> result;
        for (auto segment : std::views::iota(0, NumCurveSegments + 1)) {
            result.emplace_back(ImBezierCubicCalc(curve.start, curve.control1, curve.control2, curve.end, toFloat(segment) / NumCurveSegments));
        }
        return result;
    }

    float calcDistance(ImVec2 const& point, std::vector<ImVec2> const& polyline)
    {
        auto result = std::numeric_limits<float>::max();
        for (auto const& [start, end] : std::views::zip(polyline, polyline | std::views::drop(1))) {
            auto closest = ImLineClosestPoint(start, end, point);
            result = std::min(result, std::hypot(point.x - closest.x, point.y - closest.y));
        }
        return result;
    }

    void drawEdge(ImDrawList* drawList, std::vector<ImVec2> const& polyline, ImU32 color, float thickness, bool dashed)
    {
        if (dashed) {
            auto dashLength = scale(DashLength);
            auto distance = 0.0f;
            for (auto const& [start, end] : std::views::zip(polyline, polyline | std::views::drop(1))) {
                if (std::fmod(distance, dashLength * 2) < dashLength) {
                    drawList->AddLine(start, end, color, thickness);
                }
                distance += std::hypot(end.x - start.x, end.y - start.y);
            }
        } else {
            drawList->AddPolyline(polyline.data(), toInt(polyline.size()), color, ImDrawFlags_None, thickness);
        }

        // Arrow head along the last segment
        auto const& tip = polyline.back();
        auto const& beforeTip = polyline.at(polyline.size() - 2);
        auto length = std::hypot(tip.x - beforeTip.x, tip.y - beforeTip.y);
        if (length <= 0) {
            return;
        }
        auto directionX = (tip.x - beforeTip.x) / length;
        auto directionY = (tip.y - beforeTip.y) / length;
        auto size = scale(ArrowSize);
        ImVec2 base{tip.x - directionX * size, tip.y - directionY * size};
        drawList->AddTriangleFilled(
            tip, {base.x - directionY * size / 2, base.y + directionX * size / 2}, {base.x + directionY * size / 2, base.y - directionX * size / 2}, color);
    }

    std::string truncateToWidth(std::string text, float maxWidth)
    {
        if (ImGui::CalcTextSize(text.c_str()).x <= maxWidth) {
            return text;
        }
        while (!text.empty() && ImGui::CalcTextSize((text + "...").c_str()).x > maxWidth) {
            text.pop_back();
        }
        return text + "...";
    }

    std::vector<GenomeIssue> getGeneIssues(std::vector<GenomeIssue> const& issues, int geneIndex)
    {
        auto result = issues | std::views::filter([&](auto const& issue) { return issue.geneIndex == geneIndex; });
        return std::vector(result.begin(), result.end());
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
        auto geneIssues = getGeneIssues(issues, geneIndex);
        if (!geneIssues.empty()) {
            result += "\n\n" + GenomeIssueDescription::getTooltip(geneIssues, genome);
        }
        return result;
    }

    std::string getEdgeTooltip(GenomeDesc const& genome, GeneEdge const& edge)
    {
        auto nodeIndices = edge.nodeIndices | std::views::transform([](int nodeIndex) { return std::to_string(nodeIndex); });
        auto result = "Gene " + std::to_string(edge.sourceGeneIndex) + " constructs gene " + std::to_string(edge.targetGeneIndex)
            + (edge.nodeIndices.size() == 1 ? "\nConstructor in node " : "\nConstructors in nodes ")
            + boost::algorithm::join(std::vector(nodeIndices.begin(), nodeIndices.end()), ", ") + (edge.separation ? ", with separation" : "");
        if (!edge.issues.empty()) {
            result += "\n\n" + GenomeIssueDescription::getTooltip(edge.issues, genome);
        }
        return result;
    }

    void processGraph(GenomeTabEditData const& editData)
    {
        auto const& genome = editData->genome;
        auto const& issues = editData->genomeIssues;
        auto numGenes = toInt(genome._genes.size());
        auto edges = collectEdges(genome, issues);
        auto layout = calcLayout(numGenes, edges);

        // The boxes are centered as long as the whole canvas still fits into the visible width
        auto availableWidth = ImGui::GetContentRegionAvail().x;
        auto offsetX = std::clamp((availableWidth - layout.boxAreaWidth) / 2, 0.0f, std::max(0.0f, availableWidth - layout.canvasSize.x));
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offsetX);

        auto origin = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("##canvas", layout.canvasSize);
        auto isCanvasHovered = ImGui::IsItemHovered();
        auto isCanvasClicked = ImGui::IsItemClicked();

        auto getBoxRect = [&](int geneIndex) {
            auto const& position = layout.boxPositions.at(geneIndex);
            ImVec2 boxMin{origin.x + position.x, origin.y + position.y};
            return ImRect(boxMin, {boxMin.x + layout.boxSize.x, boxMin.y + layout.boxSize.y});
        };
        std::vector<std::vector<ImVec2>> polylines;
        for (auto const& edge : edges) {
            polylines.emplace_back(sampleCurve(calcCurve(edge, layout, origin)));
        }

        // Boxes take precedence over the arrows passing them
        std::optional<int> hoveredGeneIndex;
        std::optional<int> hoveredEdgeIndex;
        if (isCanvasHovered) {
            auto mousePos = ImGui::GetMousePos();
            for (auto geneIndex : std::views::iota(0, numGenes)) {
                if (getBoxRect(geneIndex).Contains(mousePos)) {
                    hoveredGeneIndex = geneIndex;
                }
            }
            if (!hoveredGeneIndex.has_value()) {
                auto minDistance = scale(EdgeHoverDistance);
                for (auto const& [edgeIndex, polyline] : polylines | boost::adaptors::indexed(0)) {
                    auto distance = calcDistance(mousePos, polyline);
                    if (distance < minDistance) {
                        minDistance = distance;
                        hoveredEdgeIndex = toInt(edgeIndex);
                    }
                }
            }
        }

        std::set<int> removedGeneIndices;
        for (auto const& issue : issues) {
            if (issue.removesGene) {
                removedGeneIndices.insert(issue.geneIndex);
            }
        }
        auto selectedNodeIndex = editData->isNodeLevelSelected() ? editData->getSelectedNodeIndex() : std::nullopt;
        auto drawList = ImGui::GetWindowDrawList();

        for (auto const& [edgeIndex, edge] : edges | boost::adaptors::indexed(0)) {
            auto markerIssue = GenomeIssueDescription::findMarkerIssue(edge.issues);
            auto isSelected =
                editData->selectedGeneIndex == edge.sourceGeneIndex && selectedNodeIndex.has_value() && containsNode(edge, selectedNodeIndex.value());
            auto isRemoved = removedGeneIndices.contains(edge.sourceGeneIndex) || removedGeneIndices.contains(edge.targetGeneIndex);

            ImColor color = Const::TextDimColor;
            if (markerIssue.has_value()) {
                color = GenomeIssueDescription::getColor(markerIssue.value());
            } else if (isSelected) {
                color = Const::AccentColor;
            } else if (isRemoved) {
                color = Const::TextDecentColor;
            }
            auto isHighlighted = isSelected || hoveredEdgeIndex == edgeIndex;
            drawEdge(drawList, polylines.at(edgeIndex), color, scale(isHighlighted ? HighlightedEdgeThickness : EdgeThickness), edge.separation);
        }

        auto padding = scale(BoxPadding);
        for (auto geneIndex : std::views::iota(0, numGenes)) {
            auto markerIssue = GenomeIssueDescription::findMarkerIssue(getGeneIssues(issues, geneIndex));
            auto isRemoved = removedGeneIndices.contains(geneIndex);
            auto isRoot = geneIndex == 0;
            auto boxRect = getBoxRect(geneIndex);

            auto isHovered = hoveredGeneIndex == geneIndex;
            ImColor fillColor = Const::RaisedColor;
            if (isRoot) {
                fillColor = isHovered ? Const::HeaderSelectedHoveredColor : Const::HeaderColor;
            } else if (isHovered) {
                fillColor = Const::TreeNodeHighHoveredColor;
            }
            ImColor borderColor = Const::LineColor;
            ImColor labelColor = Const::TextBaseColor;
            if (isRemoved) {
                borderColor = Const::WarningColor;
                labelColor = Const::WarningColor;
            } else if (markerIssue.has_value()) {
                borderColor = GenomeIssueDescription::getColor(markerIssue.value());
                labelColor = borderColor;
            }
            auto borderThickness = scale(isRemoved || markerIssue.has_value() ? 1.5f : 1.0f);
            drawList->AddRectFilled(boxRect.Min, boxRect.Max, fillColor, scale(BoxRounding));
            drawList->AddRect(boxRect.Min, boxRect.Max, borderColor, scale(BoxRounding), ImDrawFlags_None, borderThickness);

            // The selection is a separate outline so that the border keeps showing the issue color
            if (editData->selectedGeneIndex == geneIndex) {
                auto gap = scale(SelectionGap);
                drawList->AddRect(
                    {boxRect.Min.x - gap, boxRect.Min.y - gap},
                    {boxRect.Max.x + gap, boxRect.Max.y + gap},
                    Const::AccentColor,
                    scale(BoxRounding) + gap,
                    ImDrawFlags_None,
                    scale(SelectionThickness));
            }

            auto label = "Gene " + std::to_string(geneIndex);
            drawList->AddText({boxRect.Min.x + padding, boxRect.Min.y + padding}, labelColor, label.c_str());
            auto const& name = genome._genes.at(geneIndex)._name;
            if (!name.empty()) {
                drawList->AddText(
                    {boxRect.Min.x + padding, boxRect.Min.y + padding + ImGui::GetFontSize()},
                    Const::TextDecentColor,
                    truncateToWidth(name, layout.boxSize.x - padding * 2).c_str());
            }
        }

        if (isCanvasClicked) {
            if (hoveredGeneIndex.has_value()) {
                editData->selectGene(hoveredGeneIndex.value());
            } else if (hoveredEdgeIndex.has_value()) {
                auto const& edge = edges.at(hoveredEdgeIndex.value());
                auto markerIssue = GenomeIssueDescription::findMarkerIssue(edge.issues);
                auto nodeIndex = markerIssue.has_value() && markerIssue->nodeIndex.has_value() && containsNode(edge, markerIssue->nodeIndex.value())
                    ? markerIssue->nodeIndex.value()
                    : edge.nodeIndices.front();
                editData->selectNode(edge.sourceGeneIndex, nodeIndex);
            }
        }
        if (hoveredGeneIndex.has_value()) {
            AlienGui::Tooltip(getGeneTooltip(genome, issues, hoveredGeneIndex.value()));
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
