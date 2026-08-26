#include "NeuralNetEditorWidget.h"

#include <algorithm>
#include <cmath>
#include <ranges>

#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include <Base/Math.h>
#include <Base/StringHelper.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/NumberGenerator.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr GraphRowSpacing = 19.0f;
    auto constexpr GraphRowStretchMax = 1.8f;
    auto constexpr GraphGroupSpacing = 18.0f;
    auto constexpr GraphVerticalMargin = 14.0f;
    auto constexpr GraphHorizontalMargin = 6.0f;
    auto constexpr GroupBlockPadding = 4.0f;
    auto constexpr GroupBlockRounding = 4.0f;
    auto constexpr GroupBlockNodeMargin = 9.0f;
    auto constexpr GroupLabelRailWidth = 17.0f;
    auto constexpr GroupLabelRailMargin = 4.0f;
    auto constexpr GroupBlockAlpha = 0.1f;
    auto constexpr GroupLabelRailAlpha = 0.18f;
    auto constexpr NodeRadius = 5.0f;
    auto constexpr NodeClickPadding = 3.0f;
    auto constexpr NodeLabelMargin = 5.0f;
    auto constexpr ActivationLabelMargin = 6.0f;
    auto constexpr BiasMarkerWidth = 4.0f;
    auto constexpr BiasMarkerHeight = 9.0f;
    auto constexpr BiasMarkerMargin = 5.0f;
    auto constexpr BiasMarkerRounding = 1.0f;
    auto constexpr BiasMarkerMinAlpha = 0.35f;
    auto constexpr BiasMarkerMaxAlpha = 1.0f;
    auto constexpr BiasMarkerZeroAlpha = 0.3f;
    auto constexpr BiasMarkerSelectedBrightening = 0.35f;
    auto constexpr BiasMaxValue = 2.0f;
    auto constexpr BiasLogScale = 30.0f;
    auto constexpr OutputBlockNodeMargin = BiasMarkerMargin + BiasMarkerWidth + GroupBlockPadding;
    auto constexpr CellFunctionAreaMargin = 6.0f;
    auto constexpr CellFunctionLaneSpacing = 3.0f;
    auto constexpr CellFunctionMarkerMargin = 9.0f;
    auto constexpr CellFunctionLabelMargin = 19.0f;
    auto constexpr CellFunctionLabelPadding = 4.0f;
    auto constexpr CellFunctionMarkerWidth = 4.5f;
    auto constexpr CellFunctionMarkerHeight = 4.0f;
    auto constexpr CellFunctionMarkerGap = 0.5f;
    auto constexpr CellFunctionTapOverlap = 3.0f;
    auto constexpr CellFunctionDualLabelScale = 0.85f;
    auto constexpr CellFunctionLabelAlpha = 0.95f;
    auto constexpr CellFunctionTapAlpha = 0.45f;
    auto constexpr CellFunctionOuterTapAlpha = 0.28f;
    auto constexpr CardWidth = 260.0f;
    auto constexpr CardPadding = 9.0f;
    auto constexpr CardOverlayMargin = 10.0f;
    auto constexpr ResetButtonOverlap = 7.0f;
    auto constexpr ConnectionWeightSliderMinWidth = 42.0f;
    auto constexpr ConnectionWeightIndexMargin = 4.0f;
    auto constexpr ConnectionWeightIndexAlpha = 0.6f;
    auto constexpr ToolButtonSeparatorMargin = 5.0f;
    auto constexpr ToolButtonSeparatorPadding = 3.0f;
    auto constexpr GraphMinCurveWidth = 30.0f;
    auto constexpr CardItemSpacing = 5.0f;
    auto constexpr CardTopMargin = 6.0f;
    auto constexpr CardVerticalPosition = 0.75f;
    auto constexpr ButtonAreaHeight = 50.0f;
    auto constexpr ActivationIconHeight = 18.0f;
    auto constexpr ActivationIconSamples = 48;
    auto constexpr ActivationIconDomain = 2.0f;
    auto constexpr ActivationIconMinRange = 1.0f;

    auto const DialogSize = RealVector2D(700.0f, 500.0f);

    auto const PositiveWeightColor = ImColor(77, 163, 255);
    auto const NegativeWeightColor = ImColor(255, 77, 77);
    auto const SignalNodeColor = ImColor(77, 163, 255);
    auto const MemoryNodeColor = ImColor(180, 140, 255);
    auto const TelemetryNodeColor = ImColor(111, 220, 140);
    auto const CellFunctionColor = ImColor(255, 140, 66);
    auto const SelectedCellFunctionColor = ImColor(255, 196, 153);
    auto const SelectedNodeColor = ImColor(255, 210, 77);
    auto const NodeFillColor = ImColor(18, 21, 27);
    auto const ZeroWeightColor = ImColor(90, 97, 110);
    auto const LabelColor = ImColor(170, 178, 194);

    std::vector<std::string> const TelemetryLabels = {"Energy", "Attacked", "Age", "Velocity"};
    std::vector<std::string> const ActivationFunctionShortStrings = {"tanh", "step", "id", "abs", "gauss", "mod"};

    ImColor withAlpha(ImColor const& color, float alpha)
    {
        auto result = color;
        result.Value.w = alpha;
        return result;
    }

    // Shifts a color towards opaque white, used to highlight elements belonging to the selection
    ImColor brighten(ImColor const& color, float amount)
    {
        auto result = color;
        result.Value.x += (1.0f - result.Value.x) * amount;
        result.Value.y += (1.0f - result.Value.y) * amount;
        result.Value.z += (1.0f - result.Value.z) * amount;
        result.Value.w += (1.0f - result.Value.w) * amount;
        return result;
    }

    ImColor groupColor(int inputIndex)
    {
        if (inputIndex < STANDARD_NEURONS_PER_CELL) {
            return SignalNodeColor;
        } else if (inputIndex < NEURAL_NET_OUTPUTS) {
            return MemoryNodeColor;
        }
        return TelemetryNodeColor;
    }

    float inputNodeOffsetY(int inputIndex, float rowSpacing)
    {
        auto result = GraphVerticalMargin + (toFloat(inputIndex) + 0.5f) * rowSpacing;
        if (inputIndex >= STANDARD_NEURONS_PER_CELL) {
            result += GraphGroupSpacing;
        }
        if (inputIndex >= NEURAL_NET_OUTPUTS) {
            result += GraphGroupSpacing;
        }
        return result;
    }

    // The output rows use the same spacing as the input rows and therefore end above the telemetry inputs
    float outputNodeOffsetY(int outputIndex, float rowSpacing)
    {
        auto result = GraphVerticalMargin + (toFloat(outputIndex) + 0.5f) * rowSpacing;
        if (outputIndex >= STANDARD_NEURONS_PER_CELL) {
            result += GraphGroupSpacing;
        }
        return result;
    }

    // Used for the markers accompanying the sliders, which stay small regardless of the label font
    ImFont* smallLabelFont()
    {
        return StyleRepository::get().getTinyFont();
    }

    // The labels are drawn with a font that is rasterized at the used size, otherwise thin strokes are lost when the glyphs are scaled
    float nodeLabelFontSize(ImFont* font, float fontScale = 1.0f)
    {
        return font->FontSize * fontScale;
    }

    ImVec2 calcNodeLabelSize(ImFont* font, std::string const& text, float fontScale = 1.0f)
    {
        return font->CalcTextSizeA(nodeLabelFontSize(font, fontScale), FLT_MAX, 0.0f, text.c_str());
    }

    void addNodeLabel(ImDrawList* drawList, ImFont* font, ImVec2 const& pos, ImColor const& color, std::string const& text, float fontScale = 1.0f)
    {
        drawList->AddText(font, nodeLabelFontSize(font, fontScale), pos, color, text.c_str());
    }

    // ImGui cannot render rotated text, therefore the text is written horizontally and its vertices are rotated afterwards.
    // Glyphs outside the clip rectangle would be dropped before the rotation, therefore the text is written at the center
    // of the visible area and moved to its final position together with the rotation.
    void addRotatedNodeLabel(ImDrawList* drawList, ImFont* font, ImVec2 const& center, ImColor const& color, std::string const& text, bool clockwise)
    {
        auto clipRectMin = drawList->GetClipRectMin();
        auto clipRectMax = drawList->GetClipRectMax();
        ImVec2 pivot((clipRectMin.x + clipRectMax.x) / 2, (clipRectMin.y + clipRectMax.y) / 2);

        auto textSize = calcNodeLabelSize(font, text);
        auto firstVertex = drawList->VtxBuffer.Size;
        addNodeLabel(drawList, font, {pivot.x - textSize.x / 2, pivot.y - textSize.y / 2}, color, text);
        auto lastVertex = drawList->VtxBuffer.Size;

        auto angle = clockwise ? Const::Pi / 2 : -Const::Pi / 2;
        auto cosAngle = std::cos(angle);
        auto sinAngle = std::sin(angle);
        for (auto* vertex = drawList->VtxBuffer.Data + firstVertex; vertex != drawList->VtxBuffer.Data + lastVertex; ++vertex) {
            ImVec2 offset(vertex->pos.x - pivot.x, vertex->pos.y - pivot.y);
            vertex->pos = {center.x + offset.x * cosAngle - offset.y * sinAngle, center.y + offset.x * sinAngle + offset.y * cosAngle};
        }
    }

    float evalActivationFunction(ActivationFunction activationFunction, float x)
    {
        switch (activationFunction) {
        case ActivationFunction_Tanh:
            return std::tanh(x);
        case ActivationFunction_BinaryStep:
            return x >= 0.0f ? 1.0f : 0.0f;
        case ActivationFunction_Identity:
            return x;
        case ActivationFunction_Abs:
            return std::abs(x);
        case ActivationFunction_Gaussian:
            return std::exp(-2 * x * x);
        case ActivationFunction_Mod:
            return std::fmod(std::fmod(x + 1.0f, 2.0f) + 2.0f, 2.0f) - 1.0f;
        }
        return 0.0f;
    }

    // Biases are usually far below the maximum, therefore the magnitude grows logarithmically and resolves the range around zero.
    // Returns the visual magnitude relative to its maximum.
    float calcBiasFraction(float bias)
    {
        auto normalized = std::min(1.0f, std::abs(bias) / BiasMaxValue);
        return std::log1p(BiasLogScale * normalized) / std::log1p(BiasLogScale);
    }

    // The marker uses the weight colors, so the bias is read in the same visual language as the weights
    ImColor calcBiasMarkerColor(float bias, bool isSelected)
    {
        auto result = [&] {
            if (std::abs(bias) <= NEAR_ZERO) {
                return withAlpha(ZeroWeightColor, BiasMarkerZeroAlpha);
            }
            auto alpha = BiasMarkerMinAlpha + calcBiasFraction(bias) * (BiasMarkerMaxAlpha - BiasMarkerMinAlpha);
            return withAlpha(bias > 0 ? PositiveWeightColor : NegativeWeightColor, alpha);
        }();
        return isSelected ? brighten(result, BiasMarkerSelectedBrightening) : result;
    }

    // Small block in front of an output node, placed inside the surrounding group block
    void addBiasMarker(ImDrawList* drawList, ImVec2 const& nodePos, float bias, bool isSelected)
    {
        auto maxX = nodePos.x - scale(NodeRadius + BiasMarkerMargin);
        auto halfHeight = scale(BiasMarkerHeight) / 2;
        ImVec2 min{maxX - scale(BiasMarkerWidth), nodePos.y - halfHeight};
        ImVec2 max{maxX, nodePos.y + halfHeight};

        // Opaque backdrop so that the weight curves running below do not shine through
        drawList->AddRectFilled(min, max, NodeFillColor, scale(BiasMarkerRounding));
        drawList->AddRectFilled(min, max, calcBiasMarkerColor(bias, isSelected), scale(BiasMarkerRounding));
    }

    // Draws the graph of an activation function into the given rectangle
    void drawActivationFunctionIcon(ImDrawList* drawList, ImVec2 const& min, ImVec2 const& max, ActivationFunction activationFunction, ImColor const& color)
    {
        auto centerY = (min.y + max.y) / 2;
        auto halfHeight = (max.y - min.y) / 2;
        drawList->AddLine({min.x, centerY}, {max.x, centerY}, withAlpha(LabelColor, 0.25f), scale(1.0f));

        auto sampleAt = [&](int index) {
            auto t = toFloat(index) / ActivationIconSamples;
            return evalActivationFunction(activationFunction, -ActivationIconDomain + 2 * ActivationIconDomain * t);
        };

        // Scale the graph such that it fills the rectangle without being cut off
        auto range = ActivationIconMinRange;
        for (int i = 0; i <= ActivationIconSamples; ++i) {
            range = std::max(range, std::abs(sampleAt(i)));
        }

        std::vector<ImVec2> points;
        for (int i = 0; i <= ActivationIconSamples; ++i) {
            auto t = toFloat(i) / ActivationIconSamples;
            ImVec2 point{min.x + t * (max.x - min.x), centerY - sampleAt(i) / range * halfHeight};

            // Do not connect the branches of a discontinuous function
            if (!points.empty() && std::abs(point.y - points.back().y) > halfHeight) {
                drawList->AddPolyline(points.data(), toInt(points.size()), color, 0, scale(1.3f));
                points.clear();
            }
            points.emplace_back(point);
        }
        drawList->AddPolyline(points.data(), toInt(points.size()), color, 0, scale(1.3f));
    }

    float resetButtonWidth()
    {
        auto font = smallLabelFont();
        return font->CalcTextSizeA(font->FontSize, FLT_MAX, 0.0f, ICON_FA_TIMES).x + 2 * ImGui::GetStyle().FramePadding.x;
    }

    // Small button next to a slider that resets its value
    bool resetButton()
    {
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(ResetButtonOverlap));
        ImGui::PushFont(smallLabelFont());
        auto result = ImGui::Button(ICON_FA_TIMES);
        ImGui::PopFont();
        return result;
    }

    // The connection weight sliders are wrapped into several rows if they would become too small. Only divisors of the
    // number of connections are used as row length so that all rows are equally filled.
    int calcConnectionWeightSlidersPerRow(float availableWidth, float indexWidth)
    {
        auto minSliderAreaWidth = scale(ConnectionWeightSliderMinWidth) + indexWidth + resetButtonWidth() + ImGui::GetStyle().ItemSpacing.x;
        auto maxPerRow = std::min(MAX_OBJECT_CONNECTIONS, std::max(1, toInt(availableWidth / minSliderAreaWidth)));
        for (auto perRow = maxPerRow; perRow > 1; --perRow) {
            if (MAX_OBJECT_CONNECTIONS % perRow == 0) {
                return perRow;
            }
        }
        return 1;
    }

    // Marker in front of a cell function label: a triangle pointing away from the outgoing node if the cell function
    // reads the channel and back towards it if the cell function overwrites the channel
    void addChannelMarker(ImDrawList* drawList, ImVec2 const& center, bool isRead, bool isWrite, ImColor const& color)
    {
        auto width = scale(CellFunctionMarkerWidth);
        auto height = scale(CellFunctionMarkerHeight);
        auto gap = scale(CellFunctionMarkerGap);

        auto addReadMarker = [&](float minX, float maxX) {
            drawList->AddTriangleFilled({minX, center.y - height}, {maxX, center.y}, {minX, center.y + height}, color);
        };
        auto addWriteMarker = [&](float minX, float maxX) {
            drawList->AddTriangleFilled({maxX, center.y - height}, {maxX, center.y + height}, {minX, center.y}, color);
        };

        if (isRead && isWrite) {
            addWriteMarker(center.x - width - gap, center.x - gap);
            addReadMarker(center.x + gap, center.x + width + gap);
        } else if (isRead) {
            addReadMarker(center.x - width, center.x + width);
        } else {
            addWriteMarker(center.x - width, center.x + width);
        }
    }

    // A channel that is read and overwritten shows both roles below each other and therefore uses a smaller font
    float calcChannelLabelWidth(ImFont* font, CellFunctionChannel const& channel)
    {
        if (!channel.readLabel.empty() && !channel.writeLabel.empty()) {
            return std::max(
                calcNodeLabelSize(font, channel.writeLabel, CellFunctionDualLabelScale).x,
                calcNodeLabelSize(font, channel.readLabel, CellFunctionDualLabelScale).x);
        }
        return calcNodeLabelSize(font, channel.readLabel.empty() ? channel.writeLabel : channel.readLabel).x;
    }

    // Divides the buttons of a row into groups
    void addToolButtonSeparator()
    {
        auto drawList = ImGui::GetWindowDrawList();
        auto pos = ImGui::GetCursorScreenPos();
        auto centerX = pos.x + scale(ToolButtonSeparatorMargin);
        auto padding = scale(ToolButtonSeparatorPadding);
        drawList->AddLine({centerX, pos.y + padding}, {centerX, pos.y + ImGui::GetFrameHeight() - padding}, withAlpha(LabelColor, 0.25f), scale(1.0f));
        ImGui::Dummy({2 * scale(ToolButtonSeparatorMargin), ImGui::GetFrameHeight()});
    }

    float buttonWidth(std::string const& text)
    {
        return ImGui::CalcTextSize(text.c_str()).x + 2 * ImGui::GetStyle().FramePadding.x;
    }

    // Invisible button around a node center; returns true if clicked
    bool nodeClickArea(ImVec2 const& pos, char const* id, bool& hovered)
    {
        auto clickRadius = scale(NodeRadius + NodeClickPadding);
        ImGui::SetCursorScreenPos({pos.x - clickRadius, pos.y - clickRadius});
        auto result = ImGui::InvisibleButton(id, {clickRadius * 2, clickRadius * 2});
        hovered = ImGui::IsItemHovered();
        return result;
    }
}

NeuralNetEditorWidget _NeuralNetEditorWidget::create()
{
    return NeuralNetEditorWidget(new _NeuralNetEditorWidget());
}

void _NeuralNetEditorWidget::process(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    std::vector<float>& connectionWeights,
    std::vector<CellFunctionModule> const& cellFunctionModules,
    std::optional<LiveData> const& liveData)
{
    auto& selectionData = getValueRef(_dataById);

    processEditor(weights, biases, activationFunctions, connectionWeights, cellFunctionModules, liveData, selectionData, EditorMode::Embedded);
    processDialog(weights, biases, activationFunctions, connectionWeights, cellFunctionModules, liveData, selectionData);
}

void _NeuralNetEditorWidget::openDialog()
{
    _openDialogRequested = true;
}

// The dialog is modal and edits a copy of the net, but shares the selection with the embedded editor
void _NeuralNetEditorWidget::processDialog(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    std::vector<float>& connectionWeights,
    std::vector<CellFunctionModule> const& cellFunctionModules,
    std::optional<LiveData> const& liveData,
    SelectionData& selectionData)
{
    if (_openDialogRequested) {
        _openDialogRequested = false;
        _dialogNet = NetData{weights, biases, activationFunctions};
        _dialogConnectionWeights = connectionWeights;
        _adopted = false;
        _dialog.open();
    }
    if (!_dialog.isOpen()) {
        return;
    }

    _dialog.process([&] {
        processEditor(
            _dialogNet.weights,
            _dialogNet.biases,
            _dialogNet.activationFunctions,
            _dialogConnectionWeights,
            cellFunctionModules,
            liveData,
            selectionData,
            EditorMode::Dialog);
    });

    if (_adopted) {
        _adopted = false;
        weights = _dialogNet.weights;
        biases = _dialogNet.biases;
        activationFunctions = _dialogNet.activationFunctions;
        connectionWeights = _dialogConnectionWeights;
    }
}

void _NeuralNetEditorWidget::processEditor(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    std::vector<float>& connectionWeights,
    std::vector<CellFunctionModule> const& cellFunctionModules,
    std::optional<LiveData> const& liveData,
    SelectionData& selectionData,
    EditorMode mode)
{
    // The dialog offers more room than the embedded editor and therefore labels the graph with the larger default font
    _labelFont = mode == EditorMode::Dialog ? StyleRepository::get().getDefaultFont() : StyleRepository::get().getTinyFont();

    if (ImGui::BeginChild("NeuralNetEditor", ImVec2(0, 0), 0, 0)) {
        auto narrowLayout = isNarrowLayout(ImGui::GetContentRegionAvail().x, cellFunctionModules);

        // Use a child window for the content, reserving space for the action buttons
        auto buttonAreaHeight = scale(ButtonAreaHeight);
        GraphGeometry graphGeometry;
        if (ImGui::BeginChild("NeuralNetEditorContent", ImVec2(0, -buttonAreaHeight), false)) {
            processConnectionWeightSliders(connectionWeights);
            ImGui::Separator();

            // The graph takes the height that is left over after the fixed parts have been placed
            auto reservedHeight = 0.0f;
            if (narrowLayout) {
                reservedHeight = 2 * ImGui::GetStyle().ItemSpacing.y + scale(CardTopMargin) + calcInspectorCardHeight();
            }
            auto rowSpacing = calcGraphRowSpacing(ImGui::GetContentRegionAvail().y - reservedHeight);

            graphGeometry = processGraph(weights, biases, activationFunctions, cellFunctionModules, selectionData, liveData, narrowLayout, rowSpacing);

            // There is no room for the card on top of the graph, therefore it is placed below it and does not cover any nodes
            if (narrowLayout) {
                processInspectorCard(weights, biases, activationFunctions, selectionData, graphGeometry, narrowLayout);
            }
        }
        ImGui::EndChild();

        AlienGui::Separator();

        // The floating card may reach below the graph and is therefore bounded by the position of the action buttons
        _actionButtonsMinY = ImGui::GetCursorScreenPos().y;
        processActionButtons(weights, biases, activationFunctions, mode);

        // Processed last so that the card is above the other widgets
        if (!narrowLayout) {
            processInspectorCard(weights, biases, activationFunctions, selectionData, graphGeometry, narrowLayout);
        }
    }
    ImGui::EndChild();
}

// The label shares the line with the sliders so that the graph below keeps as much height as possible
void _NeuralNetEditorWidget::processConnectionWeightSliders(std::vector<float>& connectionWeights)
{
    auto& style = ImGui::GetStyle();
    auto frameHeight = ImGui::GetFrameHeight();

    // Aligned with the group blocks of the graph below
    auto horizontalMargin = scale(GraphHorizontalMargin);
    ImGui::Indent(horizontalMargin);

    ImGui::AlignTextToFramePadding();
    AlienGui::Text("Connection weights");
    ImGui::SameLine();

    auto contentMin = ImGui::GetCursorScreenPos();
    auto contentWidth = ImGui::GetContentRegionAvail().x - horizontalMargin;

    auto indexWidth = calcNodeLabelSize(smallLabelFont(), std::to_string(MAX_OBJECT_CONNECTIONS)).x + scale(ConnectionWeightIndexMargin);
    auto slidersPerRow = calcConnectionWeightSlidersPerRow(contentWidth, indexWidth);
    auto numRows = MAX_OBJECT_CONNECTIONS / slidersPerRow;
    auto contentHeight = toFloat(numRows) * frameHeight + toFloat(numRows - 1) * style.ItemSpacing.y;

    auto drawList = ImGui::GetWindowDrawList();
    auto cellWidth = (contentWidth - style.ItemSpacing.x * toFloat(slidersPerRow - 1)) / toFloat(slidersPerRow);
    // AlienGui scales the slider width itself, therefore it is passed unscaled
    auto sliderWidth = scaleInverse(cellWidth - indexWidth - resetButtonWidth() - style.ItemSpacing.x + scale(ResetButtonOverlap));

    ImGui::PushID("ConnectionWeightSliders");
    for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
        ImVec2 cellPos{
            contentMin.x + toFloat(i % slidersPerRow) * (cellWidth + style.ItemSpacing.x),
            contentMin.y + toFloat(i / slidersPerRow) * (frameHeight + style.ItemSpacing.y)};

        auto indexLabel = std::to_string(i + 1);
        auto indexSize = calcNodeLabelSize(smallLabelFont(), indexLabel);
        addNodeLabel(
            drawList,
            smallLabelFont(),
            {cellPos.x + indexWidth - scale(ConnectionWeightIndexMargin) - indexSize.x, cellPos.y + (frameHeight - indexSize.y) / 2},
            withAlpha(LabelColor, ConnectionWeightIndexAlpha),
            indexLabel);

        ImGui::PushID(i);
        ImGui::SetCursorScreenPos({cellPos.x + indexWidth, cellPos.y});
        auto originalGrabMinSize = style.GrabMinSize;
        style.GrabMinSize = scale(8.0f);
        AlienGui::SliderFloat(AlienGui::SliderFloatParameters().format("%.2f").width(sliderWidth).textWidth(0).min(-1.0f).max(1.0f), &connectionWeights.at(i));
        style.GrabMinSize = originalGrabMinSize;
        if (resetButton()) {
            connectionWeights.at(i) = 0.0f;
        }
        ImGui::PopID();
    }
    ImGui::PopID();

    // The sliders are placed manually, therefore the whole block is registered afterwards
    ImGui::SetCursorScreenPos(contentMin);
    ImGui::Dummy({contentWidth, contentHeight});

    ImGui::Unindent(horizontalMargin);
}

_NeuralNetEditorWidget::GraphGeometry _NeuralNetEditorWidget::processGraph(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    std::vector<CellFunctionModule> const& cellFunctionModules,
    SelectionData& selectionData,
    std::optional<LiveData> const& liveData,
    bool narrowLayout,
    float rowSpacing)
{
    GraphGeometry result;
    auto graphHeight = scale(inputNodeOffsetY(NEURAL_NET_INPUTS - 1, rowSpacing) + rowSpacing / 2 + GraphVerticalMargin);

    // The labels of the nodes and cell functions have a fixed width, therefore the graph is scrolled horizontally if it cannot be shrunk any further
    auto minWidth = calcGraphMinWidth(cellFunctionModules);
    auto scrollHorizontally = narrowLayout && ImGui::GetContentRegionAvail().x < minWidth;
    if (scrollHorizontally) {
        graphHeight += ImGui::GetStyle().ScrollbarSize;
    }
    if (ImGui::BeginChild("NeuralNetGraph", ImVec2(0, graphHeight), 0, scrollHorizontally ? ImGuiWindowFlags_HorizontalScrollbar : 0)) {
        auto drawList = ImGui::GetWindowDrawList();
        auto origin = ImGui::GetCursorScreenPos();
        auto width = std::max(ImGui::GetContentRegionAvail().x, minWidth);

        LayoutData layout;
        auto horizontalMargin = scale(GraphHorizontalMargin);
        auto cellFunctionAreaWidth = calcCellFunctionAreaWidth(cellFunctionModules);
        auto inputNodeX = origin.x + horizontalMargin + calcInputNodeMargin();
        auto outputNodeX = origin.x + width - horizontalMargin - cellFunctionAreaWidth - calcOutputNodeMargin();
        for (int i = 0; i < NEURAL_NET_INPUTS; ++i) {
            layout.inputNodePos[i] = {inputNodeX, origin.y + scale(inputNodeOffsetY(i, rowSpacing))};
        }
        for (int i = 0; i < NEURAL_NET_OUTPUTS; ++i) {
            layout.outputNodePos[i] = {outputNodeX, origin.y + scale(outputNodeOffsetY(i, rowSpacing))};
        }
        layout.leftBlockMinX = origin.x + horizontalMargin;
        layout.leftBlockMaxX = inputNodeX + scale(NodeRadius + GroupBlockNodeMargin);
        layout.rightBlockMinX = outputNodeX - scale(NodeRadius + OutputBlockNodeMargin);
        layout.rightBlockMaxX = origin.x + width - horizontalMargin - cellFunctionAreaWidth;

        // The cell function blocks are lined up outside of the outgoing block, the first one closest to it
        auto laneMinX = layout.rightBlockMaxX + scale(CellFunctionAreaMargin);
        for (auto const& functionModule : cellFunctionModules) {
            auto laneWidth = calcCellFunctionLaneWidth(functionModule);
            layout.cellFunctionLanes.emplace_back(laneMinX, laneMinX + laneWidth);
            laneMinX += laneWidth + scale(CellFunctionLaneSpacing);
        }

        drawGroupBlocks(drawList, layout, rowSpacing);
        drawCellFunctionBlocks(drawList, layout, cellFunctionModules, selectionData, rowSpacing);
        drawWeightCurves(weights, selectionData, drawList, layout);
        drawInputNodes(selectionData, drawList, layout, liveData);
        drawOutputNodes(biases, activationFunctions, selectionData, drawList, layout);

        // Only the drawn content extends to the right border, therefore the scroll range is defined explicitly
        if (scrollHorizontally) {
            ImGui::SetCursorScreenPos(origin);
            ImGui::Dummy({width, 0.0f});
        }

        result.origin = origin;
        result.width = width;
        result.groupBlockGapMinX = layout.leftBlockMaxX;
        result.groupBlockGapWidth = layout.rightBlockMinX - layout.leftBlockMaxX;
    }
    ImGui::EndChild();

    return result;
}

void _NeuralNetEditorWidget::drawGroupBlock(
    ImDrawList* drawList,
    std::string const& name,
    ImColor const& color,
    ImVec2 const& min,
    ImVec2 const& max,
    bool leftSide)
{
    // Only the corners facing the graph are rounded, the outer corners are flush with the editor border
    auto rounding = scale(GroupBlockRounding);
    auto cornerFlags = leftSide ? ImDrawFlags_RoundCornersRight : ImDrawFlags_RoundCornersLeft;
    drawList->AddRectFilled(min, max, withAlpha(color, GroupBlockAlpha), rounding, cornerFlags);

    auto railMinX = leftSide ? min.x : max.x - scale(GroupLabelRailWidth);
    auto railMaxX = leftSide ? min.x + scale(GroupLabelRailWidth) : max.x;
    drawList->AddRectFilled({railMinX, min.y}, {railMaxX, max.y}, withAlpha(color, GroupLabelRailAlpha), rounding, cornerFlags);

    addRotatedNodeLabel(drawList, _labelFont, {(railMinX + railMaxX) / 2, (min.y + max.y) / 2}, withAlpha(color, 0.9f), name, !leftSide);
}

void _NeuralNetEditorWidget::drawGroupBlocks(ImDrawList* drawList, LayoutData const& layout, float rowSpacing)
{
    auto drawBlock = [&](std::string const& name, ImColor const& color, ImVec2 const& firstNodePos, ImVec2 const& lastNodePos, bool leftSide) {
        auto minX = leftSide ? layout.leftBlockMinX : layout.rightBlockMinX;
        auto maxX = leftSide ? layout.leftBlockMaxX : layout.rightBlockMaxX;
        auto padding = scale(rowSpacing / 2 + GroupBlockPadding);
        drawGroupBlock(drawList, name, color, {minX, firstNodePos.y - padding}, {maxX, lastNodePos.y + padding}, leftSide);
    };

    drawBlock("INCOMING", SignalNodeColor, layout.inputNodePos[0], layout.inputNodePos[STANDARD_NEURONS_PER_CELL - 1], true);
    drawBlock("MEMORY", MemoryNodeColor, layout.inputNodePos[STANDARD_NEURONS_PER_CELL], layout.inputNodePos[NEURAL_NET_OUTPUTS - 1], true);
    drawBlock("TELEMETRY", TelemetryNodeColor, layout.inputNodePos[NEURAL_NET_OUTPUTS], layout.inputNodePos[NEURAL_NET_INPUTS - 1], true);

    drawBlock("OUTGOING", SignalNodeColor, layout.outputNodePos[0], layout.outputNodePos[STANDARD_NEURONS_PER_CELL - 1], false);
    drawBlock("MEMORY", MemoryNodeColor, layout.outputNodePos[STANDARD_NEURONS_PER_CELL], layout.outputNodePos[NEURAL_NET_OUTPUTS - 1], false);
}

// The cell functions read and overwrite the outgoing signal after the neural net has been evaluated. Each of them is
// drawn as an own block outside of the outgoing block, listing the channels it accesses together with their meaning.
void _NeuralNetEditorWidget::drawCellFunctionBlocks(
    ImDrawList* drawList,
    LayoutData const& layout,
    std::vector<CellFunctionModule> const& cellFunctionModules,
    SelectionData const& selectionData,
    float rowSpacing)
{
    auto isInnermostLane = true;
    for (auto const& [functionModule, lane] : std::views::zip(cellFunctionModules, layout.cellFunctionLanes)) {
        if (functionModule.channels.empty()) {
            continue;
        }
        auto padding = scale(rowSpacing / 2 + GroupBlockPadding);
        auto minY = layout.outputNodePos[functionModule.channels.front().channel].y - padding;
        auto maxY = layout.outputNodePos[functionModule.channels.back().channel].y + padding;

        // The name is drawn vertically, therefore a block covering only few channels is extended around its center
        auto name = StringHelper::toUpper(functionModule.name);
        auto minHeight = calcNodeLabelSize(_labelFont, name).x + 2 * scale(GroupBlockPadding);
        if (maxY - minY < minHeight) {
            auto centerY = (minY + maxY) / 2;
            minY = centerY - minHeight / 2;
            maxY = centerY + minHeight / 2;
        }
        drawGroupBlock(drawList, name, CellFunctionColor, {lane.minX, minY}, {lane.maxX, maxY}, false);

        // The connection of an outer block crosses the blocks in front of it and is therefore drawn more faintly
        auto tapAlpha = isInnermostLane ? CellFunctionTapAlpha : CellFunctionOuterTapAlpha;
        for (auto const& channel : functionModule.channels) {
            auto nodeY = layout.outputNodePos[channel.channel].y;
            drawList->AddLine(
                {layout.rightBlockMaxX, nodeY}, {lane.minX + scale(CellFunctionTapOverlap), nodeY}, withAlpha(CellFunctionColor, tapAlpha), scale(1.0f));

            auto isRead = !channel.readLabel.empty();
            auto isWrite = !channel.writeLabel.empty();
            auto isSelected = channel.channel == selectionData.outputIndex;
            auto color = withAlpha(isSelected ? SelectedCellFunctionColor : CellFunctionColor, CellFunctionLabelAlpha);
            addChannelMarker(drawList, {lane.minX + scale(CellFunctionMarkerMargin), nodeY}, isRead, isWrite, color);

            auto labelX = lane.minX + scale(CellFunctionLabelMargin);
            if (isRead && isWrite) {
                auto lineHeight = nodeLabelFontSize(_labelFont, CellFunctionDualLabelScale);
                addNodeLabel(drawList, _labelFont, {labelX, nodeY - lineHeight}, color, channel.writeLabel, CellFunctionDualLabelScale);
                addNodeLabel(drawList, _labelFont, {labelX, nodeY}, color, channel.readLabel, CellFunctionDualLabelScale);
            } else {
                auto const& label = isWrite ? channel.writeLabel : channel.readLabel;
                addNodeLabel(drawList, _labelFont, {labelX, nodeY - calcNodeLabelSize(_labelFont, label).y / 2}, color, label);
            }
        }
        isInnermostLane = false;
    }
}

void _NeuralNetEditorWidget::drawWeightCurves(
    std::vector<NeuralNetWeight>& weights,
    SelectionData const& selectionData,
    ImDrawList* drawList,
    LayoutData const& layout)
{
    struct WeightCurve
    {
        int inputIndex;
        int outputIndex;
        float weight;
        bool isSelected;
    };
    std::vector<WeightCurve> curves;
    curves.reserve(NEURAL_NET_OUTPUTS * NEURAL_NET_INPUTS);
    for (int row = 0; row < NEURAL_NET_OUTPUTS; ++row) {
        for (int col = 0; col < NEURAL_NET_INPUTS; ++col) {
            auto weight = weights.at(row * NEURAL_NET_INPUTS + col).getValue();
            auto isSelected = row == selectionData.outputIndex && col == selectionData.inputIndex;
            if (std::abs(weight) <= NEAR_ZERO && !isSelected) {
                continue;
            }
            curves.push_back({col, row, weight, isSelected});
        }
    }
    std::sort(curves.begin(), curves.end(), [](auto const& a, auto const& b) {
        if (a.isSelected != b.isSelected) {
            return b.isSelected;
        }
        return std::abs(a.weight) < std::abs(b.weight);
    });

    for (auto const& curve : curves) {
        auto start = layout.inputNodePos[curve.inputIndex];
        auto end = layout.outputNodePos[curve.outputIndex];
        auto controlOffset = (end.x - start.x) * 0.4f;
        auto thickness = scale(std::max(0.7f, std::min(2.0f, std::abs(curve.weight)) * (curve.isSelected ? 1.7f : 1.0f)));
        auto color = std::abs(curve.weight) > NEAR_ZERO ? calcWeightColor(curve.weight, curve.isSelected ? 0.95f : 0.12f) : withAlpha(ZeroWeightColor, 0.95f);
        drawList->AddBezierCubic(
            {start.x + scale(NodeRadius), start.y},
            {start.x + controlOffset, start.y},
            {end.x - controlOffset, end.y},
            {end.x - scale(NodeRadius), end.y},
            color,
            thickness);
    }
}

void _NeuralNetEditorWidget::drawInputNodes(
    SelectionData& selectionData,
    ImDrawList* drawList,
    LayoutData const& layout,
    std::optional<LiveData> const& liveData)
{
    ImGui::PushID("InputNodes");
    for (int i = 0; i < NEURAL_NET_INPUTS; ++i) {
        auto const& pos = layout.inputNodePos[i];

        ImGui::PushID(i);
        bool hovered = false;
        if (nodeClickArea(pos, "##inputNode", hovered)) {
            selectionData.inputIndex = i;
        }
        ImGui::PopID();

        auto isSelected = i == selectionData.inputIndex;
        if (isSelected) {
            drawList->AddCircle(pos, scale(NodeRadius + 3.5f), SelectedNodeColor, 0, scale(1.2f));
        }
        drawList->AddCircleFilled(pos, scale(NodeRadius), NodeFillColor);
        drawList->AddCircle(pos, scale(NodeRadius), hovered ? SelectedNodeColor : groupColor(i), 0, scale(1.5f));
        if (i >= STANDARD_NEURONS_PER_CELL && i < NEURAL_NET_OUTPUTS) {
            drawList->AddCircleFilled(pos, scale(2.0f), MemoryNodeColor);
        }

        auto label = getInputLabel(i);
        auto textSize = calcNodeLabelSize(_labelFont, label);
        addNodeLabel(
            drawList,
            _labelFont,
            {pos.x - scale(NodeRadius + NodeLabelMargin) - textSize.x, pos.y - textSize.y / 2},
            isSelected ? ImColor(255, 255, 255) : LabelColor,
            label);

        // Live values next to memory and telemetry inputs
        if (liveData.has_value() && i >= STANDARD_NEURONS_PER_CELL) {
            std::string value;
            if (i < NEURAL_NET_OUTPUTS) {
                auto memoryIndex = i - STANDARD_NEURONS_PER_CELL;
                if (memoryIndex < toInt(liveData->memory.size())) {
                    value = StringHelper::format(liveData->memory.at(memoryIndex), 2);
                }
            } else {
                switch (i - NEURAL_NET_OUTPUTS) {
                case TelemetryInputs::Energy:
                    value = StringHelper::format(liveData->energy, 1);
                    break;
                case TelemetryInputs::Attacked:
                    value = liveData->attacked ? "yes" : "no";
                    break;
                case TelemetryInputs::Age:
                    value = std::to_string(liveData->age);
                    break;
                case TelemetryInputs::Speed:
                    value = StringHelper::format(liveData->speed, 2);
                    break;
                }
            }
            if (!value.empty()) {
                addNodeLabel(
                    drawList,
                    _labelFont,
                    {pos.x + scale(NodeRadius + GroupBlockNodeMargin + NodeLabelMargin), pos.y - textSize.y / 2},
                    withAlpha(LabelColor, 0.6f),
                    value);
            }
        }
    }
    ImGui::PopID();
}

void _NeuralNetEditorWidget::drawOutputNodes(
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    SelectionData& selectionData,
    ImDrawList* drawList,
    LayoutData const& layout)
{
    ImGui::PushID("OutputNodes");
    for (int i = 0; i < NEURAL_NET_OUTPUTS; ++i) {
        auto const& pos = layout.outputNodePos[i];

        ImGui::PushID(i);
        bool hovered = false;
        if (nodeClickArea(pos, "##outputNode", hovered)) {
            selectionData.outputIndex = i;
        }
        ImGui::PopID();

        auto isSelected = i == selectionData.outputIndex;
        if (isSelected) {
            drawList->AddCircle(pos, scale(NodeRadius + 3.5f), SelectedNodeColor, 0, scale(1.2f));
        }
        addBiasMarker(drawList, pos, biases.at(i), isSelected);
        drawList->AddCircleFilled(pos, scale(NodeRadius), NodeFillColor);
        auto borderColor = i < STANDARD_NEURONS_PER_CELL ? SignalNodeColor : MemoryNodeColor;
        drawList->AddCircle(pos, scale(NodeRadius), hovered ? SelectedNodeColor : borderColor, 0, scale(1.5f));
        if (i >= STANDARD_NEURONS_PER_CELL) {
            drawList->AddCircleFilled(pos, scale(2.0f), MemoryNodeColor);
        }

        auto label = getOutputLabel(i);
        auto textSize = calcNodeLabelSize(_labelFont, label);
        auto labelX = pos.x + scale(NodeRadius + NodeLabelMargin);
        addNodeLabel(drawList, _labelFont, {labelX, pos.y - textSize.y / 2}, isSelected ? ImColor(255, 255, 255) : LabelColor, label);

        auto const& actfnLabel = ActivationFunctionShortStrings.at(activationFunctions.at(i));
        addNodeLabel(
            drawList, _labelFont, {labelX + textSize.x + scale(ActivationLabelMargin), pos.y - textSize.y / 2}, withAlpha(LabelColor, 0.55f), actfnLabel);
    }
    ImGui::PopID();
}

void _NeuralNetEditorWidget::processInspectorCard(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    SelectionData& selectionData,
    GraphGeometry const& graphGeometry,
    bool narrowLayout)
{
    selectionData.inputIndex = std::clamp(selectionData.inputIndex, 0, NEURAL_NET_INPUTS - 1);
    selectionData.outputIndex = std::clamp(selectionData.outputIndex, 0, NEURAL_NET_OUTPUTS - 1);

    auto cardHeight = calcInspectorCardHeight();
    auto cardMargin = scale(CardTopMargin);

    auto cursorBackup = ImGui::GetCursorScreenPos();
    float cardWidth;
    if (narrowLayout) {
        // The card is placed below the graph and uses the whole width there
        cardWidth = ImGui::GetContentRegionAvail().x;
        ImGui::Dummy({0, cardMargin});
    } else {
        cardWidth = scale(CardWidth);

        // The card sits in the lower part of the graph area, is kept inside the visible area when the editor is scrolled
        // and is moved up as far as needed to leave the action buttons below the graph uncovered
        auto clipMin = ImGui::GetWindowDrawList()->GetClipRectMin();
        auto clipMax = ImGui::GetWindowDrawList()->GetClipRectMax();
        auto highestCardY = std::max(graphGeometry.origin.y, clipMin.y) + cardMargin;
        auto lowestCardY = std::min(_actionButtonsMinY, clipMax.y) - cardHeight - cardMargin;
        auto cardY = std::max(highestCardY + (lowestCardY - highestCardY) * CardVerticalPosition, highestCardY);

        ImGui::SetCursorScreenPos({graphGeometry.groupBlockGapMinX + (graphGeometry.groupBlockGapWidth - cardWidth) / 2, cardY});
    }

    // An own child window is needed so that the card is rendered above the graph and stays independent of its scroll position
    ImGui::BeginChild("InspectorCard", {cardWidth, cardHeight}, 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    processInspectorCardContent(weights, biases, activationFunctions, selectionData, cardWidth, cardHeight);
    ImGui::EndChild();

    if (!narrowLayout) {
        ImGui::SetCursorScreenPos(cursorBackup);
    }
}

void _NeuralNetEditorWidget::processInspectorCardContent(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    SelectionData const& selectionData,
    float cardWidth,
    float cardHeight)
{
    auto inputIndex = selectionData.inputIndex;
    auto outputIndex = selectionData.outputIndex;

    auto lineHeight = ImGui::GetTextLineHeight();
    auto frameHeight = ImGui::GetFrameHeight();
    auto itemSpacing = scale(CardItemSpacing);
    auto cardContentWidth = cardWidth - 2 * scale(CardPadding);

    auto drawList = ImGui::GetWindowDrawList();
    ImVec2 cardMin = ImGui::GetWindowPos();
    ImVec2 cardMax{cardMin.x + cardWidth, cardMin.y + cardHeight};

    drawList->AddRectFilled(cardMin, cardMax, Const::FloatingCardBackgroundColor, scale(4.0f));
    drawList->AddRect(cardMin, cardMax, Const::FloatingCardBorderColor, scale(4.0f), 0, scale(1.0f));

    auto contentX = cardMin.x + scale(CardPadding);
    auto posY = cardMin.y + scale(CardPadding);

    // Header showing the selected connection
    auto headerText = getInputLabel(inputIndex) + "  " ICON_FA_LONG_ARROW_ALT_RIGHT "  " + getOutputLabel(outputIndex);
    auto dotOffset = scale(NodeRadius + 4.0f);
    drawList->AddCircleFilled({contentX + scale(3.0f), posY + lineHeight / 2}, scale(3.5f), groupColor(inputIndex));
    drawList->AddText({contentX + dotOffset, posY}, ImColor(255, 255, 255), headerText.c_str());
    auto headerSize = ImGui::CalcTextSize(headerText.c_str());
    drawList->AddCircleFilled(
        {contentX + dotOffset + headerSize.x + scale(6.0f), posY + lineHeight / 2},
        scale(3.5f),
        outputIndex < STANDARD_NEURONS_PER_CELL ? SignalNodeColor : MemoryNodeColor);
    posY += lineHeight + itemSpacing;

    ImGui::PushID("InspectorCard");

    // The label is drawn manually so that the reset button fits between slider and label
    auto labelWidth = ImGui::CalcTextSize("Weight").x;
    auto spacing = ImGui::GetStyle().ItemSpacing.x;
    auto sliderWidth = scaleInverse(cardContentWidth - 2 * spacing + scale(ResetButtonOverlap) - resetButtonWidth() - labelWidth);
    auto sliderParameters = AlienGui::SliderFloatParameters().format("%.2f").width(sliderWidth).textWidth(0).min(-2.0f).max(2.0f);

    auto& weightValue = weights.at(outputIndex * NEURAL_NET_INPUTS + inputIndex);
    auto weight = weightValue.getValue();
    ImGui::PushID("Weight");
    ImGui::SetCursorScreenPos({contentX, posY});
    if (AlienGui::SliderFloat(sliderParameters, &weight)) {
        weightValue = weight;
    }
    if (resetButton()) {
        weightValue = NeuralNetWeight(0);
    }
    ImGui::SameLine();
    AlienGui::Text("Weight");
    ImGui::PopID();
    posY += frameHeight + itemSpacing;

    ImGui::PushID("Bias");
    ImGui::SetCursorScreenPos({contentX, posY});
    AlienGui::SliderFloat(sliderParameters, &biases.at(outputIndex));
    if (resetButton()) {
        biases.at(outputIndex) = 0.0f;
    }
    ImGui::SameLine();
    AlienGui::Text("Bias");
    ImGui::PopID();
    posY += frameHeight + itemSpacing;

    // Activation function chooser: one button per function showing its graph
    auto buttonSpacing = ImGui::GetStyle().ItemSpacing.x;
    auto numActivationFunctions = toFloat(ActivationFunction_Count);
    auto buttonWidth = (cardContentWidth - buttonSpacing * (numActivationFunctions - 1)) / numActivationFunctions;
    auto buttonHeight = scale(ActivationIconHeight);
    for (int i = 0; i < ActivationFunction_Count; ++i) {
        auto isSelected = activationFunctions.at(outputIndex) == i;
        ImGui::PushID(i);
        ImGui::SetCursorScreenPos({contentX + toFloat(i) * (buttonWidth + buttonSpacing), posY});
        if (isSelected) {
            ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(withAlpha(SelectedNodeColor, 0.18f)));
        }
        if (ImGui::Button("##activationFunction", {buttonWidth, buttonHeight})) {
            activationFunctions.at(outputIndex) = static_cast<ActivationFunction>(i);
        }
        if (isSelected) {
            ImGui::PopStyleColor();
        }
        AlienGui::Tooltip(Const::ActivationFunctionStrings.at(i));

        auto iconMin = ImGui::GetItemRectMin();
        auto iconMax = ImGui::GetItemRectMax();
        auto iconPadding = scale(3.0f);
        if (isSelected) {
            drawList->AddRect(iconMin, iconMax, SelectedNodeColor, ImGui::GetStyle().FrameRounding, 0, scale(1.0f));
        }
        drawActivationFunctionIcon(
            drawList,
            {iconMin.x + iconPadding, iconMin.y + iconPadding},
            {iconMax.x - iconPadding, iconMax.y - iconPadding},
            static_cast<ActivationFunction>(i),
            isSelected ? SelectedNodeColor : LabelColor);
        ImGui::PopID();
    }

    ImGui::PopID();
}

_NeuralNetEditorWidget::_NeuralNetEditorWidget()
    : _dialog("Neural network", DialogSize)
{}

// The dialog is closed from here so that its buttons and the net tools share one row
void _NeuralNetEditorWidget::processActionButtons(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    EditorMode mode)
{
    if (mode == EditorMode::Dialog) {
        if (AlienGui::Button("Adopt")) {
            _adopted = true;
            _dialog.close();
        }
        ImGui::SetItemDefaultFocus();
        ImGui::SameLine();
        if (AlienGui::Button("Cancel")) {
            _dialog.close();
        }

        // The net tools belong to the content and are therefore set off from the dialog buttons
        ImGui::SameLine();
        auto offset = ImGui::GetContentRegionAvail().x - calcNetToolButtonsWidth();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, offset));
    }
    processNetToolButtons(weights, biases, activationFunctions);
}

void _NeuralNetEditorWidget::processNetToolButtons(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions)
{
    if (AlienGui::Button("Clear")) {
        for (int i = 0; i < NEURAL_NET_OUTPUTS; ++i) {
            for (int j = 0; j < NEURAL_NET_INPUTS; ++j) {
                weights[i * NEURAL_NET_INPUTS + j] = NeuralNetWeight(0);
            }
            biases[i] = 0;
            activationFunctions[i] = ActivationFunction_Identity;
        }
    }
    ImGui::SameLine();
    if (AlienGui::Button("Identity")) {
        for (int i = 0; i < NEURAL_NET_OUTPUTS; ++i) {
            for (int j = 0; j < NEURAL_NET_INPUTS; ++j) {
                weights[i * NEURAL_NET_INPUTS + j] = (i == j) ? NeuralNetWeight(1.0f) : NeuralNetWeight(0);
            }
            biases[i] = 0.0f;
            activationFunctions[i] = ActivationFunction_Identity;
        }
    }
    ImGui::SameLine();
    if (AlienGui::Button("Randomize")) {
        for (int i = 0; i < NEURAL_NET_OUTPUTS; ++i) {
            for (int j = 0; j < NEURAL_NET_INPUTS; ++j) {
                weights[i * NEURAL_NET_INPUTS + j] = NeuralNetWeight(NumberGenerator::get().getRandomFloat(-2.0f, 2.0f));
            }
            biases[i] = NumberGenerator::get().getRandomFloat(-0.2f, 0.2f);
            activationFunctions[i] = NumberGenerator::get().getRandomInt(ActivationFunction_Count);
        }
    }

    ImGui::SameLine();
    addToolButtonSeparator();

    ImGui::SameLine();
    if (AlienGui::Button("Copy")) {
        NetData copiedNet{weights, biases, activationFunctions};
        _copiedNet = copiedNet;
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!_copiedNet.has_value());
    if (AlienGui::Button("Paste")) {
        weights = _copiedNet->weights;
        biases = _copiedNet->biases;
        activationFunctions = _copiedNet->activationFunctions;
    }
    ImGui::EndDisabled();
}

ImColor _NeuralNetEditorWidget::calcWeightColor(float value, float alpha)
{
    auto result = value > 0 ? PositiveWeightColor : NegativeWeightColor;
    result.Value.w = alpha;
    return result;
}

// The inspector card is only drawn on top of the graph if it fits into the gap between the incoming and the outgoing block.
// Otherwise the editor switches to a narrow layout in which the card is placed below the graph.
bool _NeuralNetEditorWidget::isNarrowLayout(float availableWidth, std::vector<CellFunctionModule> const& cellFunctionModules)
{
    auto requiredWidth = calcGraphMinWidth(cellFunctionModules) - scale(GraphMinCurveWidth) + scale(CardWidth + 2 * CardOverlayMargin);
    return availableWidth < requiredWidth;
}

float _NeuralNetEditorWidget::calcGraphMinWidth(std::vector<CellFunctionModule> const& cellFunctionModules)
{
    return 2 * scale(GraphHorizontalMargin) + calcInputNodeMargin() + calcOutputNodeMargin() + calcCellFunctionAreaWidth(cellFunctionModules)
        + scale(GraphMinCurveWidth);
}

float _NeuralNetEditorWidget::calcGraphRowSpacing(float availableHeight)
{
    auto fixedHeight = 2 * GraphVerticalMargin + 2 * GraphGroupSpacing;
    auto rowSpacing = (scaleInverse(availableHeight) - fixedHeight) / toFloat(NEURAL_NET_INPUTS);
    return std::clamp(rowSpacing, GraphRowSpacing, GraphRowSpacing * GraphRowStretchMax);
}

float _NeuralNetEditorWidget::calcNetToolButtonsWidth()
{
    auto& style = ImGui::GetStyle();
    auto result = buttonWidth("Clear") + buttonWidth("Identity") + buttonWidth("Randomize") + buttonWidth("Copy") + buttonWidth("Paste");
    result += 2 * scale(ToolButtonSeparatorMargin) + 5 * style.ItemSpacing.x;
    return result;
}

float _NeuralNetEditorWidget::calcInspectorCardHeight()
{
    auto itemSpacing = scale(CardItemSpacing);
    return 2 * scale(CardPadding) + ImGui::GetTextLineHeight() + 3 * itemSpacing + 2 * ImGui::GetFrameHeight() + scale(ActivationIconHeight);
}

// The group blocks are only as wide as their content requires, therefore the node columns follow the widest label
float _NeuralNetEditorWidget::calcInputNodeMargin()
{
    auto maxLabelWidth = 0.0f;
    for (auto index : std::views::iota(0, NEURAL_NET_INPUTS)) {
        maxLabelWidth = std::max(maxLabelWidth, calcNodeLabelSize(_labelFont, getInputLabel(index)).x);
    }
    return scale(GroupLabelRailWidth + GroupLabelRailMargin + NodeLabelMargin + NodeRadius) + maxLabelWidth;
}

float _NeuralNetEditorWidget::calcOutputNodeMargin()
{
    auto maxLabelWidth = 0.0f;
    for (auto index : std::views::iota(0, NEURAL_NET_OUTPUTS)) {
        maxLabelWidth = std::max(maxLabelWidth, calcNodeLabelSize(_labelFont, getOutputLabel(index)).x);
    }
    auto maxActfnLabelWidth = 0.0f;
    for (auto const& actfnLabel : ActivationFunctionShortStrings) {
        maxActfnLabelWidth = std::max(maxActfnLabelWidth, calcNodeLabelSize(_labelFont, actfnLabel).x);
    }
    return scale(GroupLabelRailWidth + GroupLabelRailMargin + ActivationLabelMargin + NodeLabelMargin + NodeRadius) + maxLabelWidth + maxActfnLabelWidth;
}

float _NeuralNetEditorWidget::calcCellFunctionLaneWidth(CellFunctionModule const& cellFunctionModule)
{
    auto maxLabelWidth = 0.0f;
    for (auto const& channel : cellFunctionModule.channels) {
        maxLabelWidth = std::max(maxLabelWidth, calcChannelLabelWidth(_labelFont, channel));
    }
    return scale(CellFunctionLabelMargin + CellFunctionLabelPadding + GroupLabelRailWidth) + maxLabelWidth;
}

float _NeuralNetEditorWidget::calcCellFunctionAreaWidth(std::vector<CellFunctionModule> const& cellFunctionModules)
{
    if (cellFunctionModules.empty()) {
        return 0.0f;
    }
    auto result = scale(CellFunctionAreaMargin) + scale(CellFunctionLaneSpacing) * toFloat(cellFunctionModules.size() - 1);
    for (auto const& functionModule : cellFunctionModules) {
        result += calcCellFunctionLaneWidth(functionModule);
    }
    return result;
}

std::string _NeuralNetEditorWidget::getInputLabel(int inputIndex)
{
    if (inputIndex < STANDARD_NEURONS_PER_CELL) {
        return "In " + std::to_string(inputIndex);
    } else if (inputIndex < NEURAL_NET_OUTPUTS) {
        return "Mem " + std::to_string(inputIndex - STANDARD_NEURONS_PER_CELL);
    }
    return TelemetryLabels.at(inputIndex - NEURAL_NET_OUTPUTS);
}

std::string _NeuralNetEditorWidget::getOutputLabel(int outputIndex)
{
    if (outputIndex < STANDARD_NEURONS_PER_CELL) {
        return "Out " + std::to_string(outputIndex);
    }
    return "Mem " + std::to_string(outputIndex - STANDARD_NEURONS_PER_CELL);
}

template <typename T>
_NeuralNetEditorWidget::SelectionData& _NeuralNetEditorWidget::getValueRef(std::unordered_map<unsigned, T>& idToValueMap)
{
    auto id = ImGui::GetID("");
    if (!idToValueMap.contains(id)) {
        idToValueMap[id] = T();
    }
    return idToValueMap.at(id);
}
