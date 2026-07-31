#include "NeuralNetEditorWidget.h"

#include <algorithm>
#include <cmath>

#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/NumberGenerator.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr GraphRowSpacing = 19.0f;
    auto constexpr GraphGroupSpacing = 14.0f;
    auto constexpr GraphHeaderHeight = 29.0f;
    auto constexpr GraphSideMargin = 85.0f;
    auto constexpr NodeRadius = 5.0f;
    auto constexpr NodeClickPadding = 3.0f;
    auto constexpr NodeLabelFontScale = 0.7f;
    auto constexpr CardWidth = 260.0f;
    auto constexpr CardPadding = 9.0f;
    auto constexpr CardMinWidth = 170.0f;
    auto constexpr ResetButtonFontScale = 0.5f;
    auto constexpr CardItemSpacing = 5.0f;
    auto constexpr CardTopMargin = 6.0f;
    auto constexpr ActivationIconHeight = 18.0f;
    auto constexpr ActivationIconSamples = 48;
    auto constexpr ActivationIconDomain = 2.0f;
    auto constexpr ActivationIconRange = 1.3f;

    auto const PositiveWeightColor = ImColor(77, 163, 255);
    auto const NegativeWeightColor = ImColor(255, 77, 77);
    auto const SignalNodeColor = ImColor(77, 163, 255);
    auto const MemoryNodeColor = ImColor(180, 140, 255);
    auto const TelemetryNodeColor = ImColor(111, 220, 140);
    auto const SelectedNodeColor = ImColor(255, 210, 77);
    auto const NodeFillColor = ImColor(18, 21, 27);
    auto const ZeroWeightColor = ImColor(90, 97, 110);
    auto const LabelColor = ImColor(170, 178, 194);
    auto const CardBackgroundColor = ImColor(23, 28, 38, 242);
    auto const CardBorderColor = ImColor(58, 65, 82);

    std::vector<std::string> const TelemetryLabels = {"Energy", "Attacked", "Age", "Speed"};
    std::vector<std::string> const ActivationFunctionShortStrings = {"tanh", "step", "id", "abs", "gauss", "mod"};

    ImColor withAlpha(ImColor const& color, float alpha)
    {
        auto result = color;
        result.Value.w = alpha;
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

    float inputNodeOffsetY(int inputIndex)
    {
        auto result = GraphHeaderHeight + toFloat(inputIndex) * GraphRowSpacing;
        if (inputIndex >= STANDARD_NEURONS_PER_CELL) {
            result += GraphHeaderHeight + GraphGroupSpacing;
        }
        if (inputIndex >= NEURAL_NET_OUTPUTS) {
            result += GraphHeaderHeight + GraphGroupSpacing;
        }
        return result;
    }

    float outputNodeOffsetY(int outputIndex)
    {
        auto graphHeight = inputNodeOffsetY(NEURAL_NET_INPUTS - 1) + GraphRowSpacing;
        auto stretch = (graphHeight - 2 * GraphHeaderHeight - GraphGroupSpacing) / toFloat(NEURAL_NET_OUTPUTS);
        auto result = GraphHeaderHeight + toFloat(outputIndex) * stretch;
        if (outputIndex >= STANDARD_NEURONS_PER_CELL) {
            result += GraphHeaderHeight + GraphGroupSpacing;
        }
        return result;
    }

    float nodeLabelFontSize()
    {
        return ImGui::GetFontSize() * NodeLabelFontScale;
    }

    ImVec2 calcNodeLabelSize(std::string const& text)
    {
        return ImGui::GetFont()->CalcTextSizeA(nodeLabelFontSize(), FLT_MAX, 0.0f, text.c_str());
    }

    void addNodeLabel(ImDrawList* drawList, ImVec2 const& pos, ImColor const& color, std::string const& text)
    {
        drawList->AddText(ImGui::GetFont(), nodeLabelFontSize(), pos, color, text.c_str());
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

    // Draws the graph of an activation function into the given rectangle
    void drawActivationFunctionIcon(ImDrawList* drawList, ImVec2 const& min, ImVec2 const& max, ActivationFunction activationFunction, ImColor const& color)
    {
        auto centerY = (min.y + max.y) / 2;
        auto halfHeight = (max.y - min.y) / 2;
        drawList->AddLine({min.x, centerY}, {max.x, centerY}, withAlpha(LabelColor, 0.25f), scale(1.0f));

        std::vector<ImVec2> points;
        for (int i = 0; i <= ActivationIconSamples; ++i) {
            auto t = toFloat(i) / ActivationIconSamples;
            auto x = -ActivationIconDomain + 2 * ActivationIconDomain * t;
            auto y = std::clamp(evalActivationFunction(activationFunction, x), -ActivationIconRange, ActivationIconRange);
            ImVec2 point{min.x + t * (max.x - min.x), centerY - y / ActivationIconRange * halfHeight};

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
        return ImGui::GetFont()->CalcTextSizeA(ImGui::GetFontSize() * ResetButtonFontScale, FLT_MAX, 0.0f, ICON_FA_TIMES).x
            + 2 * ImGui::GetStyle().FramePadding.x;
    }

    // Small button next to a slider that resets its value
    bool resetButton()
    {
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(7.0f));
        ImGui::SetWindowFontScale(ResetButtonFontScale);
        auto result = ImGui::Button(ICON_FA_TIMES);
        ImGui::SetWindowFontScale(1.0f);
        return result;
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
    std::optional<LiveData> const& liveData)
{
    auto& selectionData = getValueRef(_dataById);

    if (ImGui::BeginChild("NeuralNetEditor", ImVec2(0, 0), 0, 0)) {
        processConnectionWeightSliders(connectionWeights);
        processGraph(weights, biases, activationFunctions, selectionData, liveData);

        AlienGui::Separator();

        processActionButtons(weights, biases, activationFunctions);
    }
    ImGui::EndChild();
}

void _NeuralNetEditorWidget::processConnectionWeightSliders(std::vector<float>& connectionWeights)
{
    AlienGui::Text("Connection weights");

    auto width = ImGui::GetContentRegionAvail().x;
    auto sliderAreaWidth = width / MAX_OBJECT_CONNECTIONS - 2 * ImGui::GetStyle().FramePadding.x;
    auto resetButtonWidth = ImGui::CalcTextSize("x").x;
    auto sliderWidth = sliderAreaWidth - resetButtonWidth - ImGui::GetStyle().ItemSpacing.x;

    ImGui::PushID("ConnectionWeightSliders");
    for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
        if (i > 0) {
            ImGui::SameLine();
        }
        ImGui::PushID(i);
        ImGuiStyle& style = ImGui::GetStyle();
        auto originalGrabMinSize = style.GrabMinSize;
        style.GrabMinSize = scale(8.0f);
        AlienGui::SliderFloat(AlienGui::SliderFloatParameters().format("%.2f").width(sliderWidth).textWidth(0).min(-1.0f).max(1.0f), &connectionWeights.at(i));
        style.GrabMinSize = originalGrabMinSize;
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(7.0f));
        ImGui::SetWindowFontScale(0.5f);
        if (ImGui::Button(ICON_FA_TIMES)) {
            connectionWeights.at(i) = 0.0f;
        }
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopID();
    }
    ImGui::PopID();

    ImGui::Dummy(ImVec2(0, scale(5.0f)));
}

void _NeuralNetEditorWidget::processGraph(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    SelectionData& selectionData,
    std::optional<LiveData> const& liveData)
{
    auto graphHeight = scale(inputNodeOffsetY(NEURAL_NET_INPUTS - 1) + GraphRowSpacing);
    if (ImGui::BeginChild("NeuralNetGraph", ImVec2(0, graphHeight), 0, 0)) {
        auto drawList = ImGui::GetWindowDrawList();
        auto origin = ImGui::GetCursorScreenPos();
        auto width = ImGui::GetContentRegionAvail().x;

        LayoutData layout;
        for (int i = 0; i < NEURAL_NET_INPUTS; ++i) {
            layout.inputNodePos[i] = {origin.x + scale(GraphSideMargin), origin.y + scale(inputNodeOffsetY(i))};
        }
        for (int i = 0; i < NEURAL_NET_OUTPUTS; ++i) {
            layout.outputNodePos[i] = {origin.x + width - scale(GraphSideMargin), origin.y + scale(outputNodeOffsetY(i))};
        }

        drawWeightCurves(weights, selectionData, drawList, layout);
        drawInputNodes(selectionData, drawList, layout, liveData);
        drawOutputNodes(activationFunctions, selectionData, drawList, layout);

        // Drawn last so that the card and its widgets are above the graph
        processInspectorCard(weights, biases, activationFunctions, selectionData, drawList, origin, width);
    }
    ImGui::EndChild();
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
    auto drawGroupHeader = [&](std::string const& text, ImColor const& color, float y) {
        drawList->AddText({layout.inputNodePos[0].x - scale(GraphSideMargin - 5.0f), y}, withAlpha(color, 0.8f), text.c_str());
    };
    drawGroupHeader("SIGNALS", SignalNodeColor, layout.inputNodePos[0].y - scale(GraphHeaderHeight));
    drawGroupHeader("MEMORY", MemoryNodeColor, layout.inputNodePos[STANDARD_NEURONS_PER_CELL].y - scale(GraphHeaderHeight));
    drawGroupHeader("TELEMETRY", TelemetryNodeColor, layout.inputNodePos[NEURAL_NET_OUTPUTS].y - scale(GraphHeaderHeight));

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
        auto textSize = calcNodeLabelSize(label);
        addNodeLabel(
            drawList, {pos.x - scale(NodeRadius + 5.0f) - textSize.x, pos.y - textSize.y / 2}, isSelected ? ImColor(255, 255, 255) : LabelColor, label);

        // Live values next to memory and telemetry inputs
        if (liveData.has_value() && i >= STANDARD_NEURONS_PER_CELL) {
            std::string value;
            if (i < NEURAL_NET_OUTPUTS) {
                auto memoryIndex = i - STANDARD_NEURONS_PER_CELL;
                if (memoryIndex < toInt(liveData->memoryActivities.size())) {
                    value = StringHelper::format(liveData->memoryActivities.at(memoryIndex), 2);
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
                addNodeLabel(drawList, {pos.x + scale(NodeRadius + 5.0f), pos.y - textSize.y / 2}, withAlpha(LabelColor, 0.6f), value);
            }
        }
    }
    ImGui::PopID();
}

void _NeuralNetEditorWidget::drawOutputNodes(
    std::vector<ActivationFunction>& activationFunctions,
    SelectionData& selectionData,
    ImDrawList* drawList,
    LayoutData const& layout)
{
    auto drawGroupHeader = [&](std::string const& text, ImColor const& color, float y) {
        drawList->AddText({layout.outputNodePos[0].x + scale(NodeRadius + 5.0f), y}, withAlpha(color, 0.8f), text.c_str());
    };
    drawGroupHeader("OUTPUTS", SignalNodeColor, layout.outputNodePos[0].y - scale(GraphHeaderHeight));
    drawGroupHeader("MEMORY", MemoryNodeColor, layout.outputNodePos[STANDARD_NEURONS_PER_CELL].y - scale(GraphHeaderHeight));

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
        drawList->AddCircleFilled(pos, scale(NodeRadius), NodeFillColor);
        auto borderColor = i < STANDARD_NEURONS_PER_CELL ? SignalNodeColor : MemoryNodeColor;
        drawList->AddCircle(pos, scale(NodeRadius), hovered ? SelectedNodeColor : borderColor, 0, scale(1.5f));
        if (i >= STANDARD_NEURONS_PER_CELL) {
            drawList->AddCircleFilled(pos, scale(2.0f), MemoryNodeColor);
        }

        auto label = getOutputLabel(i);
        auto textSize = calcNodeLabelSize(label);
        addNodeLabel(drawList, {pos.x + scale(NodeRadius + 5.0f), pos.y - textSize.y / 2}, isSelected ? ImColor(255, 255, 255) : LabelColor, label);

        auto const& actfnLabel = ActivationFunctionShortStrings.at(activationFunctions.at(i));
        addNodeLabel(drawList, {pos.x + scale(NodeRadius + 5.0f) + textSize.x + scale(6.0f), pos.y - textSize.y / 2}, withAlpha(LabelColor, 0.55f), actfnLabel);
    }
    ImGui::PopID();
}

void _NeuralNetEditorWidget::processInspectorCard(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    SelectionData& selectionData,
    ImDrawList* drawList,
    ImVec2 const& graphOrigin,
    float graphWidth)
{
    auto& inputIndex = selectionData.inputIndex;
    auto& outputIndex = selectionData.outputIndex;
    inputIndex = std::clamp(inputIndex, 0, NEURAL_NET_INPUTS - 1);
    outputIndex = std::clamp(outputIndex, 0, NEURAL_NET_OUTPUTS - 1);

    auto lineHeight = ImGui::GetTextLineHeight();
    auto frameHeight = ImGui::GetFrameHeight();
    auto itemSpacing = scale(CardItemSpacing);

    // Shrink the card in narrow windows so that it does not cover the node columns
    auto scaleFactor = scale(1.0f);
    auto spaceBetweenNodeColumns = graphWidth / scaleFactor - 2 * GraphSideMargin;
    auto cardTotalWidth = std::clamp(spaceBetweenNodeColumns, CardMinWidth, CardWidth);
    auto cardContentWidth = cardTotalWidth - 2 * CardPadding;
    auto cardWidth = scale(cardTotalWidth);
    auto cardHeight = 2 * scale(CardPadding) + lineHeight + 3 * itemSpacing + 2 * frameHeight + scale(ActivationIconHeight);

    // Keep the card inside the visible area when the editor is scrolled
    auto cardY = graphOrigin.y + scale(CardTopMargin);
    auto clipMin = drawList->GetClipRectMin();
    auto clipMax = drawList->GetClipRectMax();
    if (clipMax.y - clipMin.y > cardHeight + 2 * scale(CardTopMargin)) {
        cardY = std::clamp(cardY, clipMin.y + scale(CardTopMargin), clipMax.y - cardHeight - scale(CardTopMargin));
    }

    ImVec2 cardMin{graphOrigin.x + (graphWidth - cardWidth) / 2, cardY};
    ImVec2 cardMax{cardMin.x + cardWidth, cardMin.y + cardHeight};

    drawList->AddRectFilled(cardMin, cardMax, CardBackgroundColor, scale(4.0f));
    drawList->AddRect(cardMin, cardMax, CardBorderColor, scale(4.0f), 0, scale(1.0f));

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
    auto sliderWidth = (scale(cardContentWidth) - 2 * spacing + scale(7.0f) - resetButtonWidth() - labelWidth) / scaleFactor;
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
    auto buttonWidth = (scale(cardContentWidth) - buttonSpacing * (numActivationFunctions - 1)) / numActivationFunctions;
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

_NeuralNetEditorWidget::_NeuralNetEditorWidget() {}

void _NeuralNetEditorWidget::processActionButtons(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions)
{
    if (ImGui::BeginChild("ActionButtons", ImVec2(0, scale(50.0f)))) {
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
    ImGui::EndChild();
}

ImColor _NeuralNetEditorWidget::calcWeightColor(float value, float alpha)
{
    auto result = value > 0 ? PositiveWeightColor : NegativeWeightColor;
    result.Value.w = alpha;
    return result;
}

std::string _NeuralNetEditorWidget::getInputLabel(int inputIndex)
{
    if (inputIndex < STANDARD_NEURONS_PER_CELL) {
        return "Sig " + std::to_string(inputIndex + 1);
    } else if (inputIndex < NEURAL_NET_OUTPUTS) {
        return "Mem " + std::to_string(inputIndex - STANDARD_NEURONS_PER_CELL + 1);
    }
    return TelemetryLabels.at(inputIndex - NEURAL_NET_OUTPUTS);
}

std::string _NeuralNetEditorWidget::getOutputLabel(int outputIndex)
{
    if (outputIndex < STANDARD_NEURONS_PER_CELL) {
        return "Out " + std::to_string(outputIndex + 1);
    }
    return "Mem " + std::to_string(outputIndex - STANDARD_NEURONS_PER_CELL + 1);
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
