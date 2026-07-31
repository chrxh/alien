#include "NeuralNetEditorWidget.h"

#include <algorithm>
#include <cmath>

#include <imgui.h>

#include <Base/StringHelper.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/NumberGenerator.h>

#include "AlienGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr GraphRowSpacing = 22.0f;
    auto constexpr GraphGroupSpacing = 12.0f;
    auto constexpr GraphHeaderHeight = 26.0f;
    auto constexpr GraphSideMargin = 120.0f;
    auto constexpr NodeRadius = 5.0f;
    auto constexpr NodeClickPadding = 3.0f;
    auto constexpr ConnectionNodeSpacing = 64.0f;
    auto constexpr DetailTextWidth = 130.0f;

    auto const PositiveWeightColor = ImColor(77, 163, 255);
    auto const NegativeWeightColor = ImColor(255, 77, 77);
    auto const SignalNodeColor = ImColor(77, 163, 255);
    auto const MemoryNodeColor = ImColor(180, 140, 255);
    auto const TelemetryNodeColor = ImColor(111, 220, 140);
    auto const SelectedNodeColor = ImColor(255, 210, 77);
    auto const NodeFillColor = ImColor(18, 21, 27);
    auto const ZeroWeightColor = ImColor(90, 97, 110);
    auto const LabelColor = ImColor(170, 178, 194);

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
        processConnectionWeights(connectionWeights, selectionData);
        processGraph(weights, activationFunctions, selectionData, liveData);
        processDetailPanel(weights, biases, activationFunctions, selectionData);

        AlienGui::Separator();

        processActionButtons(weights, biases, activationFunctions);
    }
    ImGui::EndChild();
}

void _NeuralNetEditorWidget::processConnectionWeights(std::vector<float>& connectionWeights, SelectionData& selectionData)
{
    auto rowHeight = ImGui::GetTextLineHeight() + scale(34.0f);
    if (ImGui::BeginChild("ConnectionWeights", ImVec2(0, rowHeight), 0, 0)) {
        auto drawList = ImGui::GetWindowDrawList();
        auto origin = ImGui::GetCursorScreenPos();

        drawList->AddText({origin.x, origin.y}, withAlpha(LabelColor, 0.8f), "CONNECTIONS");

        auto nodeY = origin.y + ImGui::GetTextLineHeight() + scale(18.0f);
        ImGui::PushID("ConnectionNodes");
        for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
            auto const& weight = connectionWeights.at(i);
            ImVec2 pos{origin.x + scale(14.0f + toFloat(i) * ConnectionNodeSpacing), nodeY};

            ImGui::PushID(i);
            bool hovered = false;
            if (nodeClickArea(pos, "##connectionNode", hovered)) {
                selectionData.connectionIndex = i;
            }
            ImGui::PopID();

            auto isSelected = i == selectionData.connectionIndex;
            if (isSelected) {
                drawList->AddCircle(pos, scale(NodeRadius + 3.5f), SelectedNodeColor, 0, scale(1.2f));
            }
            drawList->AddCircleFilled(pos, scale(NodeRadius), NodeFillColor);
            auto borderColor = std::abs(weight) > NEAR_ZERO ? calcWeightColor(weight, 1.0f) : ZeroWeightColor;
            drawList->AddCircle(pos, scale(NodeRadius), hovered ? SelectedNodeColor : borderColor, 0, scale(1.5f));

            auto valueText = StringHelper::format(weight, 2);
            drawList->AddText(
                {pos.x + scale(NodeRadius + 5.0f), pos.y - ImGui::GetTextLineHeight() / 2},
                isSelected ? ImColor(255, 255, 255) : LabelColor,
                valueText.c_str());
        }
        ImGui::PopID();

        // Slider for the selected connection weight
        auto sliderX = origin.x + scale(14.0f + toFloat(MAX_OBJECT_CONNECTIONS) * ConnectionNodeSpacing);
        ImGui::SetCursorScreenPos({sliderX, nodeY - ImGui::GetFrameHeight() / 2});
        AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters()
                .name("Con " + std::to_string(selectionData.connectionIndex + 1))
                .format("%.2f")
                .textWidth(60.0f)
                .min(-1.0f)
                .max(1.0f),
            &connectionWeights.at(selectionData.connectionIndex));
    }
    ImGui::EndChild();
}

void _NeuralNetEditorWidget::processGraph(
    std::vector<NeuralNetWeight>& weights,
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
        auto textSize = ImGui::CalcTextSize(label.c_str());
        drawList->AddText(
            {pos.x - scale(NodeRadius + 6.0f) - textSize.x, pos.y - textSize.y / 2}, isSelected ? ImColor(255, 255, 255) : LabelColor, label.c_str());

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
                drawList->AddText({pos.x + scale(NodeRadius + 6.0f), pos.y - textSize.y / 2}, withAlpha(LabelColor, 0.6f), value.c_str());
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
        auto textSize = ImGui::CalcTextSize(label.c_str());
        drawList->AddText({pos.x + scale(NodeRadius + 6.0f), pos.y - textSize.y / 2}, isSelected ? ImColor(255, 255, 255) : LabelColor, label.c_str());

        auto const& actfnLabel = ActivationFunctionShortStrings.at(activationFunctions.at(i));
        drawList->AddText(
            {pos.x + scale(NodeRadius + 6.0f) + textSize.x + scale(8.0f), pos.y - textSize.y / 2}, withAlpha(LabelColor, 0.55f), actfnLabel.c_str());
    }
    ImGui::PopID();
}

void _NeuralNetEditorWidget::processDetailPanel(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    SelectionData& selectionData)
{
    auto& inputIndex = selectionData.inputIndex;
    auto& outputIndex = selectionData.outputIndex;
    inputIndex = std::clamp(inputIndex, 0, NEURAL_NET_INPUTS - 1);
    outputIndex = std::clamp(outputIndex, 0, NEURAL_NET_OUTPUTS - 1);

    auto weight = weights.at(outputIndex * NEURAL_NET_INPUTS + inputIndex).getValue();
    if (AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters()
                .name(getInputLabel(inputIndex) + " -> " + getOutputLabel(outputIndex))
                .format("%.2f")
                .textWidth(DetailTextWidth)
                .min(-2.0f)
                .max(2.0f),
            &weight)) {
        weights.at(outputIndex * NEURAL_NET_INPUTS + inputIndex) = weight;
    }

    AlienGui::SliderFloat(
        AlienGui::SliderFloatParameters().name("Bias (" + getOutputLabel(outputIndex) + ")").format("%.2f").textWidth(DetailTextWidth).min(-2.0f).max(2.0f),
        &biases.at(outputIndex));

    int activationFunction = activationFunctions.at(outputIndex);
    if (AlienGui::Combo(
            AlienGui::ComboParameters()
                .name("Activation (" + getOutputLabel(outputIndex) + ")")
                .values(Const::ActivationFunctionStrings)
                .textWidth(DetailTextWidth),
            activationFunction)) {
        activationFunctions.at(outputIndex) = static_cast<ActivationFunction>(activationFunction);
    }
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
