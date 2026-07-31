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
    auto constexpr GraphGroupSpacing = 16.0f;
    auto constexpr GraphHeaderHeight = 18.0f;
    auto constexpr GraphSideMargin = 75.0f;
    auto constexpr NodeRadius = 5.0f;

    auto const PositiveWeightColor = ImColor(77, 163, 255);
    auto const NegativeWeightColor = ImColor(255, 77, 77);
    auto const SignalNodeColor = ImColor(77, 163, 255);
    auto const MemoryNodeColor = ImColor(180, 140, 255);
    auto const TelemetryNodeColor = ImColor(111, 220, 140);
    auto const SelectedNodeColor = ImColor(255, 210, 77);
    auto const NodeFillColor = ImColor(18, 21, 27);
    auto const LabelColor = ImColor(170, 178, 194);
    auto const HeaderColorAlpha = 200;

    std::vector<std::string> const TelemetryLabels = {"Energy", "Attacked", "Age", "Speed"};

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
        processDetailPanel(weights, biases, activationFunctions, selectionData);

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
        drawInputNodes(drawList, layout, liveData);
        drawOutputNodes(biases, activationFunctions, selectionData, drawList, layout);
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
    };
    std::vector<WeightCurve> curves;
    curves.reserve(NEURAL_NET_OUTPUTS * NEURAL_NET_INPUTS);
    for (int row = 0; row < NEURAL_NET_OUTPUTS; ++row) {
        for (int col = 0; col < NEURAL_NET_INPUTS; ++col) {
            auto weight = weights.at(row * NEURAL_NET_INPUTS + col).getValue();
            if (std::abs(weight) <= NEAR_ZERO) {
                continue;
            }
            curves.push_back({col, row, weight});
        }
    }
    std::sort(curves.begin(), curves.end(), [&](auto const& a, auto const& b) {
        if ((a.outputIndex == selectionData.neuronIndex) != (b.outputIndex == selectionData.neuronIndex)) {
            return b.outputIndex == selectionData.neuronIndex;
        }
        return std::abs(a.weight) < std::abs(b.weight);
    });

    for (auto const& curve : curves) {
        auto isSelected = curve.outputIndex == selectionData.neuronIndex;
        auto start = layout.inputNodePos[curve.inputIndex];
        auto end = layout.outputNodePos[curve.outputIndex];
        auto controlOffset = (end.x - start.x) * 0.4f;
        auto thickness = scale(std::max(0.7f, std::min(2.0f, std::abs(curve.weight)) * (isSelected ? 1.7f : 1.0f)));
        drawList->AddBezierCubic(
            {start.x + scale(NodeRadius), start.y},
            {start.x + controlOffset, start.y},
            {end.x - controlOffset, end.y},
            {end.x - scale(NodeRadius), end.y},
            calcWeightColor(curve.weight, isSelected ? 0.9f : 0.13f),
            thickness);
    }
}

void _NeuralNetEditorWidget::drawInputNodes(ImDrawList* drawList, LayoutData const& layout, std::optional<LiveData> const& liveData)
{
    ImGui::SetWindowFontScale(0.7f);
    auto drawGroupHeader = [&](std::string const& text, ImColor const& color, float y) {
        auto headerColor = color;
        headerColor.Value.w = toFloat(HeaderColorAlpha) / 255;
        drawList->AddText({layout.inputNodePos[0].x - scale(GraphSideMargin - 5.0f), y}, headerColor, text.c_str());
    };
    drawGroupHeader("SIGNALS", SignalNodeColor, layout.inputNodePos[0].y - scale(GraphHeaderHeight));
    drawGroupHeader("MEMORY", MemoryNodeColor, layout.inputNodePos[STANDARD_NEURONS_PER_CELL].y - scale(GraphHeaderHeight));
    drawGroupHeader("TELEMETRY", TelemetryNodeColor, layout.inputNodePos[NEURAL_NET_OUTPUTS].y - scale(GraphHeaderHeight));

    for (int i = 0; i < NEURAL_NET_INPUTS; ++i) {
        auto const& pos = layout.inputNodePos[i];
        drawList->AddCircleFilled(pos, scale(NodeRadius), NodeFillColor);
        drawList->AddCircle(pos, scale(NodeRadius), groupColor(i), 0, scale(1.5f));
        if (i >= STANDARD_NEURONS_PER_CELL && i < NEURAL_NET_OUTPUTS) {
            drawList->AddCircleFilled(pos, scale(2.0f), MemoryNodeColor);
        }

        auto label = getInputLabel(i);
        auto textSize = ImGui::CalcTextSize(label.c_str());
        drawList->AddText({pos.x - scale(NodeRadius + 5.0f) - textSize.x, pos.y - textSize.y / 2}, LabelColor, label.c_str());

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
                auto valueColor = LabelColor;
                valueColor.Value.w = 0.6f;
                drawList->AddText({pos.x + scale(NodeRadius + 5.0f), pos.y - textSize.y / 2}, valueColor, value.c_str());
            }
        }
    }
    ImGui::SetWindowFontScale(1.0f);
}

void _NeuralNetEditorWidget::drawOutputNodes(
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    SelectionData& selectionData,
    ImDrawList* drawList,
    LayoutData const& layout)
{
    ImGui::SetWindowFontScale(0.7f);
    auto drawGroupHeader = [&](std::string const& text, ImColor const& color, float y) {
        auto headerColor = color;
        headerColor.Value.w = toFloat(HeaderColorAlpha) / 255;
        drawList->AddText({layout.outputNodePos[0].x + scale(NodeRadius + 5.0f), y}, headerColor, text.c_str());
    };
    drawGroupHeader("OUTPUTS", SignalNodeColor, layout.outputNodePos[0].y - scale(GraphHeaderHeight));
    drawGroupHeader("MEMORY", MemoryNodeColor, layout.outputNodePos[STANDARD_NEURONS_PER_CELL].y - scale(GraphHeaderHeight));

    for (int i = 0; i < NEURAL_NET_OUTPUTS; ++i) {
        auto const& pos = layout.outputNodePos[i];
        auto isSelected = i == selectionData.neuronIndex;

        // Click area
        ImGui::SetCursorScreenPos({pos.x - scale(NodeRadius + 3.0f), pos.y - scale(NodeRadius + 3.0f)});
        ImGui::PushID(i);
        if (ImGui::InvisibleButton("##outputNode", {scale((NodeRadius + 3.0f) * 2), scale((NodeRadius + 3.0f) * 2)})) {
            selectionData.neuronIndex = i;
        }
        auto isHovered = ImGui::IsItemHovered();
        ImGui::PopID();

        if (isSelected) {
            drawList->AddCircle(pos, scale(NodeRadius + 3.5f), SelectedNodeColor, 0, scale(1.2f));
        }
        drawList->AddCircleFilled(pos, scale(NodeRadius), NodeFillColor);
        auto borderColor = isSelected ? SelectedNodeColor : (i < STANDARD_NEURONS_PER_CELL ? SignalNodeColor : MemoryNodeColor);
        drawList->AddCircle(pos, scale(NodeRadius), isHovered ? SelectedNodeColor : borderColor, 0, scale(1.5f));
        if (i >= STANDARD_NEURONS_PER_CELL) {
            drawList->AddCircleFilled(pos, scale(2.0f), MemoryNodeColor);
        }

        auto label = getOutputLabel(i);
        auto textSize = ImGui::CalcTextSize(label.c_str());
        drawList->AddText({pos.x + scale(NodeRadius + 5.0f), pos.y - textSize.y / 2}, isSelected ? ImColor(255, 255, 255) : LabelColor, label.c_str());

        auto actfnLabel = Const::ActivationFunctionStrings.at(activationFunctions.at(i));
        auto actfnColor = LabelColor;
        actfnColor.Value.w = 0.55f;
        drawList->AddText({pos.x + scale(NodeRadius + 5.0f) + textSize.x + scale(6.0f), pos.y - textSize.y / 2}, actfnColor, actfnLabel.c_str());
    }
    ImGui::SetWindowFontScale(1.0f);
}

void _NeuralNetEditorWidget::processDetailPanel(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    SelectionData& selectionData)
{
    auto& neuronIndex = selectionData.neuronIndex;
    neuronIndex = std::clamp(neuronIndex, 0, NEURAL_NET_OUTPUTS - 1);

    AlienGui::Text(getOutputLabel(neuronIndex) + " - incoming weights");

    ImGuiStyle& style = ImGui::GetStyle();
    auto originalGrabMinSize = style.GrabMinSize;
    style.GrabMinSize = scale(8.0f);

    auto columnWidth = ImGui::GetContentRegionAvail().x / 2;
    auto sliderTextWidth = scale(55.0f);
    auto sliderWidth = columnWidth - sliderTextWidth - 2 * ImGui::GetStyle().FramePadding.x;

    ImGui::PushID("WeightSliders");
    ImGui::SetWindowFontScale(0.8f);
    for (int col = 0; col < NEURAL_NET_INPUTS; ++col) {
        auto columnIndex = col % 2;
        if (columnIndex > 0) {
            ImGui::SameLine(columnWidth);
        }
        ImGui::PushID(col);
        auto weight = weights.at(neuronIndex * NEURAL_NET_INPUTS + col).getValue();
        if (AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name(getInputLabel(col)).format("%.2f").width(sliderWidth).textWidth(sliderTextWidth).min(-2.0f).max(2.0f),
                &weight)) {
            weights.at(neuronIndex * NEURAL_NET_INPUTS + col) = weight;
        }
        ImGui::PopID();
    }
    ImGui::SetWindowFontScale(1.0f);
    ImGui::PopID();

    AlienGui::Separator();

    auto bias = biases.at(neuronIndex);
    if (AlienGui::SliderFloat(
            AlienGui::SliderFloatParameters().name("Bias").format("%.2f").width(sliderWidth).textWidth(sliderTextWidth).min(-2.0f).max(2.0f), &bias)) {
        biases.at(neuronIndex) = bias;
    }
    ImGui::SameLine(columnWidth);
    int activationFunction = activationFunctions.at(neuronIndex);
    if (AlienGui::Combo(
            AlienGui::ComboParameters().name("Activation").values(Const::ActivationFunctionStrings).textWidth(sliderTextWidth), activationFunction)) {
        activationFunctions.at(neuronIndex) = static_cast<ActivationFunction>(activationFunction);
    }

    style.GrabMinSize = originalGrabMinSize;
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
