#include "NeuralNetEditorWidget.h"

#include <algorithm>
#include <cmath>

#include <glad/glad.h>

#include <imgui.h>
#include <Fonts/IconsFontAwesome5.h>

#include <Base/Math.h>

#include <EngineInterface/NumberGenerator.h>

#include "AlienGui.h"
#include "HelpStrings.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr BiasHeight = 8.0f;
}

NeuralNetEditorWidget _NeuralNetEditorWidget::create()
{
    return NeuralNetEditorWidget(new _NeuralNetEditorWidget());
}

void _NeuralNetEditorWidget::process(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions,
    std::vector<float>& connectionWeights)
{
    auto& selectionData = getValueRef(_dataById);

    if (ImGui::BeginChild("NeuralNetEditor", ImVec2(0, 0), 0, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        auto drawList = ImGui::GetWindowDrawList();

        LayoutData layout;
        layout.width = ImGui::GetContentRegionAvail().x;
        layout.connectionButtonWidth = layout.width / MAX_OBJECT_CONNECTIONS - 2 * ImGui::GetStyle().FramePadding.x;
        layout.channelButtonWidth = layout.width / NEURONS_PER_CELL - 2 * ImGui::GetStyle().FramePadding.x;

        processConnectionWeightSliders(connectionWeights, layout);
        processChannelWeightSliders(weights, selectionData, layout);
        processBiasVisualization(biases, drawList, layout);
        processNeuronSelectionButtons(selectionData, layout);
        processBiasSliders(biases, layout);
        processActivationFunctions(activationFunctions, drawList, layout);
        drawConnectionWeightLines(connectionWeights, drawList, layout);
        drawChannelWeightLines(weights, drawList, layout);

        AlienGui::Separator();

        processActionButtons(weights, biases, activationFunctions);
    }
    ImGui::EndChild();
}

void _NeuralNetEditorWidget::processConnectionWeightSliders(std::vector<float>& connectionWeights, LayoutData& layout)
{
    pushDefaultColors();
    ImGui::Button("Connection weights", {layout.width - 2 * ImGui::GetStyle().FramePadding.x, 0});
    popColors();

    auto resetButtonWidth = ImGui::CalcTextSize("x").x /* + 2 * ImGui::GetStyle().FramePadding.x*/;
    auto sliderWidth = layout.connectionButtonWidth - resetButtonWidth - ImGui::GetStyle().ItemSpacing.x;

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
        layout.connectionButtonBottomLeft[i] = {ImGui::GetItemRectMin().x, ImGui::GetItemRectMax().y};
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - scale(7.0f));
        ImGui::SetWindowFontScale(0.5f);
        if (ImGui::Button(ICON_FA_TIMES)) {
            connectionWeights.at(i) = 0.0f;
        }
        ImGui::SetWindowFontScale(1.0f);
        layout.connectionButtonBottomRight[i] = {ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y};
        ImGui::PopID();
    }
    ImGui::PopID();

    ImGui::Dummy(ImVec2(0, scale(20.0f)));
}

void _NeuralNetEditorWidget::processChannelWeightSliders(std::vector<NeuralNetWeight>& weights, SelectionData& selectionData, LayoutData& layout)
{
    pushDefaultColors();
    ImGui::Button("Neuron weights", {layout.width - 2 * ImGui::GetStyle().FramePadding.x, 0});
    popColors();

    ImGui::PushID("ChannelWeightSliders");
    layout.channelsButtonTopCenter = ImVec2{(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, ImGui::GetItemRectMin().y};
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        if (i > 0) {
            ImGui::SameLine();
        }
        ImGui::PushID(i);
        ImGui::SetWindowFontScale(0.8f);
        ImGuiStyle& style = ImGui::GetStyle();
        auto originalGrabMinSize = style.GrabMinSize;
        style.GrabMinSize = scale(4.0f);
        auto weight = weights.at(i + selectionData.neuronIndex * NEURONS_PER_CELL).getValue();
        AlienGui::SliderFloat(AlienGui::SliderFloatParameters().format("%.2f").width(layout.channelButtonWidth).textWidth(0).min(-2.0f).max(2.0f), &weight);
        weights.at(i + selectionData.neuronIndex * NEURONS_PER_CELL) = weight;
        style.GrabMinSize = originalGrabMinSize;
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopID();

        layout.inputChannelBottomCenter[i] = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, ImGui::GetItemRectMax().y};
        layout.inputButtonTopLeft[i] = {ImGui::GetItemRectMin().x, ImGui::GetItemRectMin().y};
    }
    ImGui::PopID();

    ImGui::Dummy(ImVec2(0, scale(70.0f)));
}

void _NeuralNetEditorWidget::processBiasVisualization(std::vector<float>& biases, ImDrawList* drawList, LayoutData const& layout)
{
    auto yPos = ImGui::GetCursorScreenPos().y;
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        drawList->AddRectFilled(
            {layout.inputChannelBottomCenter[i].x - layout.channelButtonWidth / 4, yPos - scale(BiasHeight)},
            {layout.inputChannelBottomCenter[i].x + layout.channelButtonWidth / 4, yPos},
            calcBiasColor(biases[i]));
    }
}

void _NeuralNetEditorWidget::processNeuronSelectionButtons(SelectionData& selectionData, LayoutData& layout)
{
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        if (i > 0) {
            ImGui::SameLine();
        }
        ImGui::SetWindowFontScale(0.7f);
        if (i == selectionData.neuronIndex) {
            pushHighlightingColors();
        } else {
            pushDefaultColors();
        }
        ImGui::SetCursorScreenPos({layout.inputButtonTopLeft[i].x, ImGui::GetCursorScreenPos().y});
        if (ImGui::Button((std::to_string(i + 1) + "##output").c_str(), {layout.channelButtonWidth, 0})) {
            selectionData.neuronIndex = i;
        }
        popColors();
        ImGui::SetWindowFontScale(1.0f);
        layout.neuronTopCenter[i] = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, ImGui::GetItemRectMin().y - scale(BiasHeight)};
    }
}

void _NeuralNetEditorWidget::processBiasSliders(std::vector<float>& biases, LayoutData const& layout)
{
    ImGui::PushID("BiasSliders");
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        if (i > 0) {
            ImGui::SameLine();
        }
        ImGui::PushID(i);
        ImGui::SetWindowFontScale(0.8f);
        ImGuiStyle& style = ImGui::GetStyle();
        auto originalGrabMinSize = style.GrabMinSize;
        style.GrabMinSize = scale(4.0f);
        auto bias = biases.at(i);
        AlienGui::SliderFloat(AlienGui::SliderFloatParameters().format("%.2f").width(layout.channelButtonWidth).textWidth(0).min(-2.0f).max(2.0f), &bias);
        biases.at(i) = bias;
        style.GrabMinSize = originalGrabMinSize;
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopID();
    }
    ImGui::PopID();
}

void _NeuralNetEditorWidget::processActivationFunctions(std::vector<ActivationFunction>& activationFunctions, ImDrawList* drawList, LayoutData const& layout)
{
    RealVector2D const plotSize{layout.channelButtonWidth * 0.8f, scale(20.0f)};
    auto calcPlotPosition = [&](RealVector2D const& refPos, float x, ActivationFunction activationFunction) {
        float value = 0;
        switch (activationFunction) {
        case ActivationFunction_Tanh:
            value = Math::tanh(x);
            break;
        case ActivationFunction_BinaryStep:
            value = Math::binaryStep(x);
            break;
        case ActivationFunction_Identity:
            value = x / 4;
            break;
        case ActivationFunction_Abs:
            value = std::abs(x) / 4;
            break;
        case ActivationFunction_Gaussian:
            value = Math::gaussian(x);
            break;
        }
        return RealVector2D{refPos.x + plotSize.x / 2 + x * plotSize.x / 8, refPos.y - value * plotSize.y / 2};
    };

    auto yPos = ImGui::GetCursorScreenPos().y;
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        RealVector2D refPos{layout.inputChannelBottomCenter[i].x - plotSize.x / 2, yPos + plotSize.y / 2};

        for (float dx = 0; dx <= plotSize.x + NEAR_ZERO; dx += plotSize.x / 8) {
            auto color = std::abs(dx - plotSize.x / 2) < NEAR_ZERO ? Const::NeuronEditorZeroLinePlotColor : Const::NeuronEditorGridColor;
            drawList->AddLine({refPos.x + dx, refPos.y - plotSize.y / 2}, {refPos.x + dx, refPos.y + plotSize.y / 2}, color, 1.0f);
        }
        for (float dy = -plotSize.y / 2; dy <= plotSize.y / 2 + NEAR_ZERO; dy += plotSize.y / 6) {
            auto color = std::abs(dy) < NEAR_ZERO ? Const::NeuronEditorZeroLinePlotColor : Const::NeuronEditorGridColor;
            drawList->AddLine({refPos.x, refPos.y + dy}, {refPos.x + plotSize.x, refPos.y + dy}, color, 1.0f);
        }

        std::optional<RealVector2D> lastPos;
        for (float dx = -4.0f; dx < 4.0f; dx += 0.2f) {
            RealVector2D pos = calcPlotPosition(refPos, dx, activationFunctions[i]);
            if (lastPos) {
                drawList->AddLine({lastPos->x, lastPos->y}, {pos.x, pos.y}, Const::NeuronEditorPlotColor, 1.0f);
            }
            lastPos = pos;
        }
    }

    ImGui::PushID("ActivationFunctions");
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        if (i > 0) {
            ImGui::SameLine();
        }
        ImGui::PushID(i);
        ImGui::SetCursorScreenPos({layout.inputButtonTopLeft[i].x, yPos});
        if (ImGui::InvisibleButton("##actfn", {layout.channelButtonWidth, plotSize.y})) {
            activationFunctions[i] = (activationFunctions[i] + 1) % ActivationFunction_Count;
        }
        ImGui::PopID();
    }
    ImGui::PopID();
}

void _NeuralNetEditorWidget::drawConnectionWeightLines(std::vector<float>& connectionWeights, ImDrawList* drawList, LayoutData const& layout)
{
    for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
        auto value = connectionWeights[i];
        if (std::abs(value) <= NEAR_ZERO) {
            continue;
        }
        auto halfWidth = layout.connectionButtonWidth * std::min(1.0f, std::abs(value)) * 0.35f;
        auto centerX = (layout.connectionButtonBottomLeft[i].x + layout.connectionButtonBottomRight[i].x) / 2.0f - scale(7.0f);
        auto topY = layout.connectionButtonBottomLeft[i].y;
        drawList->AddQuadFilled(
            {centerX - halfWidth, topY},
            {centerX + halfWidth, topY},
            {centerX + halfWidth, layout.channelsButtonTopCenter.y},
            {centerX - halfWidth, layout.channelsButtonTopCenter.y},
            calcWeightColor(value));
    }
}

void _NeuralNetEditorWidget::drawChannelWeightLines(std::vector<NeuralNetWeight>& weights, ImDrawList* drawList, LayoutData const& layout)
{
    struct WeightLine
    {
        int i;
        int j;
        float weightFloat;
    };
    std::vector<WeightLine> weightLines;
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        for (int j = 0; j < NEURONS_PER_CELL; ++j) {
            auto weightFloat = weights[j * NEURONS_PER_CELL + i].getValue();
            weightLines.push_back({i, j, weightFloat});
        }
    }
    std::sort(weightLines.begin(), weightLines.end(), [](auto const& a, auto const& b) { return std::abs(a.weightFloat) < std::abs(b.weightFloat); });
    for (auto const& wl : weightLines) {
        auto thickness = std::min(4.0f, std::abs(wl.weightFloat));
        drawList->AddLine(layout.inputChannelBottomCenter[wl.i], layout.neuronTopCenter[wl.j], calcWeightColor(wl.weightFloat), scale(thickness));
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
            for (int i = 0; i < NEURONS_PER_CELL; ++i) {
                for (int j = 0; j < NEURONS_PER_CELL; ++j) {
                    weights[i * NEURONS_PER_CELL + j] = NeuralNetWeight(0);
                }
                biases[i] = 0;
                activationFunctions[i] = ActivationFunction_Identity;
            }
        }
        ImGui::SameLine();
        if (AlienGui::Button("Identity")) {
            for (int i = 0; i < NEURONS_PER_CELL; ++i) {
                for (int j = 0; j < NEURONS_PER_CELL; ++j) {
                    weights[i * NEURONS_PER_CELL + j] = (i == j) ? NeuralNetWeight(1.0f) : NeuralNetWeight(0);
                }
                biases[i] = 0.0f;
                activationFunctions[i] = ActivationFunction_Identity;
            }
        }
        ImGui::SameLine();
        if (AlienGui::Button("Randomize")) {
            for (int i = 0; i < NEURONS_PER_CELL; ++i) {
                for (int j = 0; j < NEURONS_PER_CELL; ++j) {
                    weights[i * NEURONS_PER_CELL + j] = NeuralNetWeight(NumberGenerator::get().getRandomFloat(-2.0f, 2.0f));
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

void _NeuralNetEditorWidget::pushDefaultColors()
{
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_FrameBg]);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyle().Colors[ImGuiCol_FrameBgHovered]);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyle().Colors[ImGuiCol_FrameBg]);
}

void _NeuralNetEditorWidget::pushHighlightingColors()
{
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_FrameBgActive]);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyle().Colors[ImGuiCol_FrameBgActive]);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyle().Colors[ImGuiCol_FrameBgActive]);
}

void _NeuralNetEditorWidget::popColors()
{
    ImGui::PopStyleColor(3);
}

ImColor _NeuralNetEditorWidget::calcWeightColor(float value)
{
    auto factor = std::min(1.0f, std::abs(value));
    if (value > NEAR_ZERO) {
        return ImColor::HSV(0.33f, 0.7f, std::max(1.0f * factor, 0.1f), 0.5f);
    } else if (value < -NEAR_ZERO) {
        return ImColor::HSV(0.0f, 0.7f, std::max(1.0f * factor, 0.1f), 0.5f);
    } else {
        return ImColor::HSV(0.0f, 0.0f, 0.1f, 0.5f);
    }
}

ImColor _NeuralNetEditorWidget::calcBiasColor(float value)
{
    auto factor = std::min(1.0f, std::abs(value));
    if (value > NEAR_ZERO) {
        return ImColor::HSV(0.33f, 0.7f, std::max(1.0f * factor, 0.2f), 1.0f);
    } else if (value < -NEAR_ZERO) {
        return ImColor::HSV(0.0f, 0.7f, std::max(1.0f * factor, 0.2f), 1.0f);
    } else {
        return ImColor::HSV(0.0f, 0.0f, 0.2f, 0.5f);
    }
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
