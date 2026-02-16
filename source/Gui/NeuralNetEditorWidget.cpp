#include "NeuralNetEditorWidget.h"

#include <algorithm>
#include <cmath>

#include <glad/glad.h>

#include <imgui.h>

#include <Base/Math.h>

#include <EngineInterface/NumberGenerator.h>

#include "AlienGui.h"
#include "HelpStrings.h"
#include "StyleRepository.h"

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
    auto pushDefaultColors = [] {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_FrameBg]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyle().Colors[ImGuiCol_FrameBgHovered]);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyle().Colors[ImGuiCol_FrameBg]);
    };
    auto pushHighlightingColors = [] {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyle().Colors[ImGuiCol_FrameBgActive]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyle().Colors[ImGuiCol_FrameBgActive]);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyle().Colors[ImGuiCol_FrameBgActive]);
    };
    auto popColors = [] { ImGui::PopStyleColor(3); };
    auto calcWeightColor = [](float value) {
        auto factor = std::min(1.0f, std::abs(value));
        if (value > NEAR_ZERO) {
            return ImColor::HSV(0.61f, 0.7f, std::max(1.0f * factor, 0.1f), 0.5f);
        } else if (value < -NEAR_ZERO) {
            return ImColor::HSV(0.0f, 0.7f, std::max(1.0f * factor, 0.1f), 0.5f);
        } else {
            return ImColor::HSV(0.0f, 0.0f, 0.1f, 0.5f);
        }
    };
    auto calcBiasColor = [](float value) {
        auto factor = std::min(1.0f, std::abs(value));
        if (value > NEAR_ZERO) {
            return ImColor::HSV(0.61f, 0.7f, std::max(1.0f * factor, 0.2f), 1.0f);
        } else if (value < -NEAR_ZERO) {
            return ImColor::HSV(0.0f, 0.7f, std::max(1.0f * factor, 0.2f), 1.0f);
        } else {
            return ImColor::HSV(0.0f, 0.0f, 0.2f, 0.5f);
        }
    };
    auto& selectionData = getValueRef(_dataById);
    auto constexpr BiasHeight = 8.0f;

    if (ImGui::BeginChild("NeuralNetEditor", ImVec2(0, 0), 0, ImGuiWindowFlags_AlwaysVerticalScrollbar)) {
        auto drawList = ImGui::GetWindowDrawList();

        // Connection weights info button
        auto width = ImGui::GetContentRegionAvail().x;
        auto connectionButtonWidth = width / MAX_OBJECT_CONNECTIONS - 2 * ImGui::GetStyle().FramePadding.x;

        pushDefaultColors();
        ImGui::Button("Connection weights", {width - 2 * ImGui::GetStyle().FramePadding.x, 0});
        popColors();

        // Connection weight sliders
        ImGui::PushID("ConnectionWeightSliders");
        ImVec2 connectionButtonBottomLeft[MAX_OBJECT_CONNECTIONS];
        ImVec2 connectionButtonBottomRight[MAX_OBJECT_CONNECTIONS];
        for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
            if (i > 0) {
                ImGui::SameLine();
            }
            ImGui::PushID(i);
            ImGuiStyle& style = ImGui::GetStyle();
            auto originalGrabMinSize = style.GrabMinSize;
            style.GrabMinSize = scale(8.0f);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().format("%.2f").width(connectionButtonWidth).textWidth(0).min(-2.0f).max(2.0f), &connectionWeights.at(i));
            style.GrabMinSize = originalGrabMinSize;
            ImGui::PopID();
            connectionButtonBottomLeft[i] = {ImGui::GetItemRectMin().x, ImGui::GetItemRectMax().y};
            connectionButtonBottomRight[i] = {ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y};
        }
        ImGui::PopID();

        // Placeholder for connection weights
        ImGui::Dummy(ImVec2(0, scale(20.0f)));

        // Channel weights info button
        pushDefaultColors();
        ImGui::Button("Channel weights", {width - 2 * ImGui::GetStyle().FramePadding.x, 0});
        popColors();

        // Channel weight sliders
        ImGui::PushID("ChannelWeightSliders");
        auto channelsButtonTopCenter = ImVec2{(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, ImGui::GetItemRectMin().y};
        auto channelButtonWidth = width / MAX_CHANNELS - 2 * ImGui::GetStyle().FramePadding.x;
        ImVec2 inputChannelBottomCenter[MAX_CHANNELS];
        ImVec2 inputButtonTopLeft[MAX_CHANNELS];
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            if (i > 0) {
                ImGui::SameLine();
            }
            ImGui::PushID(i);
            ImGui::SetWindowFontScale(0.8f);
            ImGuiStyle& style = ImGui::GetStyle();
            auto originalGrabMinSize = style.GrabMinSize;
            style.GrabMinSize = scale(4.0f);
            auto weight = weights.at(i + selectionData.neuronIndex * MAX_CHANNELS).getValue();
            AlienGui::SliderFloat(AlienGui::SliderFloatParameters().format("%.2f").width(channelButtonWidth).textWidth(0).min(-2.0f).max(2.0f), &weight);
            weights.at(i + selectionData.neuronIndex * MAX_CHANNELS) = weight;
            style.GrabMinSize = originalGrabMinSize;
            ImGui::SetWindowFontScale(1.0f);
            ImGui::PopID();

            inputChannelBottomCenter[i] = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, ImGui::GetItemRectMax().y};
            inputButtonTopLeft[i] = {ImGui::GetItemRectMin().x, ImGui::GetItemRectMin().y};
        }
        ImGui::PopID();

        // Placeholder for channel weights
        ImGui::Dummy(ImVec2(0, scale(70.0f)));

        // Bias visualization
        {
            auto yPos = ImGui::GetCursorScreenPos().y;
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                drawList->AddRectFilled(
                    {inputChannelBottomCenter[i].x - channelButtonWidth / 4, yPos - scale(BiasHeight)},
                    {inputChannelBottomCenter[i].x + channelButtonWidth / 4, yPos},
                    calcBiasColor(biases[i]));
            }
        }

        // Neuron selection buttons
        ImVec2 neuronTopCenter[MAX_CHANNELS];
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            if (i > 0) {
                ImGui::SameLine();
            }
            ImGui::SetWindowFontScale(0.7f);
            if (i == selectionData.neuronIndex) {
                pushHighlightingColors();
            } else {
                pushDefaultColors();
            }
            ImGui::SetCursorScreenPos({inputButtonTopLeft[i].x, ImGui::GetCursorScreenPos().y});
            if (ImGui::Button((std::to_string(i + 1) + "##output").c_str(), {channelButtonWidth, 0})) {
                selectionData.neuronIndex = i;
            }
            popColors();
            ImGui::SetWindowFontScale(1.0f);
            neuronTopCenter[i] = {(ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) / 2, ImGui::GetItemRectMin().y - scale(BiasHeight)};
        }

        // Bias sliders
        ImGui::PushID("BiasSliders");
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            if (i > 0) {
                ImGui::SameLine();
            }
            ImGui::PushID(i);
            ImGui::SetWindowFontScale(0.8f);
            ImGuiStyle& style = ImGui::GetStyle();
            auto originalGrabMinSize = style.GrabMinSize;
            style.GrabMinSize = scale(4.0f);
            auto bias = biases.at(i);
            AlienGui::SliderFloat(AlienGui::SliderFloatParameters().format("%.2f").width(channelButtonWidth).textWidth(0).min(-2.0f).max(2.0f), &bias);
            biases.at(i) = bias;
            style.GrabMinSize = originalGrabMinSize;
            ImGui::SetWindowFontScale(1.0f);
            ImGui::PopID();
        }
        ImGui::PopID();

        // Draw activation function plots below bias sliders
        {
            RealVector2D const plotSize{channelButtonWidth * 0.8f, scale(20.0f)};
            auto calcPlotPosition = [&](RealVector2D const& refPos, float x, ActivationFunction activationFunction) {
                float value = 0;
                switch (activationFunction) {
                case ActivationFunction_Sigmoid:
                    value = Math::sigmoid(x);
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
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                RealVector2D refPos{inputChannelBottomCenter[i].x - plotSize.x / 2, yPos + plotSize.y / 2};

                // Draw grid lines
                for (float dx = 0; dx <= plotSize.x + NEAR_ZERO; dx += plotSize.x / 8) {
                    auto color = std::abs(dx - plotSize.x / 2) < NEAR_ZERO ? Const::NeuronEditorZeroLinePlotColor : Const::NeuronEditorGridColor;
                    drawList->AddLine({refPos.x + dx, refPos.y - plotSize.y / 2}, {refPos.x + dx, refPos.y + plotSize.y / 2}, color, 1.0f);
                }
                for (float dy = -plotSize.y / 2; dy <= plotSize.y / 2 + NEAR_ZERO; dy += plotSize.y / 6) {
                    auto color = std::abs(dy) < NEAR_ZERO ? Const::NeuronEditorZeroLinePlotColor : Const::NeuronEditorGridColor;
                    drawList->AddLine({refPos.x, refPos.y + dy}, {refPos.x + plotSize.x, refPos.y + dy}, color, 1.0f);
                }

                // Draw activation function curve
                std::optional<RealVector2D> lastPos;
                for (float dx = -4.0f; dx < 4.0f; dx += 0.2f) {
                    RealVector2D pos = calcPlotPosition(refPos, dx, activationFunctions[i]);
                    if (lastPos) {
                        drawList->AddLine({lastPos->x, lastPos->y}, {pos.x, pos.y}, Const::NeuronEditorPlotColor, 1.0f);
                    }
                    lastPos = pos;
                }
            }

            // Invisible buttons for clicking to cycle activation function
            ImGui::PushID("ActivationFunctions");
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                if (i > 0) {
                    ImGui::SameLine();
                }
                ImGui::PushID(i);
                ImGui::SetCursorScreenPos({inputButtonTopLeft[i].x, yPos});
                if (ImGui::InvisibleButton("##actfn", {channelButtonWidth, plotSize.y})) {
                    activationFunctions[i] = (activationFunctions[i] + 1) % ActivationFunction_Count;
                }
                ImGui::PopID();
            }
            ImGui::PopID();
        }

        // Draw connection weight lines
        {
            for (int i = 0; i < MAX_OBJECT_CONNECTIONS; ++i) {
                auto value = connectionWeights[i];
                if (std::abs(value) <= NEAR_ZERO) {
                    continue;
                }
                auto halfWidth = /*scale(2.0f) * factor;*/ connectionButtonWidth * std::min(1.0f, std::abs(value) / 2.0f) * 0.35f;
                auto centerX = (connectionButtonBottomLeft[i].x + connectionButtonBottomRight[i].x) / 2.0f;
                auto topY = connectionButtonBottomLeft[i].y;
                drawList->AddQuadFilled(
                    {centerX - halfWidth, topY},
                    {centerX + halfWidth, topY},
                    {centerX + halfWidth, channelsButtonTopCenter.y},
                    {centerX - halfWidth, channelsButtonTopCenter.y},
                    calcWeightColor(value));
            }
        }
        // Draw channel weight lines
        {
            struct WeightLine
            {
                int i;
                int j;
                float weightFloat;
            };
            std::vector<WeightLine> weightLines;
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    auto weightFloat = weights[j * MAX_CHANNELS + i].getValue();
                    weightLines.push_back({i, j, weightFloat});
                }
            }
            std::sort(weightLines.begin(), weightLines.end(), [](auto const& a, auto const& b) { return std::abs(a.weightFloat) < std::abs(b.weightFloat); });
            for (auto const& wl : weightLines) {
                auto thickness = std::min(4.0f, std::abs(wl.weightFloat));
                drawList->AddLine(inputChannelBottomCenter[wl.i], neuronTopCenter[wl.j], calcWeightColor(wl.weightFloat), scale(thickness));
            }
        }

        AlienGui::Separator();

        processActionButtons(weights, biases, activationFunctions);
    }
    ImGui::EndChild();
}

_NeuralNetEditorWidget::_NeuralNetEditorWidget() {}

void _NeuralNetEditorWidget::processActionButtons(
    std::vector<NeuralNetWeight>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions)
{
    if (ImGui::BeginChild("ActionButtons", ImVec2(0, scale(50.0f)))) {
        if (AlienGui::Button("Clear")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    weights[i * MAX_CHANNELS + j] = NeuralNetWeight(0);
                }
                biases[i] = 0;
                activationFunctions[i] = ActivationFunction_Identity;
            }
        }
        ImGui::SameLine();
        if (AlienGui::Button("Identity")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    weights[i * MAX_CHANNELS + j] = (i == j) ? NeuralNetWeight(1.0f) : NeuralNetWeight(0);
                }
                biases[i] = 0.0f;
                activationFunctions[i] = ActivationFunction_Identity;
            }
        }
        ImGui::SameLine();
        if (AlienGui::Button("Randomize")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    weights[i * MAX_CHANNELS + j] = NeuralNetWeight(NumberGenerator::get().getRandomFloat(-2.0f, 2.0f));
                }
                biases[i] = NumberGenerator::get().getRandomFloat(-2.0f, 2.0f);
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

template <typename T>
_NeuralNetEditorWidget::SelectionData& _NeuralNetEditorWidget::getValueRef(std::unordered_map<unsigned, T>& idToValueMap)
{
    auto id = ImGui::GetID("");
    if (!idToValueMap.contains(id)) {
        idToValueMap[id] = T();
    }
    return idToValueMap.at(id);
}
