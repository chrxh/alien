#include "NeuralNetWidget.h"

#include <imgui.h>

#include "Base/Math.h"

#include "EngineInterface/NumberGenerator.h"

#include "AlienGui.h"
#include "HelpStrings.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr WidgetTextColumnWidth = 60.0f;
}

NeuralNetWidget _NeuralNetWidget::create()
{
    return NeuralNetWidget(new _NeuralNetWidget());
}

void _NeuralNetWidget::process(std::vector<float>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions)
{
    if (ImGui::BeginChild("NeuralNetEditor", ImVec2(0, 0))) {
        auto& selectionData = getValueRef(_dataById);

        processNetwork(selectionData, weights, biases, activationFunctions);

        ImGui::SameLine();
        processEditWidgets(selectionData, weights, biases, activationFunctions);

        processActionButtons(weights, biases, activationFunctions);
    }
    ImGui::EndChild();
}

_NeuralNetWidget::_NeuralNetWidget() {}

void _NeuralNetWidget::processNetwork(
    SelectionData& selectionData,
    std::vector<float>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions)
{
    if (ImGui::BeginChild("Network", ImVec2(ImGui::GetContentRegionAvail().x / 2, scale(200.0f)))) {
        // #TODO GCC incompatibily:
        // auto weights_span = std::mdspan(weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        auto pushDefaultColors = [] {
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::ToggleButtonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::ToggleButtonHoveredColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::ToggleButtonHoveredColor);
        };
        auto pushHighlightingColors = [] {
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::ToggleButtonActiveColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::ToggleButtonActiveColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::ToggleButtonActiveColor);
        };
        RealVector2D const ioButtonSize{scale(30.0f), scale(20.0f)};
        RealVector2D const plotSize{scale(50.0f), scale(20.0f)};
        auto const biasFieldWidth = ImGui::GetStyle().FramePadding.x * 2;
        auto width = ImGui::GetContentRegionAvail().x;

        auto drawList = ImGui::GetWindowDrawList();

        // Position functions
        auto startPos = ImGui::GetCursorScreenPos();
        auto style = ImGui::GetStyle();
        auto calcInputPos = [&](int i) { return ImVec2{startPos.x, startPos.y + (ioButtonSize.y + style.FramePadding.y + scale(1.0f)) * i}; };
        auto calcOutputPos = [&](int i) {
            return ImVec2{
                startPos.x + width - ioButtonSize.x - plotSize.x - biasFieldWidth - style.FramePadding.x * 2,
                startPos.y + (ioButtonSize.y + style.FramePadding.y + scale(1.0f)) * i};
        };

        // Draw selection
        {
            auto inputPos = calcInputPos(selectionData.inputNeuronIndex);
            auto outputPos = calcOutputPos(selectionData.outputNeuronIndex);
            drawList->AddLine(
                {inputPos.x + ioButtonSize.x, inputPos.y + ioButtonSize.y / 2},
                {outputPos.x, outputPos.y + ioButtonSize.y / 2},
                ImColor::HSV(0.0f, 0.0f, 0.35f, 1.0f),
                8.0f);
        }

        // Draw weights and biases
        auto calcColor = [](float value) {
            auto factor = std::min(1.0f, std::abs(value));
            if (value > NEAR_ZERO) {
                return ImColor::HSV(0.61f, 0.5f, 0.8f * factor);
            } else if (value < -NEAR_ZERO) {
                return ImColor::HSV(0.0f, 0.5f, 0.8f * factor);
            } else {
                return ImColor::HSV(0.0f, 0.0f, 0.1f);
            }
        };
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            auto inputPos = calcInputPos(i);
            for (int j = 0; j < MAX_CHANNELS; ++j) {
                if (std::abs(weights[j * MAX_CHANNELS + i]) <= NEAR_ZERO) {
                    continue;
                }
                auto thickness = std::min(4.0f, std::abs(weights[j * MAX_CHANNELS + i]));
                auto outputPos = calcOutputPos(j);
                drawList->AddLine(
                    {inputPos.x + ioButtonSize.x, inputPos.y + ioButtonSize.y / 2},
                    {outputPos.x, outputPos.y + ioButtonSize.y / 2},
                    calcColor(weights[j * MAX_CHANNELS + i]),
                    thickness);
            }

        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            auto outputPos = calcOutputPos(i);
            if (i == selectionData.outputNeuronIndex) {
                drawList->AddRectFilled(
                    {outputPos.x, outputPos.y}, {outputPos.x + biasFieldWidth, outputPos.y + ioButtonSize.y}, ImColor::HSV(0.0f, 0.0f, 0.35f, 1.0f));
            }
            drawList->AddRectFilled(
                {outputPos.x, outputPos.y + ioButtonSize.y / 4}, {outputPos.x + biasFieldWidth, outputPos.y + ioButtonSize.y * 3 / 4}, calcColor(biases[i]));
        }

        // Process buttons
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            ImGui::PushID(i);

            // Input button
            ImGui::SetCursorScreenPos(calcInputPos(i));
            if (i == selectionData.inputNeuronIndex) {
                pushHighlightingColors();
            } else {
                pushDefaultColors();
            }
            if (ImGui::Button(("#" + std::to_string(i) + "###Input").c_str(), {ioButtonSize.x, ioButtonSize.y})) {
                selectionData.inputNeuronIndex = i;
            }
            ImGui::PopStyleColor(3);


            // Output button
            auto outputButtonPos = calcOutputPos(i);
            ImGui::SetCursorScreenPos({outputButtonPos.x + biasFieldWidth, outputButtonPos.y});
            if (i == selectionData.outputNeuronIndex) {
                pushHighlightingColors();
            } else {
                pushDefaultColors();
            }
            if (ImGui::Button(("#" + std::to_string(i) + "###Output").c_str(), {ioButtonSize.x, ioButtonSize.y})) {
                selectionData.outputNeuronIndex = i;
            }
            ImGui::PopStyleColor(3);

            ImGui::PopID();
        }

        // Draw activation functions
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
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            auto outputButtonPos = calcOutputPos(i);
            std::optional<RealVector2D> lastPos;
            RealVector2D refPos{outputButtonPos.x + ioButtonSize.x + biasFieldWidth + ImGui::GetStyle().FramePadding.x * 2, outputButtonPos.y + plotSize.y / 2};
            for (float dx = 0; dx <= plotSize.x + NEAR_ZERO; dx += plotSize.x / 8) {
                auto color = std::abs(dx - plotSize.x / 2) < NEAR_ZERO ? Const::NeuronEditorZeroLinePlotColor : Const::NeuronEditorGridColor;
                drawList->AddLine({refPos.x + dx, refPos.y - plotSize.y / 2}, {refPos.x + dx, refPos.y + plotSize.y / 2}, color, 1.0f);
            }
            for (float dy = -plotSize.y / 2; dy <= plotSize.y / 2 + NEAR_ZERO; dy += plotSize.y / 6) {
                auto color = std::abs(dy) < NEAR_ZERO ? Const::NeuronEditorZeroLinePlotColor : Const::NeuronEditorGridColor;
                drawList->AddLine({refPos.x, refPos.y + dy}, {refPos.x + plotSize.x, refPos.y + dy}, color, 1.0f);
            }
            for (float dx = -4.0f; dx < 4.0f; dx += 0.2f) {
                RealVector2D pos = calcPlotPosition(refPos, dx, activationFunctions[i]);
                if (lastPos) {
                    drawList->AddLine({lastPos->x, lastPos->y}, {pos.x, pos.y}, Const::NeuronEditorPlotColor, 1.0f);
                }
                lastPos = pos;
            }
        }
    }
    ImGui::EndChild();
}

void _NeuralNetWidget::processEditWidgets(
    SelectionData& selectionData,
    std::vector<float>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions)
{
    if (ImGui::BeginChild("EditWidgets", ImVec2(0, 0))) {

        int activationFunction = activationFunctions.at(selectionData.outputNeuronIndex);
        AlienGui::Combo(AlienGui::ComboParameters().name("ActFn").values(Const::ActivationFunctionStrings).textWidth(WidgetTextColumnWidth), activationFunction);
        activationFunctions.at(selectionData.outputNeuronIndex) = static_cast<ActivationFunction>(activationFunction);

        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Weight").step(0.05f).textWidth(WidgetTextColumnWidth),
            weights[selectionData.outputNeuronIndex * MAX_CHANNELS + selectionData.inputNeuronIndex]);

        AlienGui::InputFloat(
            AlienGui::InputFloatParameters().name("Bias").step(0.05f).textWidth(WidgetTextColumnWidth), biases.at(selectionData.outputNeuronIndex));
    }
    ImGui::EndChild();
}

void _NeuralNetWidget::processActionButtons(std::vector<float>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions)
{
    if (ImGui::BeginChild("ActionButtons", ImVec2(0, scale(50.0f)))) {
        if (AlienGui::Button("Clear")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    weights[i * MAX_CHANNELS + j] = 0;
                }
                biases[i] = 0;
                activationFunctions[i] = ActivationFunction_Identity;
            }
        }
        ImGui::SameLine();
        if (AlienGui::Button("Identity")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    weights[i * MAX_CHANNELS + j] = i == j ? 1.0f : 0.0f;
                }
                biases[i] = 0.0f;
                activationFunctions[i] = ActivationFunction_Identity;
            }
        }
        ImGui::SameLine();
        if (AlienGui::Button("Randomize")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    weights[i * MAX_CHANNELS + j] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                }
                biases[i] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                activationFunctions[i] = NumberGenerator::get().getRandomInt(ActivationFunction_Count);
            }
        }
    }
    ImGui::EndChild();
}

template <typename T>
_NeuralNetWidget::SelectionData& _NeuralNetWidget::getValueRef(std::unordered_map<unsigned, T>& idToValueMap)
{
    auto id = ImGui::GetID("");
    if (!idToValueMap.contains(id)) {
        idToValueMap[id] = T();
    }
    return idToValueMap.at(id);
}
