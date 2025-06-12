#include "NeuralNetWidget.h"

#include <imgui.h>

#include "Base/Math.h"

#include "EngineInterface/NumberGenerator.h"

#include "AlienGui.h"
#include "StyleRepository.h"

NeuralNetWidget _NeuralNetWidget::create()
{
    return NeuralNetWidget(new _NeuralNetWidget());
}

namespace
{
    template <typename T>
    int& getIdBasedValue(std::unordered_map<unsigned int, T>& idToValueMap, T const& defaultValue)
    {
        auto id = ImGui::GetID("");
        if (!idToValueMap.contains(id)) {
            idToValueMap[id] = defaultValue;
        }
        return idToValueMap.at(id);
    }
}

void _NeuralNetWidget::process(std::vector<float>& weights, std::vector<float>& biases, std::vector<ActivationFunction>& activationFunctions)
{
    if (ImGui::BeginChild("Neural net", ImVec2(0, 0))) {
        // #TODO GCC incompatibily:
        // auto weights_span = std::mdspan(weights.data(), MAX_CHANNELS, MAX_CHANNELS);
        auto& selectedInput = getIdBasedValue(_neuronSelectedInput, 0);
        auto& selectedOutput = getIdBasedValue(_neuronSelectedOutput, 0);
        auto setDefaultColors = [] {
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::ToggleButtonColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::ToggleButtonHoveredColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::ToggleButtonHoveredColor);
        };
        auto setHightlightingColors = [] {
            ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::ToggleButtonActiveColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::ToggleButtonActiveColor);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::ToggleButtonActiveColor);
        };
        RealVector2D const ioButtonSize{scale(90.0f), scale(26.0f)};
        RealVector2D const plotSize{scale(50.0f), scale(23.0f)};
        auto const rightMargin = 0.0f;  //scale(_parameters._rightMargin);
        auto const biasFieldWidth = ImGui::GetStyle().FramePadding.x * 2;

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        auto windowPos = ImGui::GetWindowPos();

        RealVector2D inputPos[MAX_CHANNELS];
        RealVector2D outputPos[MAX_CHANNELS];

        //draw buttons and save positions to visualize weights
        for (int i = 0; i < MAX_CHANNELS; ++i) {

            auto buttonStartPos = ImGui::GetCursorPos();

            //input button
            i == selectedInput ? setHightlightingColors() : setDefaultColors();
            if (ImGui::Button(("Input #" + std::to_string(i)).c_str(), {ioButtonSize.x, ioButtonSize.y})) {
                selectedInput = i;
            }
            ImGui::PopStyleColor(3);

            inputPos[i] = RealVector2D(
                windowPos.x - ImGui::GetScrollX() + buttonStartPos.x + ioButtonSize.x,
                windowPos.y - ImGui::GetScrollY() + buttonStartPos.y + ioButtonSize.y / 2);

            ImGui::SameLine(0, ImGui::GetContentRegionAvail().x - ioButtonSize.x * 2 - plotSize.x - ImGui::GetStyle().FramePadding.x - rightMargin);
            buttonStartPos = ImGui::GetCursorPos();
            outputPos[i] = RealVector2D(
                windowPos.x - ImGui::GetScrollX() + buttonStartPos.x - biasFieldWidth - ImGui::GetStyle().FramePadding.x,
                windowPos.y - ImGui::GetScrollY() + buttonStartPos.y + ioButtonSize.y / 2);

            //output button
            i == selectedOutput ? setHightlightingColors() : setDefaultColors();
            if (ImGui::Button(("Output #" + std::to_string(i)).c_str(), {ioButtonSize.x, ioButtonSize.y})) {
                selectedOutput = i;
            }
            ImGui::PopStyleColor(3);
        }
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            for (int j = 0; j < MAX_CHANNELS; ++j) {
                // #TODO GCC incompatibily:
                // if (std::abs(weights_span[j, i]) > NEAR_ZERO) {
                if (std::abs(weights[j * MAX_CHANNELS + i]) <= NEAR_ZERO) {
                    continue;
                }
                drawList->AddLine({inputPos[i].x, inputPos[i].y}, {outputPos[j].x, outputPos[j].y}, Const::NeuronEditorConnectionColor, 2.0f);
            }
        }
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

        //visualize weights
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            for (int j = 0; j < MAX_CHANNELS; ++j) {
                // #TODO GCC incompatibily:
                // if (std::abs(weights_span[j, i]) <= NEAR_ZERO) {
                if (std::abs(weights[j * MAX_CHANNELS + i]) <= NEAR_ZERO) {
                    continue;
                }
                // #TODO GCC incompatibily:
                // auto thickness = std::min(4.0f, std::abs(weights_span[j, i]));
                auto thickness = std::min(4.0f, std::abs(weights[j * MAX_CHANNELS + i]));
                // #TODO GCC incompatibily:
                // drawList->AddLine({inputPos[i].x, inputPos[i].y}, {outputPos[j].x, outputPos[j].y}, calcColor(weights_span[j, i]), thickness);
                drawList->AddLine({inputPos[i].x, inputPos[i].y}, {outputPos[j].x, outputPos[j].y}, calcColor(weights[j * MAX_CHANNELS + i]), thickness);
            }
        }

        //visualize activation functions
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
            std::optional<RealVector2D> lastPos;
            RealVector2D refPos{outputPos[i].x + ioButtonSize.x + biasFieldWidth + ImGui::GetStyle().FramePadding.x * 2, outputPos[i].y};
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

        //visualize biases
        for (int i = 0; i < MAX_CHANNELS; ++i) {
            drawList->AddRectFilled(
                {outputPos[i].x, outputPos[i].y - biasFieldWidth}, {outputPos[i].x + biasFieldWidth, outputPos[i].y + biasFieldWidth}, calcColor(biases[i]));
        }

        //draw selection
        drawList->AddRectFilled(
            {outputPos[selectedOutput].x, outputPos[selectedOutput].y - biasFieldWidth},
            {outputPos[selectedOutput].x + biasFieldWidth, outputPos[selectedOutput].y + biasFieldWidth},
            ImColor::HSV(0.0f, 0.0f, 1.0f, 0.35f));
        drawList->AddLine(
            {inputPos[selectedInput].x, inputPos[selectedInput].y},
            {outputPos[selectedOutput].x, outputPos[selectedOutput].y},
            ImColor::HSV(0.0f, 0.0f, 1.0f, 0.35f),
            8.0f);

        auto const editorWidth = ImGui::GetContentRegionAvail().x - rightMargin;
        auto const editorColumnWidth = 280.0f;
        auto const editorColumnTextWidth = 155.0f;
        auto const numWidgets = 3;
        auto numColumns = AlienGui::DynamicTableLayout::calcNumColumns(editorWidth - ImGui::GetStyle().FramePadding.x * 4, editorColumnWidth);
        auto numRows = numWidgets / numColumns;
        if (numWidgets % numColumns != 0) {
            ++numRows;
        }
        if (ImGui::BeginChild("##", ImVec2(editorWidth, scale(toFloat(numRows) * 26.0f + 18.0f + 28.0f)), true)) {
            AlienGui::DynamicTableLayout table(editorColumnWidth);
            if (table.begin()) {
                int activationFunction = activationFunctions.at(selectedOutput);
                AlienGui::Combo(AlienGui::ComboParameters().name("Activation function").textWidth(editorColumnTextWidth), activationFunction);
                activationFunctions.at(selectedOutput) = static_cast<ActivationFunction>(activationFunction);
                table.next();
                // #TODO GCC incompatibily:
                // AlienGui::InputFloat(
                //     AlienGui::InputFloatParameters().name("Weight").step(0.05f).textWidth(editorColumnTextWidth).tooltip(Const::GenomeNeuronWeightAndBiasTooltip),
                //     weights_span[selectedOutput, selectedInput]);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Weight").step(0.05f).textWidth(editorColumnTextWidth),
                    weights[selectedOutput * MAX_CHANNELS + selectedInput]);
                table.next();
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Bias").step(0.05f).textWidth(editorColumnTextWidth), biases.at(selectedOutput));
                table.end();
            }
            if (AlienGui::Button("Clear")) {
                for (int i = 0; i < MAX_CHANNELS; ++i) {
                    for (int j = 0; j < MAX_CHANNELS; ++j) {
                        // #TODO GCC incompatibily:
                        // weights_span[i, j] = 0;
                        weights[i * MAX_CHANNELS + j] = 0;
                    }
                    biases[i] = 0;
                    activationFunctions[i] = ActivationFunction_Sigmoid;
                }
            }
            ImGui::SameLine();
            if (AlienGui::Button("Identity")) {
                for (int i = 0; i < MAX_CHANNELS; ++i) {
                    for (int j = 0; j < MAX_CHANNELS; ++j) {
                        // #TODO GCC incompatibily:
                        // weights_span[i, j] = i == j ? 1.0f : 0.0f;
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
                        // #TODO GCC incompatibily:
                        // weights_span[i, j] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                        weights[i * MAX_CHANNELS + j] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                    }
                    biases[i] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                    activationFunctions[i] = NumberGenerator::get().getRandomInt(ActivationFunction_Count);
                }
            }
        }
        ImGui::EndChild();
    }
    ImGui::EndChild();
}

_NeuralNetWidget::_NeuralNetWidget() {}
