#include "MultiplierWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"
#include "Fonts/AlienIconFont.h"

#include "EngineInterface/SimulationController.h"
#include "AlienImGui.h"
#include "EditorModel.h"
#include "MessageDialog.h"

namespace
{
    auto const ModeText = std::unordered_map<MultiplierMode, std::string>{
        {MultiplierMode::Grid, "Grid multiplier"},
        {MultiplierMode::Random, "Random multiplier"},
    };

    auto const MaxContentTextWidth = 200.0f;
}

_MultiplierWindow::_MultiplierWindow(EditorModel const& editorModel, SimulationController const& simController, Viewport const& viewport)
    : _AlienWindow("Multiplier", "editor.multiplier", false)
    , _editorModel(editorModel)
    , _simController(simController)
    , _viewport(viewport)
{}

void _MultiplierWindow::processIntern()
{
    if (AlienImGui::ToolbarButton(ICON_GRID)) {
        _mode = MultiplierMode::Grid;
    }

    ImGui::SameLine();
    if (AlienImGui::ToolbarButton(ICON_RANDOM)) {
        _mode = MultiplierMode::Random;
    }

    if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        ImGui::BeginDisabled(_editorModel->isSelectionEmpty());

        AlienImGui::Group(ModeText.at(_mode));
        if (_mode == MultiplierMode::Grid) {
            processGridPanel();
        }
        if (_mode == MultiplierMode::Random) {
            processRandomPanel();
        }
        ImGui::EndDisabled();

        AlienImGui::Separator();
        ImGui::BeginDisabled(
            _editorModel->isSelectionEmpty()
            || (_selectionDataAfterMultiplication && _selectionDataAfterMultiplication->compareNumbers(_editorModel->getSelectionShallowData())));
        if (AlienImGui::Button("Build")) {
            _origSelection = _simController->getSelectedSimulationData(true);
            auto multiplicationResult = [&] {
                if (_mode == MultiplierMode::Grid) {
                    return DescriptionHelper::gridMultiply(_origSelection, _gridParameters);
                }
                auto data = _simController->getSimulationData();
                auto overlappingCheckSuccessful = true;
                auto result = DescriptionHelper::randomMultiply(
                    _origSelection, _randomParameters, _simController->getWorldSize(), std::move(data), overlappingCheckSuccessful);
                if (!overlappingCheckSuccessful) {
                    MessageDialog::getInstance().show("Random multiplication", "Non-overlapping copies could not be created.");
                }
                return result;
            }();
            _simController->removeSelectedEntities(true);
            _simController->addAndSelectSimulationData(multiplicationResult);

            _editorModel->update();
            _selectionDataAfterMultiplication = _editorModel->getSelectionShallowData();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::BeginDisabled(
            _editorModel->isSelectionEmpty() || !_selectionDataAfterMultiplication
            || !_selectionDataAfterMultiplication->compareNumbers(_editorModel->getSelectionShallowData()));
        if (AlienImGui::Button("Undo")) {
            _simController->removeSelectedEntities(true);
            _simController->addAndSelectSimulationData(_origSelection);
            _selectionDataAfterMultiplication = std::nullopt;
        }
        ImGui::EndDisabled();
    }
    ImGui::EndChild();
}

void _MultiplierWindow::processGridPanel()
{
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name(ICON_FA_ARROW_RIGHT " Number of copies").textWidth(MaxContentTextWidth), _gridParameters._horizontalNumber);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Distance").textWidth(MaxContentTextWidth).format("%.1f"),
        _gridParameters._horizontalDistance);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Angle increment").textWidth(MaxContentTextWidth).format("%.1f"),
        _gridParameters._horizontalAngleInc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Velocity X increment").textWidth(MaxContentTextWidth).format("%.2f").step(0.05f),
        _gridParameters._horizontalVelXinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Velocity Y increment").textWidth(MaxContentTextWidth).format("%.2f").step(0.05f),
        _gridParameters._horizontalVelYinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Angular velocity increment").textWidth(MaxContentTextWidth).format("%.1f").step(0.1f),
        _gridParameters._horizontalAngularVelInc);
    AlienImGui::Separator();
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name(ICON_FA_ARROW_DOWN " Number of copies").textWidth(MaxContentTextWidth), _gridParameters._verticalNumber);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Distance").textWidth(MaxContentTextWidth).format("%.1f"),
        _gridParameters._verticalDistance);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Angle increment").textWidth(MaxContentTextWidth).format("%.1f"),
        _gridParameters._verticalAngleInc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Velocity X increment").textWidth(MaxContentTextWidth).format("%.2f").step(0.05f),
        _gridParameters._verticalVelXinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Velocity Y increment").textWidth(MaxContentTextWidth).format("%.2f").step(0.05f),
        _gridParameters._verticalVelYinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Angular velocity increment").textWidth(MaxContentTextWidth).format("%.1f").step(0.1f),
        _gridParameters._verticalAngularVelInc);
}

void _MultiplierWindow::processRandomPanel()
{
    AlienImGui::InputInt(
        AlienImGui::InputIntParameters().name("Number of copies").textWidth(MaxContentTextWidth), _randomParameters._number);
    AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Min angle").textWidth(MaxContentTextWidth).format("%.1f"), _randomParameters._minAngle);
    AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Max angle").textWidth(MaxContentTextWidth).format("%.1f"), _randomParameters._maxAngle);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Min velocity X").textWidth(MaxContentTextWidth).format("%.2f").step(0.05f), _randomParameters._minVelX);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Max velocity X").textWidth(MaxContentTextWidth).format("%.2f").step(0.05f), _randomParameters._maxVelX);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Min velocity Y").textWidth(MaxContentTextWidth).format("%.2f").step(0.05f), _randomParameters._minVelY);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Max velocity Y").textWidth(MaxContentTextWidth).format("%.2f").step(0.05f), _randomParameters._maxVelY);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Min angular velocity").textWidth(MaxContentTextWidth).format("%.1f").step(0.1f),
        _randomParameters._minAngularVel);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Max angular velocity").textWidth(MaxContentTextWidth).format("%.1f").step(0.1f),
        _randomParameters._maxAngularVel);
    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Overlapping check").textWidth(MaxContentTextWidth), _randomParameters._overlappingCheck);
}

