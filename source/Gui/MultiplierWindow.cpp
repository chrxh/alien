#include "MultiplierWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"
#include "Fonts/AlienIconFont.h"

#include "EngineInterface/SimulationController.h"
#include "AlienImGui.h"
#include "EditorModel.h"
#include "MessageDialog.h"
#include "StyleRepository.h"

namespace
{
    auto const ModeText = std::unordered_map<MultiplierMode, std::string>{
        {MultiplierMode::Grid, "Grid multiplier"},
        {MultiplierMode::Random, "Random multiplier"},
    };

    auto const RightColumnWidth = 200.0f;
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

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        ImGui::BeginDisabled(_editorModel->isSelectionEmpty());

        AlienImGui::Group(ModeText.at(_mode));
        if (_mode == MultiplierMode::Grid) {
            processGridPanel();
        }
        if (_mode == MultiplierMode::Random) {
            processRandomPanel();
        }
        ImGui::EndDisabled();
    }
    ImGui::EndChild();

        AlienImGui::Separator();
    ImGui::BeginDisabled(
        _editorModel->isSelectionEmpty()
        || (_selectionDataAfterMultiplication && _selectionDataAfterMultiplication->compareNumbers(_editorModel->getSelectionShallowData())));
    if (AlienImGui::Button("Build")) {
        onBuild();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(
        _editorModel->isSelectionEmpty() || !_selectionDataAfterMultiplication
        || !_selectionDataAfterMultiplication->compareNumbers(_editorModel->getSelectionShallowData()));
    if (AlienImGui::Button("Undo")) {
        onUndo();
    }
    ImGui::EndDisabled();

    validationAndCorrection();
}

void _MultiplierWindow::processGridPanel()
{
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name(ICON_FA_ARROW_RIGHT " Number of copies").textWidth(RightColumnWidth), _gridParameters._horizontalNumber);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Distance").textWidth(RightColumnWidth).format("%.1f"),
        _gridParameters._horizontalDistance);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Angle increment").textWidth(RightColumnWidth).format("%.1f"),
        _gridParameters._horizontalAngleInc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Velocity X increment").textWidth(RightColumnWidth).format("%.2f").step(0.05f),
        _gridParameters._horizontalVelXinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Velocity Y increment").textWidth(RightColumnWidth).format("%.2f").step(0.05f),
        _gridParameters._horizontalVelYinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Angular velocity increment").textWidth(RightColumnWidth).format("%.1f").step(0.1f),
        _gridParameters._horizontalAngularVelInc);
    AlienImGui::Separator();
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name(ICON_FA_ARROW_DOWN " Number of copies").textWidth(RightColumnWidth), _gridParameters._verticalNumber);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Distance").textWidth(RightColumnWidth).format("%.1f"),
        _gridParameters._verticalDistance);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Angle increment").textWidth(RightColumnWidth).format("%.1f"),
        _gridParameters._verticalAngleInc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Velocity X increment").textWidth(RightColumnWidth).format("%.2f").step(0.05f),
        _gridParameters._verticalVelXinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Velocity Y increment").textWidth(RightColumnWidth).format("%.2f").step(0.05f),
        _gridParameters._verticalVelYinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Angular velocity increment").textWidth(RightColumnWidth).format("%.1f").step(0.1f),
        _gridParameters._verticalAngularVelInc);
}

void _MultiplierWindow::processRandomPanel()
{
    AlienImGui::InputInt(
        AlienImGui::InputIntParameters().name("Number of copies").textWidth(RightColumnWidth), _randomParameters._number);
    AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Min angle").textWidth(RightColumnWidth).format("%.1f"), _randomParameters._minAngle);
    AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Max angle").textWidth(RightColumnWidth).format("%.1f"), _randomParameters._maxAngle);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Min velocity X").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._minVelX);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Max velocity X").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._maxVelX);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Min velocity Y").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._minVelY);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Max velocity Y").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._maxVelY);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Min angular velocity").textWidth(RightColumnWidth).format("%.1f").step(0.1f),
        _randomParameters._minAngularVel);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Max angular velocity").textWidth(RightColumnWidth).format("%.1f").step(0.1f),
        _randomParameters._maxAngularVel);
    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Overlapping check").textWidth(RightColumnWidth), _randomParameters._overlappingCheck);
}

void _MultiplierWindow::validationAndCorrection()
{
    _gridParameters._horizontalNumber = std::max(1, _gridParameters._horizontalNumber);
    _gridParameters._horizontalDistance = std::max(0.0f, _gridParameters._horizontalDistance);
    _gridParameters._verticalNumber = std::max(1, _gridParameters._verticalNumber);
    _gridParameters._verticalDistance = std::max(0.0f, _gridParameters._verticalDistance);

    _randomParameters._number = std::max(1, _randomParameters._number);
    _randomParameters._maxAngle = std::max(_randomParameters._minAngle, _randomParameters._maxAngle);
    _randomParameters._maxVelX = std::max(_randomParameters._minVelX, _randomParameters._maxVelX);
    _randomParameters._maxVelY = std::max(_randomParameters._minVelY, _randomParameters._maxVelY);
    _randomParameters._maxAngularVel = std::max(_randomParameters._minAngularVel, _randomParameters._maxAngularVel);
}

void _MultiplierWindow::onBuild()
{
    _origSelection = _simController->getSelectedSimulationData(true);
    auto multiplicationResult = [&] {
        if (_mode == MultiplierMode::Grid) {
            return DescriptionHelper::gridMultiply(_origSelection, _gridParameters);
        } else {
            auto data = _simController->getSimulationData();
            auto overlappingCheckSuccessful = true;
            auto result = DescriptionHelper::randomMultiply(
                _origSelection, _randomParameters, _simController->getWorldSize(), std::move(data), overlappingCheckSuccessful);
            if (!overlappingCheckSuccessful) {
                MessageDialog::getInstance().show("Random multiplication", "Non-overlapping copies could not be created.");
            }
            return result;
        }
    }();
    _simController->removeSelectedObjects(true);
    _simController->addAndSelectSimulationData(multiplicationResult);

    _editorModel->update();
    _selectionDataAfterMultiplication = _editorModel->getSelectionShallowData();
}

void _MultiplierWindow::onUndo()
{
    _simController->removeSelectedObjects(true);
    _simController->addAndSelectSimulationData(_origSelection);
    _selectionDataAfterMultiplication = std::nullopt;
}

