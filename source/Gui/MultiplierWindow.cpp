#include "MultiplierWindow.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>
#include "Fonts/AlienIconFont.h"

#include "EngineInterface/SimulationFacade.h"
#include "AlienGui.h"
#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "StyleRepository.h"
#include "EditorController.h"

namespace
{
    auto const ModeText = std::unordered_map<MultiplierMode, std::string>{
        {MultiplierMode_Grid, "Grid multiplier"},
        {MultiplierMode_Random, "Random multiplier"},
    };

    auto const RightColumnWidth = 200.0f;
}

void MultiplierWindow::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

MultiplierWindow::MultiplierWindow()
    : AlienWindow("Multiplier", "editors.multiplier", false)
{}

void MultiplierWindow::processIntern()
{
    AlienGui::SelectableToolbarButton(ICON_GRID, _mode, MultiplierMode_Grid, MultiplierMode_Grid);

    ImGui::SameLine();
    AlienGui::SelectableToolbarButton(ICON_RANDOM, _mode, MultiplierMode_Random, MultiplierMode_Random);

    if (ImGui::BeginChild("##", ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50.0f)), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        ImGui::BeginDisabled(EditorModel::get().isSelectionEmpty());

        AlienGui::Group(ModeText.at(_mode));
        if (_mode == MultiplierMode_Grid) {
            processGridPanel();
        }
        if (_mode == MultiplierMode_Random) {
            processRandomPanel();
        }
        ImGui::EndDisabled();
    }
    ImGui::EndChild();

        AlienGui::Separator();
    ImGui::BeginDisabled(
        EditorModel::get().isSelectionEmpty()
        || (_selectionDataAfterMultiplication && _selectionDataAfterMultiplication->compareNumbers(EditorModel::get().getSelectionShallowData())));
    if (AlienGui::Button("Build")) {
        onBuild();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(
        EditorModel::get().isSelectionEmpty() || !_selectionDataAfterMultiplication
        || !_selectionDataAfterMultiplication->compareNumbers(EditorModel::get().getSelectionShallowData()));
    if (AlienGui::Button("Undo")) {
        onUndo();
    }
    ImGui::EndDisabled();

    validateAndCorrect();
}

bool MultiplierWindow::isShown()
{
    return _on && EditorController::get().isOn();
}

void MultiplierWindow::processGridPanel()
{
    AlienGui::InputInt(AlienGui::InputIntParameters().name(ICON_FA_ARROW_RIGHT " Number of copies").textWidth(RightColumnWidth), _gridParameters._horizontalNumber);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Distance").textWidth(RightColumnWidth).format("%.1f"),
        _gridParameters._horizontalDistance);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Angle increment").textWidth(RightColumnWidth).format("%.1f"),
        _gridParameters._horizontalAngleInc);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Velocity X increment").textWidth(RightColumnWidth).format("%.2f").step(0.05f),
        _gridParameters._horizontalVelXinc);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Velocity Y increment").textWidth(RightColumnWidth).format("%.2f").step(0.05f),
        _gridParameters._horizontalVelYinc);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Angular velocity increment").textWidth(RightColumnWidth).format("%.1f").step(0.1f),
        _gridParameters._horizontalAngularVelInc);
    AlienGui::Separator();
    AlienGui::InputInt(AlienGui::InputIntParameters().name(ICON_FA_ARROW_DOWN " Number of copies").textWidth(RightColumnWidth), _gridParameters._verticalNumber);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Distance").textWidth(RightColumnWidth).format("%.1f"),
        _gridParameters._verticalDistance);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Angle increment").textWidth(RightColumnWidth).format("%.1f"),
        _gridParameters._verticalAngleInc);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Velocity X increment").textWidth(RightColumnWidth).format("%.2f").step(0.05f),
        _gridParameters._verticalVelXinc);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Velocity Y increment").textWidth(RightColumnWidth).format("%.2f").step(0.05f),
        _gridParameters._verticalVelYinc);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Angular velocity increment").textWidth(RightColumnWidth).format("%.1f").step(0.1f),
        _gridParameters._verticalAngularVelInc);
}

void MultiplierWindow::processRandomPanel()
{
    AlienGui::InputInt(
        AlienGui::InputIntParameters().name("Number of copies").textWidth(RightColumnWidth), _randomParameters._number);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Min angle").textWidth(RightColumnWidth).format("%.1f"), _randomParameters._minAngle);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Max angle").textWidth(RightColumnWidth).format("%.1f"), _randomParameters._maxAngle);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Min velocity X").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._minVelX);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Max velocity X").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._maxVelX);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Min velocity Y").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._minVelY);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Max velocity Y").textWidth(RightColumnWidth).format("%.2f").step(0.05f), _randomParameters._maxVelY);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Min angular velocity").textWidth(RightColumnWidth).format("%.1f").step(0.1f),
        _randomParameters._minAngularVel);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Max angular velocity").textWidth(RightColumnWidth).format("%.1f").step(0.1f),
        _randomParameters._maxAngularVel);
    AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Overlapping check").textWidth(RightColumnWidth), _randomParameters._overlappingCheck);
}

void MultiplierWindow::validateAndCorrect()
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

void MultiplierWindow::onBuild()
{
    _origSelection = _simulationFacade->getSelectedSimulationData(true);
    auto multiplicationResult = [&] {
        if (_mode == MultiplierMode_Grid) {
            return DescriptionEditService::get().gridMultiply(_origSelection, _gridParameters);
        } else {
            auto data = _simulationFacade->getSimulationData();
            auto overlappingCheckSuccessful = true;
            auto result = DescriptionEditService::get().randomMultiply(
                _origSelection, _randomParameters, _simulationFacade->getWorldSize(), std::move(data), overlappingCheckSuccessful);
            if (!overlappingCheckSuccessful) {
                GenericMessageDialog::get().information("Random multiplication", "Non-overlapping copies could not be created.");
            }
            return result;
        }
    }();
    _simulationFacade->removeSelectedObjects(true);
    _simulationFacade->addAndSelectSimulationData(std::move(multiplicationResult));

    EditorModel::get().update();
    _selectionDataAfterMultiplication = EditorModel::get().getSelectionShallowData();
}

void MultiplierWindow::onUndo()
{
    _simulationFacade->removeSelectedObjects(true);
    _simulationFacade->addAndSelectSimulationData(CollectionDescription(_origSelection));
    _selectionDataAfterMultiplication = std::nullopt;
}

