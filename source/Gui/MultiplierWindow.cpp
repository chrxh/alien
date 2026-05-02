#include "MultiplierWindow.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/GlobalSettings.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "EditorController.h"
#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "StyleRepository.h"

#include "Fonts/AlienIconFont.h"
#include <EngineInterface/SimulationFacade.h>

namespace
{
    auto const ModeText = std::unordered_map<MultiplierMode, std::string>{
        {MultiplierMode_Grid, "Grid multiplier"},
        {MultiplierMode_Random, "Random multiplier"},
    };

    auto const RightColumnWidth = 200.0f;
}

void MultiplierWindow::initIntern()
{
    _mode = GlobalSettings::get().getValue("editors.multiplier.mode", _mode);

    _gridParameters._horizontalNumber = GlobalSettings::get().getValue("editors.multiplier.grid.horizontal number", _gridParameters._horizontalNumber);
    _gridParameters._horizontalDistance = GlobalSettings::get().getValue("editors.multiplier.grid.horizontal distance", _gridParameters._horizontalDistance);
    _gridParameters._horizontalAngleInc = GlobalSettings::get().getValue("editors.multiplier.grid.horizontal angle inc", _gridParameters._horizontalAngleInc);
    _gridParameters._horizontalVelXinc = GlobalSettings::get().getValue("editors.multiplier.grid.horizontal vel x inc", _gridParameters._horizontalVelXinc);
    _gridParameters._horizontalVelYinc = GlobalSettings::get().getValue("editors.multiplier.grid.horizontal vel y inc", _gridParameters._horizontalVelYinc);
    _gridParameters._horizontalAngularVelInc =
        GlobalSettings::get().getValue("editors.multiplier.grid.horizontal angular vel inc", _gridParameters._horizontalAngularVelInc);
    _gridParameters._verticalNumber = GlobalSettings::get().getValue("editors.multiplier.grid.vertical number", _gridParameters._verticalNumber);
    _gridParameters._verticalDistance = GlobalSettings::get().getValue("editors.multiplier.grid.vertical distance", _gridParameters._verticalDistance);
    _gridParameters._verticalAngleInc = GlobalSettings::get().getValue("editors.multiplier.grid.vertical angle inc", _gridParameters._verticalAngleInc);
    _gridParameters._verticalVelXinc = GlobalSettings::get().getValue("editors.multiplier.grid.vertical vel x inc", _gridParameters._verticalVelXinc);
    _gridParameters._verticalVelYinc = GlobalSettings::get().getValue("editors.multiplier.grid.vertical vel y inc", _gridParameters._verticalVelYinc);
    _gridParameters._verticalAngularVelInc =
        GlobalSettings::get().getValue("editors.multiplier.grid.vertical angular vel inc", _gridParameters._verticalAngularVelInc);

    _randomParameters._number = GlobalSettings::get().getValue("editors.multiplier.random.number", _randomParameters._number);
    _randomParameters._minAngle = GlobalSettings::get().getValue("editors.multiplier.random.min angle", _randomParameters._minAngle);
    _randomParameters._maxAngle = GlobalSettings::get().getValue("editors.multiplier.random.max angle", _randomParameters._maxAngle);
    _randomParameters._minVelX = GlobalSettings::get().getValue("editors.multiplier.random.min vel x", _randomParameters._minVelX);
    _randomParameters._maxVelX = GlobalSettings::get().getValue("editors.multiplier.random.max vel x", _randomParameters._maxVelX);
    _randomParameters._minVelY = GlobalSettings::get().getValue("editors.multiplier.random.min vel y", _randomParameters._minVelY);
    _randomParameters._maxVelY = GlobalSettings::get().getValue("editors.multiplier.random.max vel y", _randomParameters._maxVelY);
    _randomParameters._minAngularVel = GlobalSettings::get().getValue("editors.multiplier.random.min angular vel", _randomParameters._minAngularVel);
    _randomParameters._maxAngularVel = GlobalSettings::get().getValue("editors.multiplier.random.max angular vel", _randomParameters._maxAngularVel);
    _randomParameters._overlappingCheck = GlobalSettings::get().getValue("editors.multiplier.random.overlapping check", _randomParameters._overlappingCheck);
}

void MultiplierWindow::shutdownIntern()
{
    GlobalSettings::get().setValue("editors.multiplier.mode", _mode);

    GlobalSettings::get().setValue("editors.multiplier.grid.horizontal number", _gridParameters._horizontalNumber);
    GlobalSettings::get().setValue("editors.multiplier.grid.horizontal distance", _gridParameters._horizontalDistance);
    GlobalSettings::get().setValue("editors.multiplier.grid.horizontal angle inc", _gridParameters._horizontalAngleInc);
    GlobalSettings::get().setValue("editors.multiplier.grid.horizontal vel x inc", _gridParameters._horizontalVelXinc);
    GlobalSettings::get().setValue("editors.multiplier.grid.horizontal vel y inc", _gridParameters._horizontalVelYinc);
    GlobalSettings::get().setValue("editors.multiplier.grid.horizontal angular vel inc", _gridParameters._horizontalAngularVelInc);
    GlobalSettings::get().setValue("editors.multiplier.grid.vertical number", _gridParameters._verticalNumber);
    GlobalSettings::get().setValue("editors.multiplier.grid.vertical distance", _gridParameters._verticalDistance);
    GlobalSettings::get().setValue("editors.multiplier.grid.vertical angle inc", _gridParameters._verticalAngleInc);
    GlobalSettings::get().setValue("editors.multiplier.grid.vertical vel x inc", _gridParameters._verticalVelXinc);
    GlobalSettings::get().setValue("editors.multiplier.grid.vertical vel y inc", _gridParameters._verticalVelYinc);
    GlobalSettings::get().setValue("editors.multiplier.grid.vertical angular vel inc", _gridParameters._verticalAngularVelInc);

    GlobalSettings::get().setValue("editors.multiplier.random.number", _randomParameters._number);
    GlobalSettings::get().setValue("editors.multiplier.random.min angle", _randomParameters._minAngle);
    GlobalSettings::get().setValue("editors.multiplier.random.max angle", _randomParameters._maxAngle);
    GlobalSettings::get().setValue("editors.multiplier.random.min vel x", _randomParameters._minVelX);
    GlobalSettings::get().setValue("editors.multiplier.random.max vel x", _randomParameters._maxVelX);
    GlobalSettings::get().setValue("editors.multiplier.random.min vel y", _randomParameters._minVelY);
    GlobalSettings::get().setValue("editors.multiplier.random.max vel y", _randomParameters._maxVelY);
    GlobalSettings::get().setValue("editors.multiplier.random.min angular vel", _randomParameters._minAngularVel);
    GlobalSettings::get().setValue("editors.multiplier.random.max angular vel", _randomParameters._maxAngularVel);
    GlobalSettings::get().setValue("editors.multiplier.random.overlapping check", _randomParameters._overlappingCheck);
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

        AlienGui::Group(AlienGui::GroupParameters().text(ModeText.at(_mode)));
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
        || (_selectionDataAfterMultiplication && _selectionDataAfterMultiplication->compareSizes(EditorModel::get().getSelectionShallowData())));
    if (AlienGui::Button("Build")) {
        onBuild();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(
        EditorModel::get().isSelectionEmpty() || !_selectionDataAfterMultiplication
        || !_selectionDataAfterMultiplication->compareSizes(EditorModel::get().getSelectionShallowData()));
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
    AlienGui::InputInt(
        AlienGui::InputIntParameters().name(ICON_FA_ARROW_RIGHT " Number of copies").textWidth(RightColumnWidth), _gridParameters._horizontalNumber);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Distance").textWidth(RightColumnWidth).format("%.1f"), _gridParameters._horizontalDistance);
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
    AlienGui::InputInt(
        AlienGui::InputIntParameters().name(ICON_FA_ARROW_DOWN " Number of copies").textWidth(RightColumnWidth), _gridParameters._verticalNumber);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Distance").textWidth(RightColumnWidth).format("%.1f"), _gridParameters._verticalDistance);
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
    AlienGui::InputInt(AlienGui::InputIntParameters().name("Number of copies").textWidth(RightColumnWidth), _randomParameters._number);
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
        AlienGui::InputFloatParameters().name("Min angular velocity").textWidth(RightColumnWidth).format("%.1f").step(0.1f), _randomParameters._minAngularVel);
    AlienGui::InputFloat(
        AlienGui::InputFloatParameters().name("Max angular velocity").textWidth(RightColumnWidth).format("%.1f").step(0.1f), _randomParameters._maxAngularVel);
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
    _origSelection = _SimulationFacade::get()->getSelectedSimulationData(true);
    auto multiplicationResult = [&] {
        if (_mode == MultiplierMode_Grid) {
            return DescEditService::get().gridMultiply(_origSelection, _gridParameters);
        } else {
            auto data = _SimulationFacade::get()->getSimulationData();
            auto overlappingCheckSuccessful = true;
            auto result = DescEditService::get().randomMultiply(
                _origSelection, _randomParameters, _SimulationFacade::get()->getWorldSize(), std::move(data), overlappingCheckSuccessful);
            if (!overlappingCheckSuccessful) {
                GenericMessageDialog::get().information("Random multiplication", "Non-overlapping copies could not be created.");
            }
            return result;
        }
    }();
    _SimulationFacade::get()->removeSelectedObjects(true);
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(multiplicationResult));

    EditorModel::get().update();
    _selectionDataAfterMultiplication = EditorModel::get().getSelectionShallowData();
}

void MultiplierWindow::onUndo()
{
    _SimulationFacade::get()->removeSelectedObjects(true);
    _SimulationFacade::get()->addAndSelectSimulationData(Desc(_origSelection));
    _selectionDataAfterMultiplication = std::nullopt;
}
