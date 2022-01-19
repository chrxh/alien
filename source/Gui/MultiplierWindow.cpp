#include "MultiplierWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"
#include "Fonts/AlienIconFont.h"

#include "EngineInterface/SimulationController.h"
#include "AlienImGui.h"
#include "EditorModel.h"

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
    ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
    if (AlienImGui::BeginToolbarButton(ICON_GRID)) {
        _mode = MultiplierMode::Grid;
    }
    AlienImGui::EndToolbarButton();

    ImGui::SameLine();
    if (AlienImGui::BeginToolbarButton(ICON_RANDOM)) {
        _mode = MultiplierMode::Random;
    }
    AlienImGui::EndToolbarButton();

    AlienImGui::Group(ModeText.at(_mode));
    if (_mode == MultiplierMode::Grid) {
        processGridPanel();
    }
    ImGui::EndDisabled();
}

void _MultiplierWindow::processGridPanel()
{
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name(ICON_FA_ARROW_RIGHT " Number").textWidth(MaxContentTextWidth), _gridParameters._horizontalNumber);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Distance").textWidth(MaxContentTextWidth).format("%.1f"),
        _gridParameters._horizontalDistance);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Angle increment").textWidth(MaxContentTextWidth).format("%.1f"),
        _gridParameters._horizontalAngleInc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Velocity X increment").textWidth(MaxContentTextWidth).format("%.2f").step(0.1f),
        _gridParameters._horizontalVelXinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Velocity Y increment").textWidth(MaxContentTextWidth).format("%.2f").step(0.1f),
        _gridParameters._horizontalVelYinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_RIGHT " Angular velocity increment").textWidth(MaxContentTextWidth).format("%.1f").step(0.1f),
        _gridParameters._horizontalAngularVelInc);
    AlienImGui::Separator();
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name(ICON_FA_ARROW_DOWN " Number").textWidth(MaxContentTextWidth), _gridParameters._verticalNumber);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Distance").textWidth(MaxContentTextWidth).format("%.1f"),
        _gridParameters._verticalDistance);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Angle increment").textWidth(MaxContentTextWidth).format("%.1f"),
        _gridParameters._verticalAngleInc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Velocity X increment").textWidth(MaxContentTextWidth).format("%.2f").step(0.1f),
        _gridParameters._verticalVelXinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Velocity Y increment").textWidth(MaxContentTextWidth).format("%.2f").step(0.1f),
        _gridParameters._verticalVelYinc);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name(ICON_FA_ARROW_DOWN " Angular velocity increment").textWidth(MaxContentTextWidth).format("%.1f").step(0.1f),
        _gridParameters._verticalAngularVelInc);

    AlienImGui::Separator();
    ImGui::BeginDisabled(
        _selectionDataAfterMultiplication && _selectionDataAfterMultiplication->compareNumbers(_editorModel->getSelectionShallowData()));
    if (ImGui::Button("Build")) {
        _origSelection = _simController->getSelectedSimulationData(true);
        auto multiplicationResult = DescriptionHelper::gridMultiply(_origSelection, _gridParameters);
        _simController->removeSelectedEntities(true);
        _simController->addAndSelectSimulationData(multiplicationResult);

        _editorModel->update();
        _selectionDataAfterMultiplication = _editorModel->getSelectionShallowData();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!_selectionDataAfterMultiplication || !_selectionDataAfterMultiplication->compareNumbers(_editorModel->getSelectionShallowData()));
    if (ImGui::Button("Undo")) {
        _simController->removeSelectedEntities(true);
        _simController->addAndSelectSimulationData(_origSelection);
        _selectionDataAfterMultiplication = std::nullopt;
    }
    ImGui::EndDisabled();
}

