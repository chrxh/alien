#include "ManipulatorWindow.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "EngineInterface/Colors.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"

#include "EditorModel.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"
#include "Viewport.h"
#include "SaveSelectionDialog.h"
#include "OpenSelectionDialog.h"

namespace
{
    auto const MaxInspectorWindowsToAdd = 10;
    auto const MaxContentTextWidth = 120.0f;
}

_ManipulatorWindow::_ManipulatorWindow(
    EditorModel const& editorModel,
    SimulationController const& simController,
    Viewport const& viewport)
    : _AlienWindow("Manipulator", "editor.manipulator", true)
    , _editorModel(editorModel)
    , _simController(simController)
    , _viewport(viewport)
{
    _saveSelectionDialog = std::make_shared<_SaveSelectionDialog>(simController);
    _openSelectionDialog = std::make_shared<_OpenSelectionDialog>(editorModel, simController, viewport);
}

void _ManipulatorWindow::processIntern()
{
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0, 0});
    if (ImGui::BeginChild(
            "##",
            ImVec2(0, ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scaleContent(50.0f)),
            false,
            ImGuiWindowFlags_HorizontalScrollbar)) {

        auto selection = _editorModel->getSelectionShallowData();
        if (hasSelectionChanged(selection)) {
            _angle = 0;
            _angularVel = 0;
        }

        //load button
        if (AlienImGui::ToolbarButton(ICON_FA_FOLDER_OPEN)) {
            _openSelectionDialog->show();
        }

        //save button
        ImGui::BeginDisabled(!isCopyingPossible());
        ImGui::SameLine();
        if (AlienImGui::ToolbarButton(ICON_FA_SAVE)) {
            _saveSelectionDialog->show(_includeClusters);
        }
        ImGui::EndDisabled();

        //copy button
        ImGui::SameLine();
        ImGui::BeginDisabled(!isCopyingPossible());
        if (AlienImGui::ToolbarButton(ICON_FA_COPY)) {
            onCopy();
        }
        ImGui::EndDisabled();

        //paste button
        ImGui::SameLine();
        ImGui::BeginDisabled(!isPastingPossible());
        if (AlienImGui::ToolbarButton(ICON_FA_PASTE)) {
            onPaste();
        }
        ImGui::EndDisabled();

        //delete button
        ImGui::SameLine();
        ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
        if (AlienImGui::ToolbarButton(ICON_FA_TRASH)) {
            _simController->removeSelectedEntities(_includeClusters);
            _editorModel->update();
        }
        ImGui::EndDisabled();

        //inspector button
        ImGui::SameLine();
        ImGui::BeginDisabled(!isInspectionPossible());
        if (AlienImGui::ToolbarButton(ICON_FA_MICROSCOPE)) {
            onInspectEntities();
        }
        ImGui::EndDisabled();

        ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
        AlienImGui::Group("Center position and velocity");

        auto const& selectionData = _editorModel->getSelectionShallowData();

        auto centerPosX = _includeClusters ? selectionData.clusterCenterPosX : selectionData.centerPosX;
        auto origCenterPosX = centerPosX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Position X")
                .textWidth(MaxContentTextWidth)
                .format("%.2f"),
            centerPosX);

        auto centerPosY = _includeClusters ? selectionData.clusterCenterPosY : selectionData.centerPosY;
        auto origCenterPosY = centerPosY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Position Y")
                .textWidth(MaxContentTextWidth)
                .format("%.2f"),
            centerPosY);

        auto centerVelX = _includeClusters ? selectionData.clusterCenterVelX : selectionData.centerVelX;
        auto origCenterVelX = centerVelX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Velocity X")
                .textWidth(MaxContentTextWidth)
                .step(0.1f)
                .format("%.2f"),
            centerVelX);

        auto centerVelY = _includeClusters ? selectionData.clusterCenterVelY : selectionData.centerVelY;
        auto origCenterVelY = centerVelY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Velocity Y")
                .textWidth(MaxContentTextWidth)
                .step(0.1f)
                .format("%.2f"),
            centerVelY);

        AlienImGui::Group("Center rotation");
        auto origAngle = _angle;
        AlienImGui::SliderInputFloat(
            AlienImGui::SliderInputFloatParameters()
                .name("Angle")
                .textWidth(MaxContentTextWidth)
                .inputWidth(StyleRepository::getInstance().scaleContent(50))
                .min(-180.0f)
                .max(180.0f)
                .format("%.1f"),
            _angle);

        auto origAngularVel = _angularVel;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Angular velocity")
                .textWidth(MaxContentTextWidth)
                .step(0.01f)
                .format("%.2f"),
            _angularVel);

        if (centerPosX != origCenterPosX || centerPosY != origCenterPosY) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _includeClusters;
            updateData.posDeltaX = centerPosX - origCenterPosX;
            updateData.posDeltaY = centerPosY - origCenterPosY;
            _simController->shallowUpdateSelectedEntities(updateData);
            _editorModel->update();
        }

        if (centerVelX != origCenterVelX || centerVelY != origCenterVelY) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _includeClusters;
            updateData.velDeltaX = centerVelX - origCenterVelX;
            updateData.velDeltaY = centerVelY - origCenterVelY;
            _simController->shallowUpdateSelectedEntities(updateData);
            _editorModel->update();
        }

        if (_angle != origAngle) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _includeClusters;
            updateData.angleDelta = _angle - origAngle;
            _simController->shallowUpdateSelectedEntities(updateData);
            _editorModel->update();
        }

        if (_angularVel != origAngularVel) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _includeClusters;
            updateData.angularVelDelta = _angularVel - origAngularVel;
            _simController->shallowUpdateSelectedEntities(updateData);
            _editorModel->update();
        }

        AlienImGui::Group("Color");
        if (colorButton("    ##color1", Const::IndividualCellColor1)) {
            _simController->colorSelectedEntities(0, _includeClusters);
            _editorModel->setDefaultColorCode(0);
        }
        ImGui::SameLine();
        if (colorButton("    ##color2", Const::IndividualCellColor2)) {
            _simController->colorSelectedEntities(1, _includeClusters);
            _editorModel->setDefaultColorCode(1);
        }
        ImGui::SameLine();
        if (colorButton("    ##color3", Const::IndividualCellColor3)) {
            _simController->colorSelectedEntities(2, _includeClusters);
            _editorModel->setDefaultColorCode(2);
        }
        ImGui::SameLine();
        if (colorButton("    ##color4", Const::IndividualCellColor4)) {
            _simController->colorSelectedEntities(3, _includeClusters);
            _editorModel->setDefaultColorCode(3);
        }
        ImGui::SameLine();
        if (colorButton("    ##color5", Const::IndividualCellColor5)) {
            _simController->colorSelectedEntities(4, _includeClusters);
            _editorModel->setDefaultColorCode(4);
        }
        ImGui::SameLine();
        if (colorButton("    ##color6", Const::IndividualCellColor6)) {
            _simController->colorSelectedEntities(5, _includeClusters);
            _editorModel->setDefaultColorCode(5);
        }
        ImGui::SameLine();
        if (colorButton("    ##color7", Const::IndividualCellColor7)) {
            _simController->colorSelectedEntities(6, _includeClusters);
            _editorModel->setDefaultColorCode(7);
        }

        ImGui::EndDisabled();

        _lastSelection = selection;
    }
    ImGui::EndChild();
    ImGui::PopStyleVar();

    AlienImGui::Separator();
    if (AlienImGui::ToggleButton("Roll out to cell clusters", _includeClusters)) {
        _angle = 0;
        _angularVel = 0;
    }

    _saveSelectionDialog->process();
    _openSelectionDialog->process();
}

bool _ManipulatorWindow::isInspectionPossible() const
{
    auto selection = _editorModel->getSelectionShallowData();
    return !_editorModel->isSelectionEmpty() && selection.numCells + selection.numParticles <= MaxInspectorWindowsToAdd;
}

void _ManipulatorWindow::onInspectEntities()
{
    DataDescription selectedData = _simController->getSelectedSimulationData(false);
    _editorModel->inspectEntities(DescriptionHelper::getEntities(selectedData));
}

bool _ManipulatorWindow::isCopyingPossible() const
{
    return !_editorModel->isSelectionEmpty();
}

void _ManipulatorWindow::onCopy()
{
    _copiedSelection = _simController->getSelectedSimulationData(_includeClusters);
}

bool _ManipulatorWindow::isPastingPossible() const
{
    return _copiedSelection.has_value();
}

void _ManipulatorWindow::onPaste()
{
    auto data = *_copiedSelection;
    auto center = _viewport->getCenterInWorldPos();
    data.setCenter(center);
    _simController->addAndSelectSimulationData(data);
    _editorModel->update();
}

bool _ManipulatorWindow::colorButton(std::string id, uint32_t cellColor)
{
    float h, s, v;
    AlienImGui::convertRGBtoHSV(cellColor, h, s,v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s * 0.6f, v * 0.6f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.7f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(h, s * 0.8f, v * 1.0f));
    auto result = ImGui::Button(id.c_str());
    ImGui::PopStyleColor(3);

    return result;
}

bool _ManipulatorWindow::hasSelectionChanged(SelectionShallowData const& selection) const
{
    if(!_lastSelection) {
        return false;
    }
    return _lastSelection->numCells != selection.numCells || _lastSelection->numParticles != selection.numParticles;
}
