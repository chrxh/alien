#include "ManipulatorWindow.h"

#include <imgui.h>

#include "IconFontCppHeaders/IconsFontAwesome5.h"

#include "EngineInterface/Colors.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineImpl/SimulationController.h"

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
    : _editorModel(editorModel)
    , _simController(simController)
    , _viewport(viewport)
{
    _on = GlobalSettings::getInstance().getBoolState("editor.manipulator.active", true);
    _saveSelectionDialog = boost::make_shared<_SaveSelectionDialog>(simController);
    _openSelectionDialog = boost::make_shared<_OpenSelectionDialog>(editorModel, simController, viewport);
}

_ManipulatorWindow::~_ManipulatorWindow()
{
    GlobalSettings::getInstance().setBoolState("editor.manipulator.active", _on);
}

void _ManipulatorWindow::process()
{
    if (!_on) {
        return;
    }
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin("Manipulator", &_on)) {
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
            if (AlienImGui::BeginToolbarButton(ICON_FA_FOLDER_OPEN)) {
                _openSelectionDialog->show();
            }
            AlienImGui::EndToolbarButton();

            //save button
            ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
            ImGui::SameLine();
            if (AlienImGui::BeginToolbarButton(ICON_FA_SAVE)) {
                _saveSelectionDialog->show(_includeClusters);
            }
            AlienImGui::EndToolbarButton();
            ImGui::EndDisabled();

            //copy button
            ImGui::SameLine();
            ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
            if (AlienImGui::BeginToolbarButton(ICON_FA_COPY)) {
                _copiedSelection = _simController->getSelectedSimulationData(_includeClusters);
            }
            AlienImGui::EndToolbarButton();
            ImGui::EndDisabled();

            //paste button
            ImGui::SameLine();
            ImGui::BeginDisabled(!_copiedSelection);
            if (AlienImGui::BeginToolbarButton(ICON_FA_PASTE)) {
                auto data = *_copiedSelection;
                auto center = _viewport->getCenterInWorldPos();
                data.setCenter(center);
                _simController->addAndSelectSimulationData(data);
                _editorModel->update();
            }
            AlienImGui::EndToolbarButton();
            ImGui::EndDisabled();

            //delete button
            ImGui::SameLine();
            ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
            if (AlienImGui::BeginToolbarButton(ICON_FA_TRASH)) {
                _simController->removeSelectedEntities(_includeClusters);
                _editorModel->update();
            }
            AlienImGui::EndToolbarButton();
            ImGui::EndDisabled();

            //inspector button
            ImGui::SameLine();
            ImGui::BeginDisabled(!isInspectionPossible());
            if (AlienImGui::BeginToolbarButton(ICON_FA_MICROSCOPE)) {
                onInspectEntities();
            }
            AlienImGui::EndToolbarButton();
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
            }
            ImGui::SameLine();
            if (colorButton("    ##color2", Const::IndividualCellColor2)) {
                _simController->colorSelectedEntities(1, _includeClusters);
            }
            ImGui::SameLine();
            if (colorButton("    ##color3", Const::IndividualCellColor3)) {
                _simController->colorSelectedEntities(2, _includeClusters);
            }
            ImGui::SameLine();
            if (colorButton("    ##color4", Const::IndividualCellColor4)) {
                _simController->colorSelectedEntities(3, _includeClusters);
            }
            ImGui::SameLine();
            if (colorButton("    ##color5", Const::IndividualCellColor5)) {
                _simController->colorSelectedEntities(4, _includeClusters);
            }
            ImGui::SameLine();
            if (colorButton("    ##color6", Const::IndividualCellColor6)) {
                _simController->colorSelectedEntities(5, _includeClusters);
            }
            ImGui::SameLine();
            if (colorButton("    ##color7", Const::IndividualCellColor7)) {
                _simController->colorSelectedEntities(6, _includeClusters);
            }

            ImGui::EndDisabled();

            _lastSelection = selection;
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();

        AlienImGui::Separator();
        if (ImGui::Checkbox("Roll out to cell clusters", &_includeClusters)) {
            _angle = 0;
            _angularVel = 0;
        }
    }
    ImGui::End();

    _saveSelectionDialog->process();
    _openSelectionDialog->process();
}

bool _ManipulatorWindow::isOn() const
{
    return _on;
}

void _ManipulatorWindow::setOn(bool value)
{
    _on = value;
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
