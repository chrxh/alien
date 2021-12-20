#include "ManipulatorWindow.h"

#include <imgui.h>

#include "EngineInterface/Colors.h"
#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineImpl/SimulationController.h"

#include "EditorModel.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"
#include "Viewport.h"

namespace
{
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
    auto maxContentTextWidthScaled = StyleRepository::getInstance().scaleContent(MaxContentTextWidth);
    
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin("Manipulator", &_on)) {
        auto selection = _editorModel->getSelectionShallowData();
        if (hasSelectionChanged(selection)) {
            _angle = 0;
            _angularVel = 0;
        }
        if (ImGui::Checkbox("Roll out to cell clusters", &_includeClusters)) {
            _angle = 0;
            _angularVel = 0;
        }
        ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
        AlienImGui::Group("Edit");
        if (ImGui::Button("Copy")) {
            _copiedSelection = _simController->getSelectedSimulationData(_includeClusters);
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::BeginDisabled(!_copiedSelection);
        if (ImGui::Button("Paste")) {
            auto data = *_copiedSelection;
            auto center = _viewport->getCenterInWorldPos();
            data.setCenter(center);
            _simController->addAndSelectSimulationData(data);
            _editorModel->update();
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::BeginDisabled(_editorModel->isSelectionEmpty());
        ImGui::Button("Delete");
        ImGui::SameLine();
        ImGui::Button("Remove tensions");

        AlienImGui::Group("Center position and velocity");

        auto const& selectionData = _editorModel->getSelectionShallowData();

        auto centerPosX = _includeClusters ? selectionData.clusterCenterPosX : selectionData.centerPosX;
        auto origCenterPosX = centerPosX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Position X").textWidth(maxContentTextWidthScaled).format("%.2f"), centerPosX);

        auto centerPosY = _includeClusters ? selectionData.clusterCenterPosY : selectionData.centerPosY;
        auto origCenterPosY = centerPosY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Position Y").textWidth(maxContentTextWidthScaled).format("%.2f"), centerPosY);

        auto centerVelX = _includeClusters ? selectionData.clusterCenterVelX : selectionData.centerVelX;
        auto origCenterVelX = centerVelX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Velocity X").textWidth(maxContentTextWidthScaled).step(0.1f).format("%.2f"),
            centerVelX);

        auto centerVelY = _includeClusters ? selectionData.clusterCenterVelY : selectionData.centerVelY;
        auto origCenterVelY = centerVelY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Velocity Y").textWidth(maxContentTextWidthScaled).step(0.1f).format("%.2f"),
            centerVelY);

        AlienImGui::Group("Center rotation");
        auto origAngle = _angle;
        AlienImGui::SliderInputFloat(
            AlienImGui::SliderInputFloatParameters()
                .name("Angle")
                .textWidth(maxContentTextWidthScaled)
                .inputWidth(StyleRepository::getInstance().scaleContent(50))
                .min(-180.0f)
                .max(180.0f)
                .format("%.1f"),
            _angle);

        auto origAngularVel = _angularVel;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Angular velocity")
                .textWidth(maxContentTextWidthScaled)
                .step(0.01f)
                .format("%.2f"),
            _angularVel);

        if (centerPosX != origCenterPosX || centerPosY != origCenterPosY) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _includeClusters;
            updateData.posDeltaX = centerPosX - origCenterPosX;
            updateData.posDeltaY = centerPosY - origCenterPosY;
            _simController->shallowUpdateSelection(updateData);
            _editorModel->update();
        }

        if (centerVelX != origCenterVelX || centerVelY != origCenterVelY) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _includeClusters;
            updateData.velDeltaX = centerVelX - origCenterVelX;
            updateData.velDeltaY = centerVelY - origCenterVelY;
            _simController->shallowUpdateSelection(updateData);
            _editorModel->update();
        }

        if (_angle != origAngle) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _includeClusters;
            updateData.angleDelta = _angle - origAngle;
            _simController->shallowUpdateSelection(updateData);
            _editorModel->update();
        }

        if (_angularVel != origAngularVel) {
            ShallowUpdateSelectionData updateData;
            updateData.considerClusters = _includeClusters;
            updateData.angularVelDelta = _angularVel - origAngularVel;
            _simController->shallowUpdateSelection(updateData);
            _editorModel->update();
        }

        AlienImGui::Group("Colorize");
        colorButton("    ##color1", Const::IndividualCellColor1);
        ImGui::SameLine();
        colorButton("    ##color2", Const::IndividualCellColor2);
        ImGui::SameLine();
        colorButton("    ##color3", Const::IndividualCellColor3);
        ImGui::SameLine();
        colorButton("    ##color4", Const::IndividualCellColor4);
        ImGui::SameLine();
        colorButton("    ##color5", Const::IndividualCellColor5);
        ImGui::SameLine();
        colorButton("    ##color6", Const::IndividualCellColor6);
        ImGui::SameLine();
        colorButton("    ##color7", Const::IndividualCellColor7);

        ImGui::EndDisabled();

        _lastSelection = selection;
    }
    ImGui::End();
}

bool _ManipulatorWindow::isOn() const
{
    return _on;
}

void _ManipulatorWindow::setOn(bool value)
{
    _on = value;
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
