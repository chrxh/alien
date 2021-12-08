#include "ActionsWindow.h"

#include "imgui.h"

#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineImpl/SimulationController.h"

#include "EditorModel.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"

namespace
{
    auto const ItemTextWidth = 120.0f;
}

_ActionsWindow::_ActionsWindow(
    EditorModel const& editorModel,
    SimulationController const& simController,
    StyleRepository const& styleRepository)
    : _editorModel(editorModel)
    , _simController(simController)
    , _styleRepository(styleRepository)
{
    _on = GlobalSettings::getInstance().getBoolState("editor.actions.active", true);
}

_ActionsWindow::~_ActionsWindow()
{
    GlobalSettings::getInstance().setBoolState("editor.actions.active", _on);
}

void _ActionsWindow::process()
{
    if (!_on) {
        return;
    }

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin("Actions", &_on)) {
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

        AlienImGui::Group("Center position and velocity");

        auto const& selectionData = _editorModel->getSelectionShallowData();

        auto centerPosX = _includeClusters ? selectionData.clusterCenterPosX : selectionData.centerPosX;
        auto origCenterPosX = centerPosX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Position X").textWidth(ItemTextWidth).format("%.2f"), centerPosX);

        auto centerPosY = _includeClusters ? selectionData.clusterCenterPosY : selectionData.centerPosY;
        auto origCenterPosY = centerPosY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Position Y").textWidth(ItemTextWidth).format("%.2f"), centerPosY);

        auto centerVelX = _includeClusters ? selectionData.clusterCenterVelX : selectionData.centerVelX;
        auto origCenterVelX = centerVelX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Velocity X").textWidth(ItemTextWidth).step(0.1f).format("%.2f"),
            centerVelX);

        auto centerVelY = _includeClusters ? selectionData.clusterCenterVelY : selectionData.centerVelY;
        auto origCenterVelY = centerVelY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Velocity Y").textWidth(ItemTextWidth).step(0.1f).format("%.2f"),
            centerVelY);

        AlienImGui::Group("Center rotation");
        auto origAngle = _angle;
        AlienImGui::SliderInputFloat(
            AlienImGui::SliderInputFloatParameters()
                .name("Angle")
                .textWidth(ItemTextWidth)
                .min(-180.0f)
                .max(180.0f)
                .format("%.1f"),
            _angle);

        auto origAngularVel = _angularVel;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Angular velocity")
                .textWidth(ItemTextWidth)
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

        ImGui::EndDisabled();

        ImGui::End();

        _lastSelection = selection;
    }
}

bool _ActionsWindow::isOn() const
{
    return _on;
}

void _ActionsWindow::setOn(bool value)
{
    _on = value;
}

bool _ActionsWindow::hasSelectionChanged(SelectionShallowData const& selection) const
{
    if(!_lastSelection) {
        return false;
    }
    return _lastSelection->numCells != selection.numCells || _lastSelection->numParticles != selection.numParticles;
}
