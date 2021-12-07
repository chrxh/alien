#include "ActionsWindow.h"

#include "imgui.h"

#include "EngineInterface/ShallowUpdateSelectionData.h"
#include "EngineImpl/SimulationController.h"

#include "EditorModel.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"

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
        ImGui::Checkbox("Roll out to cell clusters", &_includeClusters);
        ImGui::BeginDisabled(_editorModel->isSelectionEmpty());

        AlienImGui::Group("Center properties");

        auto const& selectionData = _editorModel->getSelectionShallowData();

        auto centerPosX = _includeClusters ? selectionData.clusterCenterPosX : selectionData.centerPosX;
        auto origCenterPosX = centerPosX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Position X").format("%.2f"), centerPosX);

        auto centerPosY = _includeClusters ? selectionData.clusterCenterPosY : selectionData.centerPosY;
        auto origCenterPosY = centerPosY;
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Position Y").format("%.2f"), centerPosY);

        auto centerVelX = _includeClusters ? selectionData.clusterCenterVelX : selectionData.centerVelX;
        auto origCenterVelX = centerVelX;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Velocity X").step(0.1f).format("%.2f"), centerVelX);

        auto centerVelY = _includeClusters ? selectionData.clusterCenterVelY : selectionData.centerVelY;
        auto origCenterVelY = centerVelY;
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Velocity Y").step(0.1f).format("%.2f"), centerVelY);

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

        AlienImGui::Group("Rotation");
        AlienImGui::SliderInputFloat(AlienImGui::SliderInputFloatParameters().name("Angle").min(-180.0f).max(180.0f).format("%.1f deg"), _angle);

        ImGui::EndDisabled();

        ImGui::End();
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
