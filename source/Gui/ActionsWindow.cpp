#include "ActionsWindow.h"

#include "imgui.h"

#include "EditorModel.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "AlienImGui.h"

_ActionsWindow::_ActionsWindow(
    EditorModel const& editorModel,
    StyleRepository const& styleRepository)
    : _editorModel(editorModel)
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

        if (ImGui::BeginTable("##", 2, ImGuiTableFlags_SizingStretchProp)) {

            auto selectionData = _editorModel->getSelectionShallowData();
            auto origSelectionData = selectionData;

            //center pos x
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
            auto& centerPosX = _includeClusters ? selectionData.clusterCenterPosX : selectionData.centerPosX;
            ImGui::InputFloat("##centerX", &centerPosX, 1.0f, 0, "%.2f");
            ImGui::PopItemWidth();

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("Position X");

            //center pos y
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
            auto& centerPosY = _includeClusters ? selectionData.clusterCenterPosY : selectionData.centerPosY;
            ImGui::InputFloat("##centerY", &centerPosY, 1.0f, 0, "%.2f");
            ImGui::PopItemWidth();

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("Position Y");

            //center vel x
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
            auto& centerVelX = _includeClusters ? selectionData.clusterCenterVelX: selectionData.centerVelX;
            ImGui::InputFloat("##velX", &centerVelX, 0.1f, 0, "%.2f");
            ImGui::PopItemWidth();

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("Velocity X");

            //center vel y
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x);
            auto& centerVelY = _includeClusters ? selectionData.clusterCenterVelY : selectionData.centerVelY;
            ImGui::InputFloat("##velY", &centerVelY, 0.1f, 0, "%.2f");
            ImGui::PopItemWidth();

            ImGui::TableSetColumnIndex(1);
            ImGui::Text("Velocity Y");

            ImGui::EndTable();

            if (selectionData != origSelectionData) {
                _editorModel->setSelectionShallowData(selectionData);
            }
        }

        AlienImGui::Group("Rotation");

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
