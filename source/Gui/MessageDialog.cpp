#include "MessageDialog.h"

#include <imgui.h>

#include "AlienImGui.h"

MessageDialog& MessageDialog::getInstance()
{
    static MessageDialog instance;
    return instance;
}

void MessageDialog::process()
{
    if (!_show) {
        return;
    }
    ImGui::OpenPopup(_title.c_str());
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal(_title.c_str(), NULL, ImGuiWindowFlags_AlwaysAutoResize)) {

        AlienImGui::Text(_message);

        if (AlienImGui::Button("OK")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }
        ImGui::EndPopup();
    }
}

void MessageDialog::show(std::string const& title, std::string const& message)
{
    _show = true;
    _title = title;
    _message = message;
}
