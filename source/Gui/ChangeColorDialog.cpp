#include "ChangeColorDialog.h"

#include <imgui.h>

#include "Base/Resources.h"

#include "AlienImGui.h"

_ChangeColorDialog::_ChangeColorDialog()
    : _AlienDialog("About")
{}

void _ChangeColorDialog::processIntern()
{
    ImGui::PushID("source color");
    AlienImGui::ComboColor(AlienImGui::ComboColorParameters().textWidth(0).width(70), _sourceColor);
    ImGui::PopID();
    ImGui::SameLine();
    ImGui::PushID("target color");
    AlienImGui::ComboColor(AlienImGui::ComboColorParameters().textWidth(0).width(70), _targetColor);
    ImGui::PopID();
    
    if (AlienImGui::Button("OK")) {
        close();
    }
    ImGui::SetItemDefaultFocus();
}
