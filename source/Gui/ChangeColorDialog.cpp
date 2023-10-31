#include "ChangeColorDialog.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "AlienImGui.h"
#include "StyleRepository.h"

_ChangeColorDialog::_ChangeColorDialog()
    : _AlienDialog("Change color")
{}

void _ChangeColorDialog::processIntern()
{
    if (ImGui::BeginChild("##child", ImVec2(0, ImGui::GetContentRegionAvail().y - scale(50)), false)) {

        if (ImGui::BeginTable("##", 3, ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0);
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, scale(20));
            ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0);
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Source color");

            ImGui::TableSetColumnIndex(2);
            ImGui::Text("Target color");

            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            ImGui::PushID("source color");
            AlienImGui::ComboColor(AlienImGui::ComboColorParameters().textWidth(0).width(0), _sourceColor);
            ImGui::PopID();

            ImGui::TableSetColumnIndex(1);
            AlienImGui::Text(ICON_FA_LONG_ARROW_ALT_RIGHT);

            ImGui::TableSetColumnIndex(2);
            ImGui::PushID("target color");
            AlienImGui::ComboColor(AlienImGui::ComboColorParameters().textWidth(0).width(0), _targetColor);
            ImGui::PopID();

            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
    AlienImGui::Separator();

    if (AlienImGui::Button("OK")) {
        close();
    }
    ImGui::SetItemDefaultFocus();
}
