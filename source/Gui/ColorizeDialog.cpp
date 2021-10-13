#include "ColorizeDialog.h"

#include "imgui.h"

#include "Base/Definitions.h"
#include "EngineInterface/Colors.h"

_ColorizeDialog::_ColorizeDialog(SimulationController const& simController)
    : _simController(simController)
{}

void _ColorizeDialog::process()
{
    if (!_show) {
        return;
    }
    auto name = "Colorize";
    ImGui::OpenPopup(name);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal(name, NULL, ImGuiWindowFlags_AlwaysAutoResize)) {

        ImGui::Text("Choose color(s):");

        checkbox("##color1", Const::IndividualCellColor1, _checkColor1);
        ImGui::SameLine();
        checkbox("##color2", Const::IndividualCellColor2, _checkColor2);
        ImGui::SameLine();
        checkbox("##color3", Const::IndividualCellColor3, _checkColor3);
        ImGui::SameLine();
        checkbox("##color4", Const::IndividualCellColor4, _checkColor4);
        ImGui::SameLine();
        checkbox("##color5", Const::IndividualCellColor5, _checkColor5);
        ImGui::SameLine();
        checkbox("##color6", Const::IndividualCellColor6, _checkColor6);
        ImGui::SameLine();
        checkbox("##color7", Const::IndividualCellColor7, _checkColor7);

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        bool anySelected = _checkColor1 || _checkColor2 || _checkColor3 || _checkColor4 || _checkColor5 || _checkColor6
            || _checkColor7;
        ImGui::BeginDisabled(!anySelected);
        if (ImGui::Button("OK")) {
            onColorize();
            ImGui::CloseCurrentPopup();
            _show = false;
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::SetItemDefaultFocus();
        if (ImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }

        ImGui::EndPopup();
    }
}

void _ColorizeDialog::show()
{
    _show = true;
}

void _ColorizeDialog::checkbox(std::string id, uint64_t cellColor, bool& check)
{
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(
        toFloat((cellColor >> 16) & 0xff) / 255,
        toFloat((cellColor >> 8) & 0xff) / 255,
        toFloat((cellColor & 0xff)) / 255,
        h,
        s,
        v);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(h, s * 0.6f, v * 0.3f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.5f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(h, s * 0.8f, v * 0.8f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, (ImVec4)ImColor::HSV(h, s, v));
    ImGui::Checkbox(id.c_str(), &check);
    ImGui::PopStyleColor(4);
}

void _ColorizeDialog::onColorize() {
}
