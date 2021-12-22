#include "ColorizeDialog.h"

#include <imgui.h>

#include "Base/Definitions.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineImpl/SimulationController.h"

#include "AlienImGui.h"

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

        ImGui::Text("Select color(s):");

        colorCheckbox("##color1", Const::IndividualCellColor1, _checkColors[0]);
        ImGui::SameLine();
        colorCheckbox("##color2", Const::IndividualCellColor2, _checkColors[1]);
        ImGui::SameLine();
        colorCheckbox("##color3", Const::IndividualCellColor3, _checkColors[2]);
        ImGui::SameLine();
        colorCheckbox("##color4", Const::IndividualCellColor4, _checkColors[3]);
        ImGui::SameLine();
        colorCheckbox("##color5", Const::IndividualCellColor5, _checkColors[4]);
        ImGui::SameLine();
        colorCheckbox("##color6", Const::IndividualCellColor6, _checkColors[5]);
        ImGui::SameLine();
        colorCheckbox("##color7", Const::IndividualCellColor7, _checkColors[6]);

        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::Spacing();

        bool anySelected = false;
        for (bool checkColor : _checkColors) {
            anySelected |= checkColor;
        }
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

void _ColorizeDialog::colorCheckbox(std::string id, uint32_t cellColor, bool& check)
{
    float h, s, v;
    AlienImGui::convertRGBtoHSV(cellColor, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(h, s * 0.6f, v * 0.3f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.5f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(h, s * 0.8f, v * 0.8f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, (ImVec4)ImColor::HSV(h, s, v));
    ImGui::Checkbox(id.c_str(), &check);
    ImGui::PopStyleColor(4);
}

void _ColorizeDialog::onColorize()
{
    auto timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
    auto settings = _simController->getSettings();
    auto symbolMap = _simController->getSymbolMap();
    auto content = _simController->getSimulationData({0, 0}, _simController->getWorldSize());

    std::vector<int> colorCodes;
    for (int i = 0; i < 7; ++i) {
        if(_checkColors[i]) {
            colorCodes.emplace_back(i);
        }
    }
    DescriptionHelper::colorize(content, colorCodes);

    _simController->closeSimulation();
    _simController->newSimulation(timestep, settings, symbolMap);
    _simController->setSimulationData(content);
}
