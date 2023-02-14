#include "RandomizeDialog.h"

#include <imgui.h>

#include "Base/Definitions.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"

#include "AlienImGui.h"

namespace
{
    auto constexpr RightColumnWidth = 120.0f;
}
_RandomizeDialog::_RandomizeDialog(SimulationController const& simController)
    : _simController(simController)
{}

void _RandomizeDialog::process()
{
    if (!_show) {
        return;
    }
    auto name = "Randomize";
    ImGui::OpenPopup(name);
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    if (ImGui::BeginPopupModal(name, NULL)) {

        AlienImGui::Group("Colors");
        ImGui::Checkbox("##colors", &_randomizeColors);
        ImGui::BeginDisabled(!_randomizeColors);
        ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
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
        ImGui::EndDisabled();

        AlienImGui::Group("Cell Energies");
        ImGui::Checkbox("##energies", &_randomizeEnergies);
        ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
        auto posX = ImGui::GetCursorPos().x;
        ImGui::BeginDisabled(!_randomizeEnergies);
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().format("%.1f").name("Minimum energy").textWidth(RightColumnWidth), _minEnergy);
        ImGui::SetCursorPosX(posX);
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().format("%.1f").name("Maximum energy").textWidth(RightColumnWidth), _maxEnergy);
        ImGui::EndDisabled();

        AlienImGui::Group("Cell ages");
        ImGui::Checkbox("##ages", &_randomizeAges);
        ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
        posX = ImGui::GetCursorPos().x;
        ImGui::BeginDisabled(!_randomizeAges);
        AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Minimum age").textWidth(RightColumnWidth), _minAge);
        ImGui::SetCursorPosX(posX);
        AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Maximum age").textWidth(RightColumnWidth), _maxAge);
        ImGui::EndDisabled();

        AlienImGui::Separator();

        ImGui::BeginDisabled(!isOkEnabled());
        if (AlienImGui::Button("OK")) {
            onRandomize();
            ImGui::CloseCurrentPopup();
            _show = false;
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::SetItemDefaultFocus();
        if (AlienImGui::Button("Cancel")) {
            ImGui::CloseCurrentPopup();
            _show = false;
        }

        ImGui::EndPopup();

        validationAndCorrection();
    }
}

void _RandomizeDialog::show()
{
    _show = true;
}

void _RandomizeDialog::colorCheckbox(std::string id, uint32_t cellColor, bool& check)
{
    float h, s, v;
    AlienImGui::ConvertRGBtoHSV(cellColor, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(h, s * 0.6f, v * 0.3f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.5f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(h, s * 0.8f, v * 0.8f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, (ImVec4)ImColor::HSV(h, s, v));
    ImGui::Checkbox(id.c_str(), &check);
    ImGui::PopStyleColor(4);
}

void _RandomizeDialog::onRandomize()
{
    auto timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
    auto parameters = _simController->getSimulationParameters();
    auto generalSettings = _simController->getGeneralSettings();
    auto content = _simController->getClusteredSimulationData();

    std::vector<int> colorCodes;
    for (int i = 0; i < 7; ++i) {
        if(_checkColors[i]) {
            colorCodes.emplace_back(i);
        }
    }
    if (_randomizeColors) {
        DescriptionHelper::randomizeColors(content, colorCodes);
    }
    if (_randomizeEnergies) {
        DescriptionHelper::randomizeEnergies(content, _minEnergy, _maxEnergy);
    }
    if (_randomizeAges) {
        DescriptionHelper::randomizeAges(content, _minAge, _maxAge);
    }

    _simController->closeSimulation();
    _simController->newSimulation(timestep, generalSettings, parameters);
    _simController->setClusteredSimulationData(content);
}

bool _RandomizeDialog::isOkEnabled()
{
    bool result = false;
    if (_randomizeColors) {
        for (bool checkColor : _checkColors) {
            result |= checkColor;
        }
    }
    if (_randomizeEnergies) {
        result = true;
    }
    if (_randomizeAges) {
        result = true;
    }
    return result;
}

void _RandomizeDialog::validationAndCorrection()
{
    if (_minAge > _maxAge) {
        _maxAge = _minAge;
    }
    if (_minEnergy > _maxEnergy) {
        _maxEnergy = _minEnergy;
    }
}
