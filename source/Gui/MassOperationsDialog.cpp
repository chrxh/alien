#include "MassOperationsDialog.h"

#include <imgui.h>

#include "Base/Definitions.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienImGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr RightColumnWidth = 120.0f;
}

void MassOperationsDialog::initIntern(SimulationFacade simulationFacade)
{
    _simulationFacade = simulationFacade;
}

void MassOperationsDialog::processIntern()
{
    AlienImGui::Group("Cell colors");
    ImGui::PushID("cell");
    ImGui::Checkbox("##colors", &_randomizeCellColors);
    ImGui::BeginDisabled(!_randomizeCellColors);
    ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
    colorCheckbox("##color1", Const::IndividualCellColor1, _checkedCellColors[0]);
    ImGui::SameLine();
    colorCheckbox("##color2", Const::IndividualCellColor2, _checkedCellColors[1]);
    ImGui::SameLine();
    colorCheckbox("##color3", Const::IndividualCellColor3, _checkedCellColors[2]);
    ImGui::SameLine();
    colorCheckbox("##color4", Const::IndividualCellColor4, _checkedCellColors[3]);
    ImGui::SameLine();
    colorCheckbox("##color5", Const::IndividualCellColor5, _checkedCellColors[4]);
    ImGui::SameLine();
    colorCheckbox("##color6", Const::IndividualCellColor6, _checkedCellColors[5]);
    ImGui::SameLine();
    colorCheckbox("##color7", Const::IndividualCellColor7, _checkedCellColors[6]);
    ImGui::EndDisabled();
    ImGui::PopID();

    AlienImGui::Group("Genome colors");
    ImGui::PushID("genome");
    ImGui::Checkbox("##colors", &_randomizeGenomeColors);
    ImGui::BeginDisabled(!_randomizeGenomeColors);
    ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
    colorCheckbox("##color1", Const::IndividualCellColor1, _checkedGenomeColors[0]);
    ImGui::SameLine();
    colorCheckbox("##color2", Const::IndividualCellColor2, _checkedGenomeColors[1]);
    ImGui::SameLine();
    colorCheckbox("##color3", Const::IndividualCellColor3, _checkedGenomeColors[2]);
    ImGui::SameLine();
    colorCheckbox("##color4", Const::IndividualCellColor4, _checkedGenomeColors[3]);
    ImGui::SameLine();
    colorCheckbox("##color5", Const::IndividualCellColor5, _checkedGenomeColors[4]);
    ImGui::SameLine();
    colorCheckbox("##color6", Const::IndividualCellColor6, _checkedGenomeColors[5]);
    ImGui::SameLine();
    colorCheckbox("##color7", Const::IndividualCellColor7, _checkedGenomeColors[6]);
    ImGui::EndDisabled();
    ImGui::PopID();

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

    AlienImGui::Group("Detonation countdown");
    ImGui::Checkbox("##countdown", &_randomizeCountdowns);
    ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
    posX = ImGui::GetCursorPos().x;
    ImGui::BeginDisabled(!_randomizeCountdowns);
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Minimum value").textWidth(RightColumnWidth), _minCountdown);
    ImGui::SetCursorPosX(posX);
    AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Maximum value").textWidth(RightColumnWidth), _maxCountdown);
    ImGui::EndDisabled();

    AlienImGui::Group("Mutants");
    ImGui::Checkbox("##mutationId", &_randomizeMutationId);
    ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
    AlienImGui::Text("Randomize mutation ids");

    AlienImGui::Group("Options");
    ImGui::Checkbox("##restrictToSelectedClusters", &_restrictToSelectedClusters);
    ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
    AlienImGui::Text("Restrict to selected cell networks");

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    ImGui::BeginDisabled(!isOkEnabled());
    if (AlienImGui::Button("OK")) {
        onExecute();
        close();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::SetItemDefaultFocus();
    if (AlienImGui::Button("Cancel")) {
        close();
    }

    validateAndCorrect();
}

MassOperationsDialog::MassOperationsDialog()
    : AlienDialog("Mass operations")
{
}

void MassOperationsDialog::colorCheckbox(std::string id, uint32_t cellColor, bool& check)
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

void MassOperationsDialog::onExecute()
{
    auto timestep = static_cast<uint32_t>(_simulationFacade->getCurrentTimestep());
    auto parameters = _simulationFacade->getSimulationParameters();
    auto generalSettings = _simulationFacade->getGeneralSettings();
    auto content = [&] {
        if (_restrictToSelectedClusters) {
            return _simulationFacade->getSelectedClusteredSimulationData(true);
        } else {
            return _simulationFacade->getClusteredSimulationData();
        }
    }();

    auto getColorVector = [](bool* colors) {
        std::vector<int> result;
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (colors[i]) {
                result.emplace_back(i);
            }
        }
        return result;
    };
    if (_randomizeCellColors) {
        DescriptionEditService::get().randomizeCellColors(content, getColorVector(_checkedCellColors));
    }
    if (_randomizeGenomeColors) {
        DescriptionEditService::get().randomizeGenomeColors(content, getColorVector(_checkedGenomeColors));
    }
    if (_randomizeEnergies) {
        DescriptionEditService::get().randomizeEnergies(content, _minEnergy, _maxEnergy);
    }
    if (_randomizeAges) {
        DescriptionEditService::get().randomizeAges(content, _minAge, _maxAge);
    }
    if (_randomizeCountdowns) {
        DescriptionEditService::get().randomizeCountdowns(content, _minCountdown, _maxCountdown);
    }
    if (_randomizeMutationId) {
        DescriptionEditService::get().randomizeMutationIds(content);
    }

    if (_restrictToSelectedClusters) {
        _simulationFacade->removeSelectedObjects(true);
        _simulationFacade->addAndSelectSimulationData(DataDescription(content));
    } else {
        _simulationFacade->closeSimulation();
        _simulationFacade->newSimulation(timestep, generalSettings, parameters);
        _simulationFacade->setClusteredSimulationData(content);       
    }
}

bool MassOperationsDialog::isOkEnabled()
{
    bool result = false;
    if (_randomizeCellColors) {
        for (bool checkColor : _checkedCellColors) {
            result |= checkColor;
        }
    }
    if (_randomizeGenomeColors) {
        for (bool checkColor : _checkedGenomeColors) {
            result |= checkColor;
        }
    }

    if (_randomizeEnergies) {
        result = true;
    }
    if (_randomizeAges) {
        result = true;
    }
    if (_randomizeCountdowns) {
        result = true;
    }
    if (_randomizeMutationId) {
        result = true;
    }
    return result;
}

void MassOperationsDialog::validateAndCorrect()
{
    _minAge = std::max(0, _minAge);
    _maxAge = std::max(0, _maxAge);
    _minEnergy = std::max(0.0f, _minEnergy);
    _maxEnergy = std::max(0.0f, _maxEnergy);
    _minCountdown = std::max(0, _minCountdown);
    _maxCountdown = std::max(0, _maxCountdown);

    if (_minAge > _maxAge) {
        _maxAge = _minAge;
    }
    if (_minEnergy > _maxEnergy) {
        _maxEnergy = _minEnergy;
    }
    if (_minCountdown> _maxCountdown) {
        _maxCountdown = _minCountdown;
    }
}
