#include "MassOperationsDialog.h"

#include <imgui.h>

#include <Base/Definitions.h>

#include <EngineInterface/Colors.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/GenomeDescInfoService.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "MutationRateDialog.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr RightColumnWidth = 120.0f;
    auto constexpr MinColumnWidth = 300.0f;

    std::vector<std::string> getActiveMutations(MutationRatesDesc const& mutationRates)
    {
        std::vector<std::string> activeMutations;
        if (mutationRates._connectionMutation1._probability > 0.0f || mutationRates._connectionMutation2._probability > 0.0f) {
            activeMutations.push_back("Connection mutations");
        }
        if (mutationRates._neuronMutation1._probability > 0.0f || mutationRates._neuronMutation2._probability > 0.0f) {
            activeMutations.push_back("Neuron mutations");
        }
        if (mutationRates._lineageMutationProbability > 0.0f) {
            activeMutations.push_back("Lineage mutation");
        }
        return activeMutations;
    }
}

void MassOperationsDialog::initIntern() {}

void MassOperationsDialog::processIntern()
{
    auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;

    if (ImGui::BeginChild("##content", ImVec2(0, -scale(50.0f)), 0)) {
        AlienGui::DynamicTableLayout table(MinColumnWidth);
        if (table.begin()) {

            AlienGui::Group(AlienGui::GroupParameters().text("Randomize object colors"));
            ImGui::PushID("cell");
            ImGui::Checkbox("##colors", &_randomizeCellColors);
            ImGui::BeginDisabled(!_randomizeCellColors);
            ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
            for (int i = 0; i < MAX_COLORS; ++i) {
                if (i > 0) {
                    ImGui::SameLine();
                }
                auto id = "##color" + std::to_string(i + 1);
                colorCheckbox(id, customizationColors.values[i].toRgbColor(), _checkedCellColors[i]);
            }
            ImGui::EndDisabled();
            ImGui::PopID();

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Randomize genome colors"));
            ImGui::PushID("genome");
            ImGui::Checkbox("##colors", &_randomizeGenomeColors);
            ImGui::BeginDisabled(!_randomizeGenomeColors);
            ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
            for (int i = 0; i < MAX_COLORS; ++i) {
                if (i > 0) {
                    ImGui::SameLine();
                }
                auto id = "##color" + std::to_string(i + 1);
                colorCheckbox(id, customizationColors.values[i].toRgbColor(), _checkedGenomeColors[i]);
            }
            ImGui::EndDisabled();
            ImGui::PopID();

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Randomize cell energies"));
            ImGui::Checkbox("##energies", &_randomizeEnergies);
            ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
            auto posX = ImGui::GetCursorPos().x;
            ImGui::BeginDisabled(!_randomizeEnergies);
            AlienGui::InputFloat(AlienGui::InputFloatParameters().format("%.1f").name("Minimum energy").textWidth(RightColumnWidth), _minEnergy);
            ImGui::SetCursorPosX(posX);
            AlienGui::InputFloat(AlienGui::InputFloatParameters().format("%.1f").name("Maximum energy").textWidth(RightColumnWidth), _maxEnergy);
            ImGui::EndDisabled();

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Randomize cell ages"));
            ImGui::Checkbox("##ages", &_randomizeAges);
            ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
            posX = ImGui::GetCursorPos().x;
            ImGui::BeginDisabled(!_randomizeAges);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Minimum age").textWidth(RightColumnWidth), _minAge);
            ImGui::SetCursorPosX(posX);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Maximum age").textWidth(RightColumnWidth), _maxAge);
            ImGui::EndDisabled();

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Randomize detonation countdown"));
            ImGui::Checkbox("##countdown", &_randomizeCountdowns);
            ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
            posX = ImGui::GetCursorPos().x;
            ImGui::BeginDisabled(!_randomizeCountdowns);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Minimum value").textWidth(RightColumnWidth), _minCountdown);
            ImGui::SetCursorPosX(posX);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Maximum value").textWidth(RightColumnWidth), _maxCountdown);
            ImGui::EndDisabled();

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Randomize mutants"));
            ImGui::Checkbox("##lineageId", &_randomizeLineageId);
            ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
            AlienGui::Text("Randomize lineage ids");

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Randomize fluid glow"));
            ImGui::Checkbox("##glow", &_randomizeGlow);
            ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
            posX = ImGui::GetCursorPos().x;
            ImGui::BeginDisabled(!_randomizeGlow);
            AlienGui::SliderFloat(AlienGui::SliderFloatParameters().format("%.2f").name("Minimum glow").min(0).max(1).textWidth(RightColumnWidth), &_minGlow);
            ImGui::SetCursorPosX(posX);
            AlienGui::SliderFloat(AlienGui::SliderFloatParameters().format("%.2f").name("Maximum glow").min(0).max(1).textWidth(RightColumnWidth), &_maxGlow);
            ImGui::EndDisabled();

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Mutation rates"));
            ImGui::Checkbox("##mutationRates", &_randomizeMutationRates);
            ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
            ImGui::BeginDisabled(!_randomizeMutationRates);
            auto buttonWidth = scale(60.0f);
            auto availableWidth = ImGui::GetContentRegionAvail().x;
            auto listBoxWidth = availableWidth - buttonWidth - ImGui::GetStyle().ItemSpacing.x;
            AlienGui::ListBox(AlienGui::ListBoxParameters().items(getActiveMutations(_mutationRates)).width(listBoxWidth));
            ImGui::SameLine();
            if (AlienGui::Button("Edit")) {
                MutationRateDialog::get().open(_mutationRates, [this](MutationRatesDesc const& mutationRates) { _mutationRates = mutationRates; });
            }
            ImGui::EndDisabled();

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Options").highlighted(true));
            ImGui::Checkbox("##restrictToSelection", &_restrictToSelectedCreatures);
            ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
            AlienGui::Text("Restrict to selection");

            table.end();
        }
    }
    ImGui::EndChild();

    AlienGui::Separator();

    ImGui::BeginDisabled(!isOkEnabled());
    if (AlienGui::Button("OK")) {
        onExecute();
        close();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::SetItemDefaultFocus();
    if (AlienGui::Button("Cancel")) {
        close();
    }

    validateAndCorrect();
}

MassOperationsDialog::MassOperationsDialog()
    : AlienDialog("Mass operations")
{}

void MassOperationsDialog::colorCheckbox(std::string id, uint32_t cellColor, bool& check)
{
    float h, s, v;
    AlienGui::ConvertRGBtoHSV(cellColor, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, (ImVec4)ImColor::HSV(h, s * 0.6f, v * 0.3f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.5f));
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, (ImVec4)ImColor::HSV(h, s * 0.8f, v * 0.8f));
    ImGui::PushStyleColor(ImGuiCol_CheckMark, (ImVec4)ImColor::HSV(h, s, v));
    ImGui::Checkbox(id.c_str(), &check);
    ImGui::PopStyleColor(4);
}

void MassOperationsDialog::onExecute()
{
    auto timestep = static_cast<uint32_t>(_SimulationFacade::get()->getCurrentTimestep());
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    auto content = [&] {
        if (_restrictToSelectedCreatures) {
            return _SimulationFacade::get()->getSelectedSimulationData(true);
        } else {
            return _SimulationFacade::get()->getSimulationData();
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
        DescEditService::get().randomizeCellColors(content, getColorVector(_checkedCellColors));
    }
    if (_randomizeGenomeColors) {
        DescEditService::get().randomizeGenomeColors(content, getColorVector(_checkedGenomeColors));
    }
    if (_randomizeEnergies) {
        DescEditService::get().randomizeEnergies(content, _minEnergy, _maxEnergy);
    }
    if (_randomizeAges) {
        DescEditService::get().randomizeAges(content, _minAge, _maxAge);
    }
    if (_randomizeCountdowns) {
        DescEditService::get().randomizeCountdowns(content, _minCountdown, _maxCountdown);
    }
    if (_randomizeLineageId) {
        DescEditService::get().randomizeLineageIds(content);
    }
    if (_randomizeGlow) {
        DescEditService::get().randomizeGlow(content, _minGlow, _maxGlow);
    }
    if (_randomizeMutationRates) {
        for (auto& genome : content._genomes) {
            genome._mutationRates = _mutationRates;
        }
    }

    if (_restrictToSelectedCreatures) {
        _SimulationFacade::get()->removeSelectedObjects(true);
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
    } else {
        _SimulationFacade::get()->closeSimulation();
        _SimulationFacade::get()->newSimulation(timestep, worldSize, parameters);
        _SimulationFacade::get()->setSimulationData(content);
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
    if (_randomizeLineageId) {
        result = true;
    }
    if (_randomizeGlow) {
        result = true;
    }
    if (_randomizeMutationRates) {
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
    _minGlow = std::clamp(_minGlow, 0.0f, 1.0f);
    _maxGlow = std::clamp(_maxGlow, 0.0f, 1.0f);

    if (_minAge > _maxAge) {
        _maxAge = _minAge;
    }
    if (_minEnergy > _maxEnergy) {
        _maxEnergy = _minEnergy;
    }
    if (_minCountdown > _maxCountdown) {
        _maxCountdown = _minCountdown;
    }
    if (_minGlow > _maxGlow) {
        _maxGlow = _minGlow;
    }
}
