#include "MassOperationsDialog.h"

#include <imgui.h>

#include <Base/Definitions.h>
#include <Base/GlobalSettings.h>

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
    auto constexpr SettingsPrefix = "dialogs.mass operations.";

    void loadValue(bool& value, std::string const& key)
    {
        value = GlobalSettings::get().getValue(SettingsPrefix + key, value);
    }

    void loadValue(int& value, std::string const& key)
    {
        value = GlobalSettings::get().getValue(SettingsPrefix + key, value);
    }

    void loadValue(float& value, std::string const& key)
    {
        value = GlobalSettings::get().getValue(SettingsPrefix + key, value);
    }

    void saveValue(bool value, std::string const& key)
    {
        GlobalSettings::get().setValue(SettingsPrefix + key, value);
    }

    void saveValue(int value, std::string const& key)
    {
        GlobalSettings::get().setValue(SettingsPrefix + key, value);
    }

    void saveValue(float value, std::string const& key)
    {
        GlobalSettings::get().setValue(SettingsPrefix + key, value);
    }

    void loadMutationRates(MutationRatesDesc& mutationRates)
    {
        loadValue(mutationRates._lineageMutationProbability, "mutation rates.lineage mutation probability");

        loadValue(mutationRates._connectionMutation1._probability, "mutation rates.connection mutation 1.probability");
        loadValue(mutationRates._connectionMutation1._sigma, "mutation rates.connection mutation 1.sigma");
        loadValue(mutationRates._connectionMutation2._probability, "mutation rates.connection mutation 2.probability");
        loadValue(mutationRates._connectionMutation2._sigma, "mutation rates.connection mutation 2.sigma");

        loadValue(mutationRates._neuronMutation1._probability, "mutation rates.neuron mutation 1.probability");
        loadValue(mutationRates._neuronMutation1._weightSigma, "mutation rates.neuron mutation 1.weight sigma");
        loadValue(mutationRates._neuronMutation1._biasSigma, "mutation rates.neuron mutation 1.bias sigma");
        loadValue(mutationRates._neuronMutation1._activationFunctionProbability, "mutation rates.neuron mutation 1.activation function probability");

        loadValue(mutationRates._neuronMutation2._probability, "mutation rates.neuron mutation 2.probability");
        loadValue(mutationRates._neuronMutation2._weightSigma, "mutation rates.neuron mutation 2.weight sigma");
        loadValue(mutationRates._neuronMutation2._biasSigma, "mutation rates.neuron mutation 2.bias sigma");
        loadValue(mutationRates._neuronMutation2._activationFunctionProbability, "mutation rates.neuron mutation 2.activation function probability");
    }

    void saveMutationRates(MutationRatesDesc const& mutationRates)
    {
        saveValue(mutationRates._lineageMutationProbability, "mutation rates.lineage mutation probability");

        saveValue(mutationRates._connectionMutation1._probability, "mutation rates.connection mutation 1.probability");
        saveValue(mutationRates._connectionMutation1._sigma, "mutation rates.connection mutation 1.sigma");
        saveValue(mutationRates._connectionMutation2._probability, "mutation rates.connection mutation 2.probability");
        saveValue(mutationRates._connectionMutation2._sigma, "mutation rates.connection mutation 2.sigma");

        saveValue(mutationRates._neuronMutation1._probability, "mutation rates.neuron mutation 1.probability");
        saveValue(mutationRates._neuronMutation1._weightSigma, "mutation rates.neuron mutation 1.weight sigma");
        saveValue(mutationRates._neuronMutation1._biasSigma, "mutation rates.neuron mutation 1.bias sigma");
        saveValue(mutationRates._neuronMutation1._activationFunctionProbability, "mutation rates.neuron mutation 1.activation function probability");

        saveValue(mutationRates._neuronMutation2._probability, "mutation rates.neuron mutation 2.probability");
        saveValue(mutationRates._neuronMutation2._weightSigma, "mutation rates.neuron mutation 2.weight sigma");
        saveValue(mutationRates._neuronMutation2._biasSigma, "mutation rates.neuron mutation 2.bias sigma");
        saveValue(mutationRates._neuronMutation2._activationFunctionProbability, "mutation rates.neuron mutation 2.activation function probability");
    }
}

void MassOperationsDialog::initIntern()
{
    loadValue(_randomizeCellColors, "randomize cell colors");
    for (int i = 0; i < MAX_COLORS; ++i) {
        loadValue(_checkedCellColors[i], "checked cell colors." + std::to_string(i));
    }

    loadValue(_randomizeGenomeColors, "randomize genome colors");
    for (int i = 0; i < MAX_COLORS; ++i) {
        loadValue(_checkedGenomeColors[i], "checked genome colors." + std::to_string(i));
    }

    loadValue(_randomizeEnergies, "randomize energies");
    loadValue(_minEnergy, "minimum energy");
    loadValue(_maxEnergy, "maximum energy");

    loadValue(_randomizeAges, "randomize ages");
    loadValue(_minAge, "minimum age");
    loadValue(_maxAge, "maximum age");

    loadValue(_randomizeCountdowns, "randomize countdowns");
    loadValue(_minCountdown, "minimum countdown");
    loadValue(_maxCountdown, "maximum countdown");

    loadValue(_randomizeLineageId, "randomize lineage id");

    loadValue(_randomizeGlow, "randomize glow");
    loadValue(_minGlow, "minimum glow");
    loadValue(_maxGlow, "maximum glow");

    loadValue(_randomizeMutationRates, "randomize mutation rates");
    loadMutationRates(_mutationRates);

    loadValue(_restrictToSelectedCreatures, "restrict to selected creatures");
    validateAndCorrect();
}

void MassOperationsDialog::shutdownIntern()
{
    saveValue(_randomizeCellColors, "randomize cell colors");
    for (int i = 0; i < MAX_COLORS; ++i) {
        saveValue(_checkedCellColors[i], "checked cell colors." + std::to_string(i));
    }

    saveValue(_randomizeGenomeColors, "randomize genome colors");
    for (int i = 0; i < MAX_COLORS; ++i) {
        saveValue(_checkedGenomeColors[i], "checked genome colors." + std::to_string(i));
    }

    saveValue(_randomizeEnergies, "randomize energies");
    saveValue(_minEnergy, "minimum energy");
    saveValue(_maxEnergy, "maximum energy");

    saveValue(_randomizeAges, "randomize ages");
    saveValue(_minAge, "minimum age");
    saveValue(_maxAge, "maximum age");

    saveValue(_randomizeCountdowns, "randomize countdowns");
    saveValue(_minCountdown, "minimum countdown");
    saveValue(_maxCountdown, "maximum countdown");

    saveValue(_randomizeLineageId, "randomize lineage id");

    saveValue(_randomizeGlow, "randomize glow");
    saveValue(_minGlow, "minimum glow");
    saveValue(_maxGlow, "maximum glow");

    saveValue(_randomizeMutationRates, "randomize mutation rates");
    saveMutationRates(_mutationRates);

    saveValue(_restrictToSelectedCreatures, "restrict to selected creatures");
}

void MassOperationsDialog::processIntern()
{
    auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;

    AlienGui::Group(AlienGui::GroupParameters().text("Filter").highlighted(true));
    ImGui::Checkbox("##restrictToSelection", &_restrictToSelectedCreatures);
    ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
    AlienGui::Text("Restrict to selection");

    AlienGui::Group(AlienGui::GroupParameters().text("Operations").highlighted(true));

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
            AlienGui::ListBox(AlienGui::ListBoxParameters().items(_mutationRates.getActiveMutationTypes()).width(listBoxWidth));
            ImGui::SameLine();
            if (AlienGui::Button("Edit")) {
                MutationRateDialog::get().openNested(_mutationRates, [this](MutationRatesDesc const& mutationRates) { _mutationRates = mutationRates; });
            }
            ImGui::EndDisabled();

            table.next();

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
    MutationRateDialog::get().processNested();
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
        DescEditService::get().setMutationRates(content, _mutationRates);
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
