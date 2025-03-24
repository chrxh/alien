#include "ParametersSpecGuiService.h"

#include <algorithm>
#include <ranges>

#include "EngineInterface/SimulationParametersSpecificationService.h"
#include "EngineInterface/SimulationParametersTypes.h"
#include "EngineInterface/SimulationParameters.h"

#include "AlienImGui.h"

namespace
{
    auto constexpr RightColumnWidth = 285.0f;
}

void ParametersSpecGuiService::createWidgetsFromSpec(
    ParametersSpec const& parametersSpecs,
    int locationIndex,
    SimulationParameters& parameters,
    SimulationParameters& origParameters) const
{
    auto locationType = getLocationType(locationIndex, parameters);

    for (auto const& groupSpec : parametersSpecs._groups) {
        if (!isVisible(groupSpec, locationType)) {
            continue;
        }
        ImGui::PushID(groupSpec._name.c_str());
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name(groupSpec._name))) {
            createWidgetsFromParameterSpecs(groupSpec._parameters, locationIndex, parameters, origParameters);
        }
        ImGui::PopID();
        AlienImGui::EndTreeNode();
    }
}

void ParametersSpecGuiService::createWidgetsFromParameterSpecs(
    std::vector<ParameterSpec> const& parameterSpecs,
    int locationIndex,
    SimulationParameters& parameters,
    SimulationParameters& origParameters) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto locationType = getLocationType(locationIndex, parameters);

    for (auto const& parameterSpec : parameterSpecs) {
        if (!isVisible(parameterSpec, locationType)) {
            continue;
        }
        if (std::holds_alternative<FloatSpec>(parameterSpec._type)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._type);
            bool* pinned = floatSpec._pinnedAddress.has_value() ? &specService.getValueRef<bool>(
                                                                      parameterSpec._visibleInBase,
                                                                      parameterSpec._visibleInZone,
                                                                      parameterSpec._visibleInSource,
                                                                      floatSpec._pinnedAddress.value(),
                                                                      parameters,
                                                                      locationIndex)
                                                                : nullptr;
            if (parameterSpec._valueAddress.has_value()) {
                auto& value = specService.getValueRef<float>(parameterSpec, parameters, locationIndex);
                auto& origValue = specService.getValueRef<float>(parameterSpec, origParameters, locationIndex);
                AlienImGui::SliderFloat(
                    AlienImGui::SliderFloatParameters()
                        .name(parameterSpec._name)
                        .textWidth(RightColumnWidth)
                        .min(floatSpec._min)
                        .max(floatSpec._max)
                        .logarithmic(floatSpec._logarithmic)
                        .format(floatSpec._format)
                        .infinity(floatSpec._infinity)
                        .defaultValue(&origValue)
                        .tooltip(parameterSpec._tooltip)
                        .colorDependence(parameterSpec._colorDependence),
                    &value,
                    nullptr,
                    pinned);
            } else {
                auto getter = floatSpec._valueGetter.value();
                auto setter = floatSpec._valueSetter.value();
                auto value = getter(parameters, locationIndex);
                auto origValue = getter(origParameters, locationIndex);

                if (AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name(parameterSpec._name)
                            .textWidth(RightColumnWidth)
                            .min(floatSpec._min)
                            .max(floatSpec._max)
                            .logarithmic(floatSpec._logarithmic)
                            .format(floatSpec._format)
                            .infinity(floatSpec._infinity)
                            .defaultValue(&origValue)
                            .tooltip(parameterSpec._tooltip)
                            .colorDependence(parameterSpec._colorDependence),
                        &value,
                        nullptr,
                        pinned)) {

                    setter(value, parameters, locationIndex);
                }
            }
        } else if (std::holds_alternative<IntSpec>(parameterSpec._type)) {
            auto const& intSpec = std::get<IntSpec>(parameterSpec._type);
            auto& value = specService.getValueRef<int>(parameterSpec, parameters, locationIndex);
            auto& origValue = specService.getValueRef<int>(parameterSpec, origParameters, locationIndex);
            AlienImGui::SliderInt(
                AlienImGui::SliderIntParameters()
                    .name(parameterSpec._name)
                    .textWidth(RightColumnWidth)
                    .min(intSpec._min)
                    .max(intSpec._max)
                    .logarithmic(intSpec._logarithmic)
                    .infinity(intSpec._infinity)
                    .defaultValue(&origValue)
                    .tooltip(parameterSpec._tooltip)
                    .colorDependence(parameterSpec._colorDependence),
                &value);
        } else if (std::holds_alternative<BoolSpec>(parameterSpec._type)) {
            auto& value = specService.getValueRef<bool>(parameterSpec, parameters, locationIndex);
            auto& origValue = specService.getValueRef<bool>(parameterSpec, origParameters, locationIndex);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(origValue).tooltip(parameterSpec._tooltip),
                value);
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._type)) {
            auto& value = specService.getValueRef<Char64>(parameterSpec, parameters, locationIndex);
            auto& origValue = specService.getValueRef<Char64>(parameterSpec, origParameters, locationIndex);
            AlienImGui::InputText(
                AlienImGui::InputTextParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(origValue).tooltip(parameterSpec._tooltip),
                value,
                sizeof(Char64) / sizeof(char));
        } else if (std::holds_alternative<ColorSpec>(parameterSpec._type)) {
            auto& value = specService.getValueRef<uint32_t>(parameterSpec, parameters, locationIndex);
            auto& origValue = specService.getValueRef<uint32_t>(parameterSpec, origParameters, locationIndex);
            AlienImGui::ColorButtonWithPicker(
                AlienImGui::ColorButtonWithPickerParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(origValue), value);
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._type)) {
            auto switcherSpec = std::get<AlternativeSpec>(parameterSpec._type);
            auto& value = specService.getValueRef<int>(parameterSpec, parameters, locationIndex);
            auto& origValue = specService.getValueRef<int>(parameterSpec, origParameters, locationIndex);
            std::vector<std::string> values;
            values.reserve(switcherSpec._alternatives.size());
            for (auto const& name : switcherSpec._alternatives | std::views::keys) {
                values.emplace_back(name);
            }
            AlienImGui::Switcher(
                AlienImGui::SwitcherParameters()
                    .name(parameterSpec._name)
                    .textWidth(RightColumnWidth)
                    .defaultValue(origValue)
                    .values(values)
                    .tooltip(parameterSpec._tooltip),
                value);
            createWidgetsFromParameterSpecs(switcherSpec._alternatives.at(value).second, locationIndex, parameters, origParameters);
        }
    }
}

ParametersSpecGuiService::LocationType ParametersSpecGuiService::getLocationType(int locationIndex, SimulationParameters const& parameters) const
{
    if (locationIndex == 0) {
        return LocationType::Base;
    } else  {
        for (int i = 0; i < parameters.numZones; ++i) {
            if (parameters.zone[i].locationIndex == locationIndex) {
                return LocationType::Zone;
            }
        }
        for (int i = 0; i < parameters.numRadiationSources; ++i) {
            if (parameters.radiationSource[i].locationIndex == locationIndex) {
                return LocationType::Source;
            }
        }
    }
    CHECK(false);
}

bool ParametersSpecGuiService::isVisible(ParameterGroupSpec const& groupSpec, LocationType locationType) const
{
    return std::any_of(groupSpec._parameters.begin(), groupSpec._parameters.end(), [&](auto const& parameterSpec) {
        return isVisible(parameterSpec, locationType);
    });
}

bool ParametersSpecGuiService::isVisible(ParameterSpec const& parameterSpec, LocationType locationType) const
{
    switch (locationType) {
    case LocationType::Base:
        return parameterSpec._visibleInBase;
    case LocationType::Zone:
        return parameterSpec._visibleInZone;
    case LocationType::Source:
        return parameterSpec._visibleInSource;
    }
    CHECK(false);
}
