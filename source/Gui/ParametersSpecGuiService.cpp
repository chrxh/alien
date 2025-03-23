#include "ParametersSpecGuiService.h"

#include <algorithm>

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
    auto& specService = SimulationParametersSpecificationService::get();
    auto locationType = getLocationType(locationIndex, parameters);

    for (auto const& groupSpec : parametersSpecs._groups) {
        if (!isVisible(groupSpec, locationType)) {
            continue;
        }
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name(groupSpec._name))) {
            for (auto const& parameterSpec : groupSpec._parameters) {
                if (!isVisible(parameterSpec, locationType)) {
                    continue;
                }
                if (std::holds_alternative<FloatSpec>(parameterSpec._type)) {
                    auto const& floatSpec = std::get<FloatSpec>(parameterSpec._type);
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
                            .tooltip(parameterSpec._tooltip),
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
                        AlienImGui::ColorButtonWithPickerParameters()
                            .name("Background color")
                            .textWidth(RightColumnWidth).defaultValue(origValue),
                        value);

                    //AlienImGui::InputText(
                    //    AlienImGui::InputTextParameters().name(parameter._name).textWidth(RightColumnWidth).defaultValue(origValue).tooltip(parameter._tooltip),
                    //    value,
                    //    sizeof(Char64) / sizeof(char));
                }
            }
        }
        AlienImGui::EndTreeNode();
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
