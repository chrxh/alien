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

    for (auto const& group : parametersSpecs._groups) {
        if (!isVisible(group, locationType)) {
            continue;
        }
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name(group._name))) {
            for (auto const& parameterOrAlternative : group._parameters) {
                if (!isVisible(parameterOrAlternative, locationType)) {
                    continue;
                }
                if (std::holds_alternative<ParameterSpec>(parameterOrAlternative)) {
                    auto const& parameter = std::get<ParameterSpec>(parameterOrAlternative);
                    if (std::holds_alternative<FloatParameterSpec>(parameter)) {
                        auto const& floatParameter = std::get<FloatParameterSpec>(parameter);
                        auto& value = specService.getValueRef(floatParameter, parameters, locationIndex);
                        auto& origValue = specService.getValueRef(floatParameter, origParameters, locationIndex);
                        AlienImGui::SliderFloat(
                            AlienImGui::SliderFloatParameters()
                                .name(floatParameter._name)
                                .textWidth(RightColumnWidth)
                                .min(floatParameter._min)
                                .max(floatParameter._max)
                                .logarithmic(floatParameter._logarithmic)
                                .format(floatParameter._format)
                                .infinity(floatParameter._infinity)
                                .defaultValue(&origValue)
                                .tooltip(floatParameter._tooltip),
                            &value);
                    } else if (std::holds_alternative<BoolParameterSpec>(parameter)) {
                        auto const& boolParameter = std::get<BoolParameterSpec>(parameter);
                        auto& value = specService.getValueRef(boolParameter, parameters, locationIndex);
                        auto& origValue = specService.getValueRef(boolParameter, origParameters, locationIndex);
                        AlienImGui::Checkbox(
                            AlienImGui::CheckboxParameters()
                                .name(boolParameter._name)
                                .textWidth(RightColumnWidth)
                                .defaultValue(origValue)
                                .tooltip(boolParameter._tooltip),
                            value);
                    } else if (std::holds_alternative<Char64ParameterSpec>(parameter)) {
                        auto const& char64_parameter = std::get<Char64ParameterSpec>(parameter);
                        auto& value = specService.getValueRef(char64_parameter, parameters, locationIndex);
                        auto& origValue = specService.getValueRef(char64_parameter, origParameters, locationIndex);
                        AlienImGui::InputText(
                            AlienImGui::InputTextParameters()
                                .name(char64_parameter._name)
                                .textWidth(RightColumnWidth)
                                .defaultValue(origValue)
                                .tooltip(char64_parameter._tooltip),
                            value,
                            sizeof(Char64) / sizeof(char));
                    }
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
    return std::any_of(groupSpec._parameters.begin(), groupSpec._parameters.end(), [&](auto const& parameterOrAlternative) {
        return isVisible(parameterOrAlternative, locationType);
    });
}

bool ParametersSpecGuiService::isVisible(ParameterOrAlternativeSpec const& parameterOrAlternativeSpec, LocationType locationType) const
{
    auto isVisible = [&locationType](auto const& parameterOrAlternative) {
        switch (locationType) {
        case LocationType::Base:
            return parameterOrAlternative._visibleInBase;
        case LocationType::Zone:
            return parameterOrAlternative._visibleInZone;
        case LocationType::Source:
            return parameterOrAlternative._visibleInSource;
        }
        CHECK(false);
    };

    if (std::holds_alternative<ParameterSpec>(parameterOrAlternativeSpec)) {
        auto const& parameter = std::get<ParameterSpec>(parameterOrAlternativeSpec);
        if (std::holds_alternative<FloatParameterSpec>(parameter)) {
            return isVisible(std::get<FloatParameterSpec>(parameter));
        } else if (std::holds_alternative<BoolParameterSpec>(parameter)) {
            return isVisible(std::get<BoolParameterSpec>(parameter));
        } else if (std::holds_alternative<Char64ParameterSpec>(parameter)) {
            return isVisible(std::get<Char64ParameterSpec>(parameter));
        } else {
            CHECK(false);
        }
    } else {
        return isVisible(std::get<ParameterAlternativeSpec>(parameterOrAlternativeSpec));
    }
}
