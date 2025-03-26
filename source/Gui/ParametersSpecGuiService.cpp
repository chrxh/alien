#include "ParametersSpecGuiService.h"

#include <algorithm>
#include <ranges>

#include <boost/range/adaptors.hpp>

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
        auto isExpertSettings = groupSpec._expertSettingAddress.has_value();
        auto isGroupVisibleActive = true;
        if (isExpertSettings) {
            isGroupVisibleActive = specService.getExpertSettingsToggleRef(groupSpec, parameters);
        }
        ImGui::PushID(groupSpec._name.c_str());
        if (AlienImGui::BeginTreeNode(
                AlienImGui::TreeNodeParameters().name(groupSpec._name).visible(isGroupVisibleActive).blinkWhenActivated(isExpertSettings))) {
            createWidgetsFromParameterSpecs(groupSpec._parameters, locationIndex, parameters, origParameters);
        }
        ImGui::PopID();
        AlienImGui::EndTreeNode();
    }
}

namespace
{
    template <int numRows, int numCols, typename T>
    std::vector<std::vector<T>> toVector(T const v[numRows][numCols])
    {
        std::vector<std::vector<T>> result;
        for (int row = 0; row < numRows; ++row) {
            std::vector<T> rowVector;
            for (int col = 0; col < numCols; ++col) {
                rowVector.emplace_back(v[row][col]);
            }
            result.emplace_back(rowVector);
        }
        return result;
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

    for (auto const& [index, parameterSpec] : parameterSpecs | boost::adaptors::indexed(0)) {
        ImGui::PushID(toInt(index));
        if (!isVisible(parameterSpec, locationType)) {
            continue;
        }
        if (parameterSpec._colorDependence == ColorDependence::Matrix) {
            if (std::holds_alternative<FloatSpec>(parameterSpec._type)) {
                auto const& floatSpec = std::get<FloatSpec>(parameterSpec._type);
                auto& value = *reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(&specService.getParameterRef<float>(parameterSpec, parameters, locationIndex));
                auto& origValue = *reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(&specService.getParameterRef<float>(parameterSpec, origParameters, locationIndex));
                AlienImGui::InputFloatColorMatrix(
                    AlienImGui::InputFloatColorMatrixParameters()
                        .name(parameterSpec._name)
                        .max(floatSpec._min)
                        .max(floatSpec._max)
                        .logarithmic(floatSpec._logarithmic)
                        .format(floatSpec._format)
                        .textWidth(RightColumnWidth)
                        .tooltip(parameterSpec._tooltip)
                        .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origValue)),
                    value);
            } else if (std::holds_alternative<IntSpec>(parameterSpec._type)) {
                auto const& intSpec = std::get<IntSpec>(parameterSpec._type);
                auto& value = *reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(&specService.getParameterRef<int>(parameterSpec, parameters, locationIndex));
                auto& origValue = *reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(&specService.getParameterRef<int>(parameterSpec, origParameters, locationIndex));
                AlienImGui::InputIntColorMatrix(
                    AlienImGui::InputIntColorMatrixParameters()
                        .name(parameterSpec._name)
                        .max(intSpec._min)
                        .max(intSpec._max)
                        .logarithmic(intSpec._logarithmic)
                        .textWidth(RightColumnWidth)
                        .tooltip(parameterSpec._tooltip)
                        .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origValue)),
                    value);
            } else if (std::holds_alternative<BoolSpec>(parameterSpec._type)) {
                auto& value = *reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(&specService.getParameterRef<bool>(parameterSpec, parameters, locationIndex));
                auto& origValue = *reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(&specService.getParameterRef<bool>(parameterSpec, origParameters, locationIndex));
                AlienImGui::CheckboxColorMatrix(
                    AlienImGui::CheckboxColorMatrixParameters()
                        .name("Color transitions")
                        .textWidth(RightColumnWidth)
                        .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(origValue))
                        .tooltip(parameterSpec._tooltip),
                    value);
            }
        } else {
            if (std::holds_alternative<FloatSpec>(parameterSpec._type)) {
                auto const& floatSpec = std::get<FloatSpec>(parameterSpec._type);
                bool* pinned = floatSpec._pinnedAddress.has_value() ? &specService.getParameterRef<bool>(
                                                                          parameterSpec._visibleInBase,
                                                                          parameterSpec._visibleInZone,
                                                                          parameterSpec._visibleInSource,
                                                                          floatSpec._pinnedAddress.value(),
                                                                          parameters,
                                                                          locationIndex)
                                                                    : nullptr;
                if (parameterSpec._valueAddress.has_value()) {
                    auto& value = specService.getParameterRef<float>(parameterSpec, parameters, locationIndex);
                    auto& origValue = specService.getParameterRef<float>(parameterSpec, origParameters, locationIndex);

                    bool* enabledValue = nullptr;
                    bool* origEnabledValue = nullptr;
                    if (parameterSpec._enabledValueBaseAddress.has_value()) {
                        enabledValue = &specService.getBaseParameterRef<bool>(parameterSpec._enabledValueBaseAddress.value(), parameters);
                        origEnabledValue = &specService.getBaseParameterRef<bool>(parameterSpec._enabledValueBaseAddress.value(), origParameters);
                    }
                    AlienImGui::SliderFloat(
                        AlienImGui::SliderFloatParameters()
                            .name(parameterSpec._name)
                            .textWidth(RightColumnWidth)
                            .min(floatSpec._min)
                            .max(floatSpec._max)
                            .logarithmic(floatSpec._logarithmic)
                            .format(floatSpec._format)
                            .infinity(floatSpec._infinity)
                            .disabledValue(&value)
                            .defaultValue(&origValue)
                            .defaultEnabledValue(origEnabledValue)
                            .tooltip(parameterSpec._tooltip)
                            .colorDependence(parameterSpec._colorDependence == ColorDependence::Vector),
                        &value,
                        enabledValue,
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
                                .colorDependence(parameterSpec._colorDependence == ColorDependence::Vector),
                            &value,
                            nullptr,
                            pinned)) {

                        setter(value, parameters, locationIndex);
                    }
                }
            } else if (std::holds_alternative<IntSpec>(parameterSpec._type)) {
                auto const& intSpec = std::get<IntSpec>(parameterSpec._type);
                auto& value = specService.getParameterRef<int>(parameterSpec, parameters, locationIndex);
                auto& origValue = specService.getParameterRef<int>(parameterSpec, origParameters, locationIndex);
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
                        .colorDependence(parameterSpec._colorDependence == ColorDependence::Vector),
                    &value);
            } else if (std::holds_alternative<BoolSpec>(parameterSpec._type)) {
                auto& value = specService.getParameterRef<bool>(parameterSpec, parameters, locationIndex);
                auto& origValue = specService.getParameterRef<bool>(parameterSpec, origParameters, locationIndex);
                AlienImGui::Checkbox(
                    AlienImGui::CheckboxParameters()
                        .name(parameterSpec._name)
                        .textWidth(RightColumnWidth)
                        .defaultValue(origValue)
                        .tooltip(parameterSpec._tooltip),
                    value);
            } else if (std::holds_alternative<Char64Spec>(parameterSpec._type)) {
                auto& value = specService.getParameterRef<Char64>(parameterSpec, parameters, locationIndex);
                auto& origValue = specService.getParameterRef<Char64>(parameterSpec, origParameters, locationIndex);
                AlienImGui::InputText(
                    AlienImGui::InputTextParameters()
                        .name(parameterSpec._name)
                        .textWidth(RightColumnWidth)
                        .defaultValue(origValue)
                        .tooltip(parameterSpec._tooltip),
                    value,
                    sizeof(Char64) / sizeof(char));
            } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._type)) {
                auto& value = specService.getParameterRef<uint32_t>(parameterSpec, parameters, locationIndex);
                auto& origValue = specService.getParameterRef<uint32_t>(parameterSpec, origParameters, locationIndex);
                AlienImGui::ColorButtonWithPicker(
                    AlienImGui::ColorButtonWithPickerParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(origValue), value);
            } else if (std::holds_alternative<ColorTransitionSpec>(parameterSpec._type)) {
                auto colorTransitionSpec = std::get<ColorTransitionSpec>(parameterSpec._type);
                auto targetColor = &specService.getParameterRef<int>(parameterSpec, parameters, locationIndex);
                auto transitionDuration = &specService.getParameterRef<int>(
                    parameterSpec._visibleInBase,
                    parameterSpec._visibleInZone,
                    parameterSpec._visibleInSource,
                    colorTransitionSpec._transitionDurationAddress.value(),
                    parameters,
                    locationIndex);
                auto origTargetColor = &specService.getParameterRef<int>(parameterSpec, origParameters, locationIndex);
                auto origTransitionDuration = &specService.getParameterRef<int>(
                    parameterSpec._visibleInBase,
                    parameterSpec._visibleInZone,
                    parameterSpec._visibleInSource,
                    colorTransitionSpec._transitionDurationAddress.value(),
                    origParameters,
                    locationIndex);
                for (int color = 0; color < MAX_COLORS; ++color) {
                    ImGui::PushID(color);
                    auto widgetParameters = AlienImGui::InputColorTransitionParameters()
                                                .textWidth(RightColumnWidth)
                                                .color(color)
                                                .defaultTargetColor(origTargetColor[color])
                                                .defaultTransitionAge(origTransitionDuration[color])
                                                .logarithmic(true)
                                                .infinity(true);
                    if (0 == color) {
                        widgetParameters.name(parameterSpec._name).tooltip(parameterSpec._tooltip);
                    }
                    AlienImGui::InputColorTransition(widgetParameters, color, targetColor[color], transitionDuration[color]);
                    ImGui::PopID();
                }
            } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._type)) {
                auto switcherSpec = std::get<AlternativeSpec>(parameterSpec._type);
                auto& value = specService.getParameterRef<int>(parameterSpec, parameters, locationIndex);
                auto& origValue = specService.getParameterRef<int>(parameterSpec, origParameters, locationIndex);
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
        ImGui::PopID();
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
