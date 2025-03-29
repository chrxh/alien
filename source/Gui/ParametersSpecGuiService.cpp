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

void ParametersSpecGuiService::createWidgetsForParameters(SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto const& parametersSpecs = specService.getSpec();
    auto locationType = getLocationType(locationIndex, parameters);

    for (auto const& groupSpec : parametersSpecs._groups) {
        if (!isVisible(groupSpec, locationType)) {
            continue;
        }
        auto isExpertSettings = groupSpec._expertToggleAddress.has_value();
        auto isGroupVisibleActive = true;
        auto name = groupSpec._name;
        if (isExpertSettings) {
            isGroupVisibleActive = *specService.getExpertToggleValueRef(groupSpec, parameters);
            name = "Expert settings: " + name;
        }
        ImGui::PushID(name.c_str());
        if (AlienImGui::BeginTreeNode(AlienImGui::TreeNodeParameters().name(name).visible(isGroupVisibleActive).blinkWhenActivated(isExpertSettings))) {
            createWidgetsFromParameterSpecs(groupSpec._parameters, parameters, origParameters, locationIndex);
        }
        ImGui::PopID();
        AlienImGui::EndTreeNode();
    }
}

void ParametersSpecGuiService::createWidgetsForExpertToggles(SimulationParameters& parameters, SimulationParameters& origParameters) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto const& parametersSpecs = specService.getSpec();

    for (auto const& groupSpec : parametersSpecs._groups) {
        if (groupSpec._expertToggleAddress.has_value()) {
            auto expertToggleValue = specService.getExpertToggleValueRef(groupSpec, parameters);
            auto origExpertToggleValue = specService.getExpertToggleValueRef(groupSpec, origParameters);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name(groupSpec._name)
                    .textWidth(0)
                    .defaultValue(*origExpertToggleValue)
                    .tooltip(groupSpec._tooltip),
                *expertToggleValue);
        }
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
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto locationType = getLocationType(locationIndex, parameters);

    for (auto const& [index, parameterSpec] : parameterSpecs | boost::adaptors::indexed(0)) {
        if (!isVisible(parameterSpec, locationType)) {
            continue;
        }
        ImGui::PushID(toInt(index));

        if (std::holds_alternative<BoolSpec>(parameterSpec._type)) {
            createWidgetsForBoolValues(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<IntSpec>(parameterSpec._type)) {
            createWidgetsForIntValues(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<FloatSpec>(parameterSpec._type)) {
            createWidgetsForFloatValues(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<Char64Spec>(parameterSpec._type)) {
            createWidgetsForChar64Values(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._type)) {
            createWidgetsForFloatColorRGBValues(parameterSpec, parameters, origParameters, locationIndex);
        }

        //if (parameterSpec._colorDependence == ColorDependence::Matrix) {
        //    if (std::holds_alternative<FloatSpec>(parameterSpec._type)) {
        //    } else if (std::holds_alternative<IntSpec>(parameterSpec._type)) {
        //    } else if (std::holds_alternative<BoolSpec>(parameterSpec._type)) {
        //    }
        //} else {
        //    if (std::holds_alternative<FloatSpec>(parameterSpec._type)) {
        //    } else if (std::holds_alternative<IntSpec>(parameterSpec._type)) {
        //    } else if (std::holds_alternative<BoolSpec>(parameterSpec._type)) {
        //    } else if (std::holds_alternative<Char64Spec>(parameterSpec._type)) {
        //    } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._type)) {
        //        auto value = specService.getValueRef<uint32_t>(parameterSpec._value, parameters, locationIndex);
        //        auto origValue = specService.getValueRef<uint32_t>(parameterSpec._value, origParameters, locationIndex);
        //        AlienImGui::ColorButtonWithPicker(
        //            AlienImGui::ColorButtonWithPickerParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origValue), *value);
        //    } else if (std::holds_alternative<ColorTransitionSpec>(parameterSpec._type)) {
        //        auto value = specService.getValueRef<ColorTransitionRules>(parameterSpec._value, parameters, locationIndex);
        //        auto origValue = specService.getValueRef<ColorTransitionRules>(parameterSpec._value, origParameters, locationIndex);
        //        for (int color = 0; color < MAX_COLORS; ++color) {
        //            ImGui::PushID(color);
        //            auto widgetParameters = AlienImGui::InputColorTransitionParameters()
        //                                        .textWidth(RightColumnWidth)
        //                                        .color(color)
        //                                        .defaultTargetColor(origValue->cellColorTransitionTargetColor[color])
        //                                        .defaultTransitionAge(origValue->cellColorTransitionDuration[color])
        //                                        .logarithmic(true)
        //                                        .infinity(true);
        //            if (0 == color) {
        //                widgetParameters.name(parameterSpec._name).tooltip(parameterSpec._tooltip);
        //            }
        //            AlienImGui::InputColorTransition(
        //                widgetParameters, color, value->cellColorTransitionTargetColor[color], value->cellColorTransitionDuration[color]);
        //            ImGui::PopID();
        //        }
        //    } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._type)) {
        //        auto switcherSpec = std::get<AlternativeSpec>(parameterSpec._type);
        //        auto value = specService.getValueRef<int>(parameterSpec._value, parameters, locationIndex);
        //        auto origValue = specService.getValueRef<int>(parameterSpec._value, origParameters, locationIndex);
        //        std::vector<std::string> values;
        //        values.reserve(switcherSpec._alternatives.size());
        //        for (auto const& name : switcherSpec._alternatives | std::views::keys) {
        //            values.emplace_back(name);
        //        }
        //        AlienImGui::Switcher(
        //            AlienImGui::SwitcherParameters()
        //                .name(parameterSpec._name)
        //                .textWidth(RightColumnWidth)
        //                .defaultValue(*origValue)
        //                .values(values)
        //                .tooltip(parameterSpec._tooltip),
        //            *value);
        //        createWidgetsFromParameterSpecs(switcherSpec._alternatives.at(*value).second, locationIndex, parameters, origParameters);
        //    }
        //}
        ImGui::PopID();
    }
}

void ParametersSpecGuiService::createWidgetsForBoolValues(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();

    if (std::holds_alternative<ColorMatrixBoolMember>(parameterSpec._member)) {

        auto value = reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(specService.getBoolRef(parameterSpec._member, parameters, locationIndex));
        auto origValue = reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(specService.getBoolRef(parameterSpec._member, origParameters, locationIndex));
        AlienImGui::CheckboxColorMatrix(
            AlienImGui::CheckboxColorMatrixParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*origValue))
                .tooltip(parameterSpec._tooltip),
            *value);

    } else {

        auto value = specService.getBoolRef(parameterSpec._member, parameters, locationIndex);
        auto origValue = specService.getBoolRef(parameterSpec._member, origParameters, locationIndex);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origValue).tooltip(parameterSpec._tooltip),
            *value);

    }
}

void ParametersSpecGuiService::createWidgetsForIntValues(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto const& intSpec = std::get<IntSpec>(parameterSpec._type);

    if (std::holds_alternative<ColorMatrixIntMember>(parameterSpec._member)) {

        auto const& intSpec = std::get<IntSpec>(parameterSpec._type);
        auto value = reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(specService.getIntRef(parameterSpec._member, parameters, locationIndex));
        auto origValue = reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(specService.getIntRef(parameterSpec._member, origParameters, locationIndex));
        AlienImGui::InputIntColorMatrix(
            AlienImGui::InputIntColorMatrixParameters()
                .name(parameterSpec._name)
                .max(intSpec._min)
                .max(intSpec._max)
                .logarithmic(intSpec._logarithmic)
                .textWidth(RightColumnWidth)
                .tooltip(parameterSpec._tooltip)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*origValue)),
            *value);

    } else {

        auto value = specService.getIntRef(parameterSpec._member, parameters, locationIndex);
        auto origValue = specService.getIntRef(parameterSpec._member, origParameters, locationIndex);
        auto enabledValue = nullptr;
        //specService.getEnabledValueRef(parameterSpec._value, parameters, locationIndex);
        auto origEnabledValue = nullptr;
        //specService.getEnabledValueRef(parameterSpec._value, origParameters, locationIndex);
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .min(intSpec._min)
                .max(intSpec._max)
                .logarithmic(intSpec._logarithmic)
                .infinity(intSpec._infinity)
                .disabledValue(value)
                .defaultValue(origValue)
                .defaultEnabledValue(origEnabledValue)
                .tooltip(parameterSpec._tooltip)
                .colorDependence(std::holds_alternative<ColorVectorFloatMember>(parameterSpec._member)),
            value,
            enabledValue);
    }
}

void ParametersSpecGuiService::createWidgetsForFloatValues(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto const& floatSpec = std::get<FloatSpec>(parameterSpec._type);

    if (std::holds_alternative<ColorMatrixFloatMember>(parameterSpec._member)
        || std::holds_alternative<ColorMatrixFloatZoneValuesMember>(parameterSpec._member)) {

        auto value = reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(specService.getFloatRef(parameterSpec._member, parameters, locationIndex));
        auto origValue = reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(specService.getFloatRef(parameterSpec._member, origParameters, locationIndex));
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name(parameterSpec._name)
                .max(floatSpec._min)
                .max(floatSpec._max)
                .logarithmic(floatSpec._logarithmic)
                .format(floatSpec._format)
                .textWidth(RightColumnWidth)
                .tooltip(parameterSpec._tooltip)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*origValue)),
            *value);

    } else if (std::holds_alternative<FloatGetterSetter>(parameterSpec._member)) {

        auto [getter, setter] = std::get<FloatGetterSetter>(parameterSpec._member);
        auto value = getter(parameters, locationIndex);
        auto origValue = getter(origParameters, locationIndex);
        bool* pinnedValue = nullptr;
        //specService.getPinnedValueRef(parameterSpec._value, parameters, locationIndex);

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
                pinnedValue)) {

            setter(value, parameters, locationIndex);
        }

    } else {

        bool* pinnedValue = nullptr;
        //specService.getPinnedValueRef(parameterSpec._value, parameters, locationIndex);
        auto value = specService.getFloatRef(parameterSpec._member, parameters, locationIndex);
        auto origValue = specService.getFloatRef(parameterSpec._member, origParameters, locationIndex);
        auto enabledValue = nullptr;
        //specService.getEnabledValueRef(parameterSpec._value, parameters, locationIndex);
        auto origEnabledValue = nullptr;
        //specService.getEnabledValueRef(parameterSpec._value, origParameters, locationIndex);
        AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .min(floatSpec._min)
                .max(floatSpec._max)
                .logarithmic(floatSpec._logarithmic)
                .format(floatSpec._format)
                .infinity(floatSpec._infinity)
                .disabledValue(value)
                .defaultValue(origValue)
                .defaultEnabledValue(origEnabledValue)
                .colorDependence(std::holds_alternative<ColorVectorFloatMember>(parameterSpec._member))
                .tooltip(parameterSpec._tooltip),
            value,
            enabledValue,
            pinnedValue);
    }
}

void ParametersSpecGuiService::createWidgetsForChar64Values(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();

    auto value = specService.getChar64Ref(parameterSpec._member, parameters, locationIndex);
    auto origValue = specService.getChar64Ref(parameterSpec._member, origParameters, locationIndex);
    AlienImGui::InputText(
        AlienImGui::InputTextParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origValue).tooltip(parameterSpec._tooltip),
        *value,
        sizeof(Char64) / sizeof(char));
}

void ParametersSpecGuiService::createWidgetsForFloatColorRGBValues(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();

    auto value = specService.getFloatColorRGBRef(parameterSpec._member, parameters, locationIndex);
    auto origValue = specService.getFloatColorRGBRef(parameterSpec._member, origParameters, locationIndex);
    
    AlienImGui::ColorButtonWithPicker(
        AlienImGui::ColorButtonWithPickerParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origValue), *value);
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
    if (locationType == LocationType::Base) {
        if (std::holds_alternative<BoolMember>(parameterSpec._member) || std::holds_alternative<IntMember>(parameterSpec._member)
            || std::holds_alternative<FloatMember>(parameterSpec._member) || std::holds_alternative<ColorVectorIntMember>(parameterSpec._member)
            || std::holds_alternative<ColorVectorFloatMember>(parameterSpec._member) || std::holds_alternative<ColorMatrixBoolMember>(parameterSpec._member)
            || std::holds_alternative<ColorMatrixIntMember>(parameterSpec._member) || std::holds_alternative<ColorMatrixFloatMember>(parameterSpec._member)
            || std::holds_alternative<BoolZoneValuesMember>(parameterSpec._member) || std::holds_alternative<FloatZoneValuesMember>(parameterSpec._member)
            || std::holds_alternative<ColorVectorFloatZoneMember>(parameterSpec._member)
            || std::holds_alternative<ColorMatrixFloatZoneValuesMember>(parameterSpec._member)
            || std::holds_alternative<ColorTransitionRulesMember>(parameterSpec._member) || std::holds_alternative<FloatGetterSetter>(parameterSpec._member)
            || std::holds_alternative<Char64Member>(parameterSpec._member) || std::holds_alternative<FloatColorRGBZoneMember>(parameterSpec._member)) {
            return true;
        }
    }
    if (locationType == LocationType::Zone) {
        if (std::holds_alternative<BoolZoneValuesMember>(parameterSpec._member) || std::holds_alternative<FloatZoneValuesMember>(parameterSpec._member)
            || std::holds_alternative<ColorVectorFloatZoneMember>(parameterSpec._member)
            || std::holds_alternative<ColorMatrixFloatZoneValuesMember>(parameterSpec._member)
            || std::holds_alternative<ColorTransitionRulesMember>(parameterSpec._member)
            || std::holds_alternative<FloatColorRGBZoneMember>(parameterSpec._member)) {
            return true;
        }
    }
    return false;
}
