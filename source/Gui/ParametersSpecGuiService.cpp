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

        if (boost::get<BoolSpec>(&parameterSpec._reference)) {
            createWidgetsForBoolSpec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (boost::get<IntSpec>(&parameterSpec._reference)) {
            createWidgetsForIntSpec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (boost::get<FloatSpec>(&parameterSpec._reference)) {
            createWidgetsForFloatSpec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (boost::get<Char64Spec>(&parameterSpec._reference)) {
            createWidgetsForChar64Spec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (boost::get<ColorPickerSpec>(&parameterSpec._reference)) {
            createWidgetsForColorPickerSpec(parameterSpec, parameters, origParameters, locationIndex);
        }

        //if (parameterSpec._colorDependence == ColorDependence::Matrix) {
        //    if (boost::get<FloatSpec>(parameterSpec._type)) {
        //    } else if (boost::get<IntSpec>(parameterSpec._type)) {
        //    } else if (boost::get<BoolSpec>(parameterSpec._type)) {
        //    }
        //} else {
        //    if (boost::get<FloatSpec>(parameterSpec._type)) {
        //    } else if (boost::get<IntSpec>(parameterSpec._type)) {
        //    } else if (boost::get<BoolSpec>(parameterSpec._type)) {
        //    } else if (boost::get<Char64Spec>(parameterSpec._type)) {
        //    } else if (boost::get<ColorPickerSpec>(parameterSpec._type)) {
        //    } else if (boost::get<ColorTransitionSpec>(parameterSpec._type)) {
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
        //    } else if (boost::get<AlternativeSpec>(parameterSpec._type)) {
        //        auto switcherSpec = boost::get<AlternativeSpec>(parameterSpec._type);
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

void ParametersSpecGuiService::createWidgetsForBoolSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto const& boolSpec = boost::get<BoolSpec>(parameterSpec._reference);

    if (boost::get<ColorMatrixBoolMember>(&boolSpec._member)) {

        auto value = reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(specService.getBoolRef(boolSpec._member, parameters, locationIndex));
        auto origValue = reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(specService.getBoolRef(boolSpec._member, origParameters, locationIndex));
        AlienImGui::CheckboxColorMatrix(
            AlienImGui::CheckboxColorMatrixParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*origValue))
                .tooltip(parameterSpec._tooltip),
            *value);

    } else {

        auto value = specService.getBoolRef(boolSpec._member, parameters, locationIndex);
        auto origValue = specService.getBoolRef(boolSpec._member, origParameters, locationIndex);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origValue).tooltip(parameterSpec._tooltip),
            *value);

    }
}

void ParametersSpecGuiService::createWidgetsForIntSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto const& intSpec = boost::get<IntSpec>(parameterSpec._reference);

    if (boost::get<ColorMatrixIntMember>(&intSpec._member)) {

        auto value = reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(specService.getIntRef(intSpec._member, parameters, locationIndex));
        auto origValue = reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(specService.getIntRef(intSpec._member, origParameters, locationIndex));
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

        auto value = specService.getIntRef(intSpec._member, parameters, locationIndex);
        auto origValue = specService.getIntRef(intSpec._member, origParameters, locationIndex);
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
                .colorDependence(boost::get<ColorVectorIntMember>(&intSpec._member) != nullptr),
            value,
            enabledValue);
    }
}

void ParametersSpecGuiService::createWidgetsForFloatSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto const& floatSpec = boost::get<FloatSpec>(parameterSpec._reference);

    if (boost::get<ColorMatrixFloatMember>(&floatSpec._member) || boost::get<ColorMatrixFloatZoneValuesMember>(&floatSpec._member)) {

        auto value = reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(specService.getFloatRef(floatSpec._member, parameters, locationIndex));
        auto origValue = reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(specService.getFloatRef(floatSpec._member, origParameters, locationIndex));
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

    } else if (boost::get<FloatGetterSetter>(&floatSpec._member)) {

        auto [getter, setter] = boost::get<FloatGetterSetter>(floatSpec._member);
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
                    .tooltip(parameterSpec._tooltip),
                &value,
                nullptr,
                pinnedValue)) {

            setter(value, parameters, locationIndex);
        }

    } else {

        bool* pinnedValue = nullptr;
        //specService.getPinnedValueRef(parameterSpec._value, parameters, locationIndex);
        auto value = specService.getFloatRef(floatSpec._member, parameters, locationIndex);
        auto origValue = specService.getFloatRef(floatSpec._member, origParameters, locationIndex);
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
                .colorDependence(
                    boost::get<ColorVectorFloatMember>(&floatSpec._member)
                    || boost::get<ColorVectorFloatZoneValuesMember>(&floatSpec._member))
                .tooltip(parameterSpec._tooltip),
            value,
            enabledValue,
            pinnedValue);
    }
}

void ParametersSpecGuiService::createWidgetsForChar64Spec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto const& char64Spec = boost::get<Char64Spec>(parameterSpec._reference);

    auto value = specService.getChar64Ref(char64Spec._member, parameters, locationIndex);
    auto origValue = specService.getChar64Ref(char64Spec._member, origParameters, locationIndex);
    AlienImGui::InputText(
        AlienImGui::InputTextParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(origValue).tooltip(parameterSpec._tooltip),
        value,
        sizeof(Char64) / sizeof(char));
}

void ParametersSpecGuiService::createWidgetsForColorPickerSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& specService = SimulationParametersSpecificationService::get();
    auto const& colorPickerSpec = boost::get<ColorPickerSpec>(parameterSpec._reference);

    auto value = specService.getFloatColorRGBRef(colorPickerSpec._member, parameters, locationIndex);
    auto origValue = specService.getFloatColorRGBRef(colorPickerSpec._member, origParameters, locationIndex);
    
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
        if (boost::get<BoolSpec>(&parameterSpec._reference)) {
            auto const& boolSpec = boost::get<BoolSpec>(parameterSpec._reference);
            if (boost::get<BoolMember>(&boolSpec._member) || boost::get<ColorMatrixBoolMember>(&boolSpec._member)
                || boost::get<BoolZoneValuesMember>(&boolSpec._member)) {
                return true;
            }
        } else if (boost::get<IntSpec>(&parameterSpec._reference)) {
            auto const& intSpec = boost::get<IntSpec>(parameterSpec._reference);
            if (boost::get<IntMember>(&intSpec._member) || boost::get<ColorVectorIntMember>(&intSpec._member)
                || boost::get<ColorMatrixIntMember>(&intSpec._member)) {
                return true;
            }
        } else if (boost::get<FloatSpec>(&parameterSpec._reference)) {
            auto const& floatSpec = boost::get<FloatSpec>(parameterSpec._reference);
            if (boost::get<FloatMember>(&floatSpec._member) || boost::get<ColorVectorFloatMember>(&floatSpec._member)
                || boost::get<ColorMatrixFloatMember>(&floatSpec._member) || boost::get<FloatZoneValuesMember>(&floatSpec._member)
                || boost::get<ColorVectorFloatZoneValuesMember>(&floatSpec._member)
                || boost::get<ColorMatrixFloatZoneValuesMember>(&floatSpec._member)
                || boost::get<FloatGetterSetter>(&floatSpec._member)) {
                return true;
            }
        } else if (boost::get<Char64Spec>(&parameterSpec._reference)) {
            auto const& char64Spec = boost::get<Char64Spec>(parameterSpec._reference);
            if (boost::get<Char64Member>(&char64Spec._member)) {
                return true;
            }
        } else if (boost::get<AlternativeSpec>(&parameterSpec._reference)) {
            auto const& alternativeSpec = boost::get<AlternativeSpec>(parameterSpec._reference);
            if (boost::get<IntMember>(&(*alternativeSpec._member))) {
                return true;
            }
        } else if (boost::get<ColorPickerSpec>(&parameterSpec._reference)) {
            auto const& colorPickerSpec = boost::get<ColorPickerSpec>(parameterSpec._reference);
            if (boost::get<FloatColorRGBZoneMember>(&colorPickerSpec._member)) {
                return true;
            }
        } else if (boost::get<ColorTransitionRulesSpec>(&parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = boost::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (boost::get<ColorTransitionRulesZoneMember>(&colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }
    if (locationType == LocationType::Zone) {
        if (boost::get<BoolSpec>(&parameterSpec._reference)) {
            auto const& boolSpec = boost::get<BoolSpec>(parameterSpec._reference);
            if (boost::get<BoolZoneValuesMember>(&boolSpec._member)) {
                return true;
            }
        } else if (boost::get<FloatSpec>(&parameterSpec._reference)) {
            auto const& floatSpec = boost::get<FloatSpec>(parameterSpec._reference);
            if (boost::get<ColorVectorFloatZoneValuesMember>(&floatSpec._member)
                || boost::get<ColorMatrixFloatZoneValuesMember>(&floatSpec._member)) {
                return true;
            }
        } else if (boost::get<ColorPickerSpec>(&parameterSpec._reference)) {
            auto const& colorPickerSpec = boost::get<ColorPickerSpec>(parameterSpec._reference);
            if (boost::get<FloatColorRGBZoneMember>(&colorPickerSpec._member)) {
                return true;
            }
        } else if (boost::get<ColorTransitionRulesSpec>(&parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = boost::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (boost::get<ColorTransitionRulesZoneMember>(&colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }

    return false;
}
