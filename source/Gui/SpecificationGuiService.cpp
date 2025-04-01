#include "SpecificationGuiService.h"

#include <algorithm>
#include <ranges>

#include <boost/range/adaptors.hpp>

#include "EngineInterface/SpecificationService.h"
#include "EngineInterface/SimulationParametersTypes.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SpecificationEvaluationService.h"

#include "AlienImGui.h"

namespace
{
    auto constexpr RightColumnWidth = 285.0f;
}

void SpecificationGuiService::createWidgetsForParameters(SimulationParameters& parameters, SimulationParameters& origParameters, int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& parametersSpecs = SpecificationService::get().getSpec();
    auto locationType = LocationHelper::getLocationType(locationIndex, parameters);

    for (auto const& groupSpec : parametersSpecs._groups) {
        if (!isVisible(groupSpec, locationType)) {
            continue;
        }
        auto isExpertSettings = groupSpec._expertToggle != nullptr;
        auto isGroupVisibleActive = true;
        auto name = groupSpec._name;
        if (isExpertSettings) {
            isGroupVisibleActive = *evaluationService.getExpertToggleRef(groupSpec._expertToggle, parameters);
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

void SpecificationGuiService::createWidgetsForExpertToggles(SimulationParameters& parameters, SimulationParameters& origParameters) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& parametersSpecs = SpecificationService::get().getSpec();

    for (auto const& groupSpec : parametersSpecs._groups) {
        if (groupSpec._expertToggle) {
            auto expertToggleValue = evaluationService.getExpertToggleRef(groupSpec._expertToggle, parameters);
            auto origExpertToggleValue = evaluationService.getExpertToggleRef(groupSpec._expertToggle, origParameters);
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

void SpecificationGuiService::createWidgetsFromParameterSpecs(
    std::vector<ParameterSpec> const& parameterSpecs,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto locationType = LocationHelper::getLocationType(locationIndex, parameters);

    for (auto const& [index, parameterSpec] : parameterSpecs | boost::adaptors::indexed(0)) {
        if (!isVisible(parameterSpec, locationType)) {
            continue;
        }
        ImGui::PushID(toInt(index));

        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            createWidgetsForBoolSpec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
            createWidgetsForIntSpec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            createWidgetsForFloatSpec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            createWidgetsForChar64Spec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            createWidgetsForAlternativeSpec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._reference)) {
            createWidgetsForColorPickerSpec(parameterSpec, parameters, origParameters, locationIndex);

        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            createWidgetsForColorTransitionRulesSpec(parameterSpec, parameters, origParameters, locationIndex);
        }

        ImGui::PopID();
    }
}

void SpecificationGuiService::createWidgetsForBoolSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& boolSpec = std::get<BoolSpec>(parameterSpec._reference);

    if (std::holds_alternative<ColorMatrixBoolMember>(boolSpec._member)) {

        auto value = reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(evaluationService.getBoolRef(boolSpec._member, parameters, locationIndex));
        auto origValue = reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(evaluationService.getBoolRef(boolSpec._member, origParameters, locationIndex));
        AlienImGui::CheckboxColorMatrix(
            AlienImGui::CheckboxColorMatrixParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*origValue))
                .tooltip(parameterSpec._tooltip),
            *value);

    } else {

        auto value = evaluationService.getBoolRef(boolSpec._member, parameters, locationIndex);
        auto origValue = evaluationService.getBoolRef(boolSpec._member, origParameters, locationIndex);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origValue).tooltip(parameterSpec._tooltip),
            *value);

    }
}

void SpecificationGuiService::createWidgetsForIntSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& intSpec = std::get<IntSpec>(parameterSpec._reference);

    if (std::holds_alternative<ColorMatrixIntMember>(intSpec._member)) {

        auto value = reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(evaluationService.getIntRef(intSpec._member, parameters, locationIndex));
        auto origValue = reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(evaluationService.getIntRef(intSpec._member, origParameters, locationIndex));
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

        auto value = evaluationService.getIntRef(intSpec._member, parameters, locationIndex);
        auto origValue = evaluationService.getIntRef(intSpec._member, origParameters, locationIndex);
        auto enabledValue = evaluationService.getEnabledRef(parameterSpec._enabled, parameters, locationIndex);
        auto origEnabledValue = evaluationService.getEnabledRef(parameterSpec._enabled, origParameters, locationIndex);
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
                .colorDependence(std::holds_alternative<ColorVectorIntMember>(intSpec._member)),
            value,
            enabledValue);
    }
}

void SpecificationGuiService::createWidgetsForFloatSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);

    if (std::holds_alternative<ColorMatrixFloatMember>(floatSpec._member) || std::holds_alternative<ColorMatrixFloatZoneValuesMember>(floatSpec._member)) {

        auto value = reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(evaluationService.getFloatRef(floatSpec._member, parameters, locationIndex));
        auto origValue = reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(evaluationService.getFloatRef(floatSpec._member, origParameters, locationIndex));
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

    } else if (std::holds_alternative<FloatGetterSetter>(floatSpec._member)) {

        auto [getter, setter] = std::get<FloatGetterSetter>(floatSpec._member);
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
        auto value = evaluationService.getFloatRef(floatSpec._member, parameters, locationIndex);
        auto origValue = evaluationService.getFloatRef(floatSpec._member, origParameters, locationIndex);
        auto enabledValue = evaluationService.getEnabledRef(parameterSpec._enabled, parameters, locationIndex);
        auto origEnabledValue = evaluationService.getEnabledRef(parameterSpec._enabled, origParameters, locationIndex);
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
                    std::holds_alternative<ColorVectorFloatMember>(floatSpec._member)
                    || std::holds_alternative<ColorVectorFloatZoneValuesMember>(floatSpec._member))
                .tooltip(parameterSpec._tooltip),
            value,
            enabledValue,
            pinnedValue);
    }
}

void SpecificationGuiService::createWidgetsForChar64Spec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& char64Spec = std::get<Char64Spec>(parameterSpec._reference);

    auto value = evaluationService.getChar64Ref(char64Spec._member, parameters, locationIndex);
    auto origValue = evaluationService.getChar64Ref(char64Spec._member, origParameters, locationIndex);
    AlienImGui::InputText(
        AlienImGui::InputTextParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(origValue).tooltip(parameterSpec._tooltip),
        value,
        sizeof(Char64) / sizeof(char));
}

void SpecificationGuiService::createWidgetsForAlternativeSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto alternativeSpec = std::get<AlternativeSpec>(parameterSpec._reference);

    auto value = evaluationService.getAlternativeRef(alternativeSpec._member, parameters, locationIndex);
    auto origValue = evaluationService.getAlternativeRef(alternativeSpec._member, origParameters, locationIndex);
    std::vector<std::string> values;
    values.reserve(alternativeSpec._alternatives.size());
    for (auto const& name : alternativeSpec._alternatives | std::views::keys) {
        values.emplace_back(name);
    }
    AlienImGui::Switcher(
        AlienImGui::SwitcherParameters()
            .name(parameterSpec._name)
            .textWidth(RightColumnWidth)
            .defaultValue(*origValue)
            .values(values)
            .tooltip(parameterSpec._tooltip),
        *value);
    createWidgetsFromParameterSpecs(alternativeSpec._alternatives.at(*value).second, parameters, origParameters, locationIndex);
}

void SpecificationGuiService::createWidgetsForColorPickerSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& colorPickerSpec = std::get<ColorPickerSpec>(parameterSpec._reference);

    auto value = evaluationService.getFloatColorRGBRef(colorPickerSpec._member, parameters, locationIndex);
    auto origValue = evaluationService.getFloatColorRGBRef(colorPickerSpec._member, origParameters, locationIndex);
    
    AlienImGui::ColorButtonWithPicker(
        AlienImGui::ColorButtonWithPickerParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origValue), *value);
}

void SpecificationGuiService::createWidgetsForColorTransitionRulesSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);

    auto value = evaluationService.getColorTransitionRulesRef(colorTransitionRulesSpec._member, parameters, locationIndex);
    auto origValue = evaluationService.getColorTransitionRulesRef(colorTransitionRulesSpec._member, origParameters, locationIndex);
    for (int color = 0; color < MAX_COLORS; ++color) {
        ImGui::PushID(color);
        auto widgetParameters = AlienImGui::InputColorTransitionParameters()
                                    .textWidth(RightColumnWidth)
                                    .color(color)
                                    .defaultTargetColor(origValue->cellColorTransitionTargetColor[color])
                                    .defaultTransitionAge(origValue->cellColorTransitionDuration[color])
                                    .logarithmic(true)
                                    .infinity(true);
        if (0 == color) {
            widgetParameters.name(parameterSpec._name).tooltip(parameterSpec._tooltip);
        }
        AlienImGui::InputColorTransition(widgetParameters, color, value->cellColorTransitionTargetColor[color], value->cellColorTransitionDuration[color]);
        ImGui::PopID();
    }
}

bool SpecificationGuiService::isVisible(ParameterGroupSpec const& groupSpec, LocationType locationType) const
{
    return std::any_of(groupSpec._parameters.begin(), groupSpec._parameters.end(), [&](auto const& parameterSpec) {
        return isVisible(parameterSpec, locationType);
    });
}

bool SpecificationGuiService::isVisible(ParameterSpec const& parameterSpec, LocationType locationType) const
{
    if (locationType == LocationType::Base) {
        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            auto const& boolSpec = std::get<BoolSpec>(parameterSpec._reference);
            if (std::holds_alternative<BoolMember>(boolSpec._member) || std::holds_alternative<ColorMatrixBoolMember>(boolSpec._member)
                || std::holds_alternative<BoolZoneValuesMember>(boolSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
            auto const& intSpec = std::get<IntSpec>(parameterSpec._reference);
            if (std::holds_alternative<IntMember>(intSpec._member) || std::holds_alternative<ColorVectorIntMember>(intSpec._member)
                || std::holds_alternative<ColorMatrixIntMember>(intSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatMember>(floatSpec._member) || std::holds_alternative<ColorVectorFloatMember>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatMember>(floatSpec._member) || std::holds_alternative<FloatZoneValuesMember>(floatSpec._member)
                || std::holds_alternative<ColorVectorFloatZoneValuesMember>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatZoneValuesMember>(floatSpec._member)
                || std::holds_alternative<FloatGetterSetter>(floatSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            auto const& char64Spec = std::get<Char64Spec>(parameterSpec._reference);
            if (std::holds_alternative<Char64Member>(char64Spec._member)) {
                return true;
            }
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            auto const& alternativeSpec = std::get<AlternativeSpec>(parameterSpec._reference);
            if (std::holds_alternative<IntMember>(alternativeSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._reference)) {
            auto const& colorPickerSpec = std::get<ColorPickerSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatColorRGBZoneMember>(colorPickerSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorTransitionRulesZoneMember>(colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }
    if (locationType == LocationType::Zone) {
        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            auto const& boolSpec = std::get<BoolSpec>(parameterSpec._reference);
            if (std::holds_alternative<BoolZoneValuesMember>(boolSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorVectorFloatZoneValuesMember>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatZoneValuesMember>(floatSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._reference)) {
            auto const& colorPickerSpec = std::get<ColorPickerSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatColorRGBZoneMember>(colorPickerSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorTransitionRulesZoneMember>(colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }

    if (locationType == LocationType::Base) {
        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            auto const& boolSpec = std::get<BoolSpec>(parameterSpec._reference);
            if (std::holds_alternative<BoolMember>(boolSpec._member) || std::holds_alternative<ColorMatrixBoolMember>(boolSpec._member)
                || std::holds_alternative<BoolZoneValuesMemberNew>(boolSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
            auto const& intSpec = std::get<IntSpec>(parameterSpec._reference);
            if (std::holds_alternative<IntMemberNew>(intSpec._member) || std::holds_alternative<ColorVectorIntMemberNew>(intSpec._member)
                || std::holds_alternative<ColorMatrixIntMemberNew>(intSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatMemberNew>(floatSpec._member) || std::holds_alternative<ColorVectorFloatMemberNew>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatMemberNew>(floatSpec._member) || std::holds_alternative<FloatZoneValuesMemberNew>(floatSpec._member)
                || std::holds_alternative<ColorVectorFloatZoneValuesMemberNew>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatZoneValuesMemberNew>(floatSpec._member)
                || std::holds_alternative<FloatGetterSetter>(floatSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            auto const& char64Spec = std::get<Char64Spec>(parameterSpec._reference);
            if (std::holds_alternative<Char64MemberNew>(char64Spec._member)) {
                return true;
            }
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            auto const& alternativeSpec = std::get<AlternativeSpec>(parameterSpec._reference);
            if (std::holds_alternative<IntMemberNew>(alternativeSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._reference)) {
            auto const& colorPickerSpec = std::get<ColorPickerSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatColorRGBZoneMemberNew>(colorPickerSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorTransitionRulesZoneMemberNew>(colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }
    if (locationType == LocationType::Zone) {
        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            auto const& boolSpec = std::get<BoolSpec>(parameterSpec._reference);
            if (std::holds_alternative<BoolZoneValuesMemberNew>(boolSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorVectorFloatZoneValuesMemberNew>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatZoneValuesMemberNew>(floatSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._reference)) {
            auto const& colorPickerSpec = std::get<ColorPickerSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatColorRGBZoneMemberNew>(colorPickerSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorTransitionRulesZoneMemberNew>(colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }

    return false;
}
