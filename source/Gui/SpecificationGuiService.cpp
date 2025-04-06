#include "SpecificationGuiService.h"

#include <algorithm>
#include <ranges>

#include <boost/range/adaptors.hpp>

#include "EngineInterface/SpecificationService.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SimulationParametersTypes.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SpecificationEvaluationService.h"

#include "AlienImGui.h"

namespace
{
    auto constexpr RightColumnWidth = 285.0f;
}

void SpecificationGuiService::createWidgetsForParameters(
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    SimulationFacade const& simulationFacade,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& parametersSpecs = SpecificationService::get().getSpec();
    auto locationType = LocationHelper::getLocationType(locationIndex, parameters);

    for (auto const& groupSpec : parametersSpecs._groups) {
        if (!evaluationService.isVisible(groupSpec, locationType)) {
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
            createWidgetsFromParameterSpecs(groupSpec._parameters, parameters, origParameters, simulationFacade, locationIndex);
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
    SimulationFacade const& simulationFacade,
    int locationIndex) const
{
    auto locationType = LocationHelper::getLocationType(locationIndex, parameters);

    for (auto const& [index, parameterSpec] : parameterSpecs | boost::adaptors::indexed(0)) {
        if (!SpecificationEvaluationService::get().isVisible(parameterSpec, locationType)) {
            continue;
        }
        ImGui::PushID(toInt(index));

        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            createWidgetsForBoolSpec(parameterSpec, parameters, origParameters, locationIndex);
        } else if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
            createWidgetsForIntSpec(parameterSpec, parameters, origParameters, locationIndex);
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            createWidgetsForFloatSpec(parameterSpec, parameters, origParameters, locationIndex);
        } else if (std::holds_alternative<Float2Spec>(parameterSpec._reference)) {
            createWidgetsForFloat2Spec(parameterSpec, parameters, origParameters, simulationFacade, locationIndex);
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            createWidgetsForChar64Spec(parameterSpec, parameters, origParameters, locationIndex);
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            createWidgetsForAlternativeSpec(parameterSpec, parameters, origParameters, simulationFacade, locationIndex);
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

    auto ref = evaluationService.getRef(boolSpec._member, parameters, locationIndex);
    auto origRef = evaluationService.getRef(boolSpec._member, origParameters, locationIndex);
    if (std::holds_alternative<ColorMatrixBoolMember>(boolSpec._member)) {

        AlienImGui::CheckboxColorMatrix(
            AlienImGui::CheckboxColorMatrixParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(origRef.value)))
                .tooltip(parameterSpec._tooltip),
            *reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(ref.value));

    } else {
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origRef.value).tooltip(parameterSpec._tooltip),
            *ref.value);

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

    auto [value, baseValue, enabledValue, pinnedValue] = evaluationService.getRef(intSpec._member, parameters, locationIndex);
    auto [origValue, origBaseValue, origEnabledValue, origPinnedValue] = evaluationService.getRef(intSpec._member, origParameters, locationIndex);

    if (std::holds_alternative<ColorMatrixIntMember>(intSpec._member)) {
        AlienImGui::InputIntColorMatrix(
            AlienImGui::InputIntColorMatrixParameters()
                .name(parameterSpec._name)
                .max(intSpec._min)
                .max(intSpec._max)
                .logarithmic(intSpec._logarithmic)
                .textWidth(RightColumnWidth)
                .tooltip(parameterSpec._tooltip)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(origValue))),
            *reinterpret_cast<int(*)[MAX_COLORS][MAX_COLORS]>(value));

    } else {
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .min(intSpec._min)
                .max(intSpec._max)
                .logarithmic(intSpec._logarithmic)
                .infinity(intSpec._infinity)
                .disabledValue(baseValue)
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

    auto [value, baseValue, enabledValue, pinnedValue] = evaluationService.getRef(floatSpec._member, parameters, locationIndex);
    auto [origValue, origBaseValue, origEnabledValue, origPinnedValue] = evaluationService.getRef(floatSpec._member, origParameters, locationIndex);

    if (std::holds_alternative<ColorMatrixFloatMember>(floatSpec._member) || std::holds_alternative<ColorMatrixFloatBaseZoneMember>(floatSpec._member)) {

        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name(parameterSpec._name)
                .max(floatSpec._min)
                .max(floatSpec._max)
                .logarithmic(floatSpec._logarithmic)
                .format(floatSpec._format)
                .textWidth(RightColumnWidth)
                .tooltip(parameterSpec._tooltip)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(origValue)))
                .disabledValue(
                    baseValue != nullptr ? std::make_optional(toVector<MAX_COLORS, MAX_COLORS>(*reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(baseValue)))
                                         : std::optional<std::vector<std::vector<float>>>()),
            *reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(value),
            enabledValue);

    } else {
        float tempValue;
        float tempOrigValue;
        if (floatSpec._getterSetter.has_value()) {
            auto [getter, setter] = floatSpec._getterSetter.value();
            tempValue = getter(parameters, locationIndex);
            tempOrigValue = getter(origParameters, locationIndex);
            value = &tempValue;
            origValue = &tempOrigValue;
        }

        if(AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .min(floatSpec._min)
                .max(floatSpec._max)
                .logarithmic(floatSpec._logarithmic)
                .format(floatSpec._format)
                .infinity(floatSpec._infinity)
                .disabledValue(baseValue)
                .defaultValue(origValue)
                .defaultEnabledValue(origEnabledValue)
                .colorDependence(
                    std::holds_alternative<ColorVectorFloatMember>(floatSpec._member)
                    || std::holds_alternative<ColorVectorFloatBaseZoneMember >(floatSpec._member))
                .tooltip(parameterSpec._tooltip),
            value,
            enabledValue,
            pinnedValue)) {

            if (floatSpec._getterSetter.has_value()) {
                auto [getter, setter] = floatSpec._getterSetter.value();
                setter(tempValue, parameters, locationIndex);
            }
        }
    }
}

void SpecificationGuiService::createWidgetsForFloat2Spec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    SimulationFacade const& simulationFacade,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& float2Spec = std::get<Float2Spec>(parameterSpec._reference);

    auto [value, baseValue, enabledValue, pinnedValue] = evaluationService.getRef(float2Spec._member, parameters, locationIndex);
    auto [origValue, origBaseValue, origEnabledValue, origPinnedValue] = evaluationService.getRef(float2Spec._member, origParameters, locationIndex);

    RealVector2D min = std::get<RealVector2D>(float2Spec._min);
    RealVector2D max = [&] {
        if (std::holds_alternative<WorldSize>(float2Spec._max)) {
            return toRealVector2D(simulationFacade->getWorldSize());
        } else {
            return std::get<RealVector2D>(float2Spec._max);
        }
    }();
    AlienImGui::SliderFloat2(
        AlienImGui::SliderFloat2Parameters()
            .name(parameterSpec._name)
            .textWidth(RightColumnWidth)
            .min(min)
            .max(max)
            .defaultValue(*origValue)
            .format("%.2f")
            //.getMousePickerEnabledFunc(getMousePickerEnabledFunc)
            //.setMousePickerEnabledFunc(setMousePickerEnabledFunc)
            //.getMousePickerPositionFunc(getMousePickerPositionFunc)
            .tooltip(parameterSpec._tooltip),
        value->x,
        value->y);
}

void SpecificationGuiService::createWidgetsForChar64Spec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& char64Spec = std::get<Char64Spec>(parameterSpec._reference);

    auto [value, baseValue, enabledValue, pinnedValue] = evaluationService.getRef(char64Spec._member, parameters, locationIndex);
    auto [origValue, origBaseValue, origEnabledValue, origPinnedValue] = evaluationService.getRef(char64Spec._member, origParameters, locationIndex);

    AlienImGui::InputText(
        AlienImGui::InputTextParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(origValue).tooltip(parameterSpec._tooltip),
        value,
        sizeof(Char64) / sizeof(char));
}

void SpecificationGuiService::createWidgetsForAlternativeSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    SimulationFacade const& simulationFacade,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto alternativeSpec = std::get<AlternativeSpec>(parameterSpec._reference);

    auto [value, baseValue, enabledValue, pinnedValue] = evaluationService.getRef(alternativeSpec._member, parameters, locationIndex);
    auto [origValue, origBaseValue, origEnabledValue, origPinnedValue] = evaluationService.getRef(alternativeSpec._member, origParameters, locationIndex);

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
    createWidgetsFromParameterSpecs(alternativeSpec._alternatives.at(*value).second, parameters, origParameters, simulationFacade, locationIndex);
}

void SpecificationGuiService::createWidgetsForColorPickerSpec(
    ParameterSpec const& parameterSpec,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int locationIndex) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& colorPickerSpec = std::get<ColorPickerSpec>(parameterSpec._reference);

    auto [value, baseValue, enabledValue, pinnedValue] = evaluationService.getRef(colorPickerSpec._member, parameters, locationIndex);
    auto [origValue, origBaseValue, origEnabledValue, origPinnedValue] = evaluationService.getRef(colorPickerSpec._member, origParameters, locationIndex);
   
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

    auto [value, baseValue, enabledValue, pinnedValue] = evaluationService.getRef(colorTransitionRulesSpec._member, parameters, locationIndex);
    auto [origValue, origBaseValue, origEnabledValue, origPinnedValue] =
        evaluationService.getRef(colorTransitionRulesSpec._member, origParameters, locationIndex);

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
