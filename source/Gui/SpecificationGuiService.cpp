#include "SpecificationGuiService.h"

#include <algorithm>
#include <ranges>

#include <boost/range/adaptors.hpp>

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SimulationParametersTypes.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SpecificationEvaluationService.h"

#include "SimulationInteractionController.h"
#include "AlienImGui.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr RightColumnWidth = 285.0f;
}

void SpecificationGuiService::createWidgetsForParameters(
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    SimulationFacade const& simulationFacade,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& parametersSpecs = SimulationParameters::getSpec();
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

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
            createWidgetsForParameterGroup(groupSpec._parameters, true, parameters, origParameters, simulationFacade, orderNumber);
        }
        ImGui::PopID();
        AlienImGui::EndTreeNode();
    }
}

void SpecificationGuiService::createWidgetsForExpertToggles(SimulationParameters& parameters, SimulationParameters& origParameters) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& parametersSpecs = SimulationParameters::getSpec();

    for (auto const& groupSpec : parametersSpecs._groups) {
        if (groupSpec._expertToggle) {
            auto expertToggleValue = evaluationService.getExpertToggleRef(groupSpec._expertToggle, parameters);
            auto origExpertToggleValue = evaluationService.getExpertToggleRef(groupSpec._expertToggle, origParameters);
            AlienImGui::Checkbox(
                AlienImGui::CheckboxParameters()
                    .name(groupSpec._name)
                    .textWidth(0)
                    .defaultValue(*origExpertToggleValue).tooltip(groupSpec._description),
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

void SpecificationGuiService::createWidgetsForParameterGroup(
    std::vector<ParameterSpec> const& parameterSpecs,
    bool enabled,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    SimulationFacade const& simulationFacade,
    int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    for (auto const& [index, parameterSpec] : parameterSpecs | boost::adaptors::indexed(0)) {
        if (!SpecificationEvaluationService::get().isVisible(parameterSpec, locationType)) {
            continue;
        }
        ImGui::PushID(toInt(index));

        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            createWidgetsForBoolSpec(parameterSpec, enabled, parameters, origParameters, orderNumber);
        } else if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
            createWidgetsForIntSpec(parameterSpec, enabled, parameters, origParameters, orderNumber);
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            createWidgetsForFloatSpec(parameterSpec, enabled, parameters, origParameters, simulationFacade, orderNumber);
        } else if (std::holds_alternative<Float2Spec>(parameterSpec._reference)) {
            createWidgetsForFloat2Spec(parameterSpec, enabled, parameters, origParameters, simulationFacade, orderNumber);
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            createWidgetsForChar64Spec(parameterSpec, enabled, parameters, origParameters, orderNumber);
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            createWidgetsForAlternativeSpec(parameterSpec, enabled, parameters, origParameters, simulationFacade, orderNumber);
        } else if (std::holds_alternative<ColorSpec>(parameterSpec._reference)) {
            createWidgetsForColorPickerSpec(parameterSpec, enabled, parameters, origParameters, orderNumber);
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            createWidgetsForColorTransitionRulesSpec(parameterSpec, enabled, parameters, origParameters, orderNumber);
        }

        ImGui::PopID();
    }
}

void SpecificationGuiService::createWidgetsForBoolSpec(
    ParameterSpec const& parameterSpec,
    bool enabled,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& boolSpec = std::get<BoolSpec>(parameterSpec._reference);

    auto ref = evaluationService.getRef(boolSpec._member, parameters, orderNumber);
    auto origRef = evaluationService.getRef(boolSpec._member, origParameters, orderNumber);
    if (ref.colorDependence == ColorDependence::ColorMatrix) {

        AlienImGui::CheckboxColorMatrix(
            AlienImGui::CheckboxColorMatrixParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(origRef.value)))
                .tooltip(parameterSpec._description),
            *reinterpret_cast<bool(*)[MAX_COLORS][MAX_COLORS]>(ref.value));

    } else {
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters()
                .name(parameterSpec._name)
                .textWidth(RightColumnWidth)
                .defaultValue(*origRef.value)
                .tooltip(parameterSpec._description),
            *ref.value);

    }
}

void SpecificationGuiService::createWidgetsForIntSpec(
    ParameterSpec const& parameterSpec,
    bool enabled,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& intSpec = std::get<IntSpec>(parameterSpec._reference);

    auto [value, disabledValue, enabledValue, pinnedValue, valueType] = evaluationService.getRef(intSpec._member, parameters, orderNumber);
    auto [origValue, origDisabledValue, origEnabledValue, origPinnedValue, origValueType] =
        evaluationService.getRef(intSpec._member, origParameters, orderNumber);

    if (valueType == ColorDependence::ColorMatrix) {
        AlienImGui::InputIntColorMatrix(
            AlienImGui::InputIntColorMatrixParameters()
                .name(parameterSpec._name)
                .max(intSpec._min)
                .max(intSpec._max)
                .logarithmic(intSpec._logarithmic)
                .textWidth(RightColumnWidth)
                .tooltip(parameterSpec._description)
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
                .disabledValue(disabledValue)
                .defaultValue(origValue)
                .defaultEnabledValue(origEnabledValue)
                .tooltip(parameterSpec._description)
                .colorDependence(valueType == ColorDependence::ColorVector),
            value,
            enabledValue);
    }
}

void SpecificationGuiService::createWidgetsForFloatSpec(
    ParameterSpec const& parameterSpec,
    bool enabled,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    SimulationFacade const& simulationFacade,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);

    auto [value, disabledValue, enabledValue, pinnedValue, valueType] = evaluationService.getRef(floatSpec._member, parameters, orderNumber);
    auto [origValue, origDisabledValue, origEnabledValue, origPinnedValue, origValueType] =
        evaluationService.getRef(floatSpec._member, origParameters, orderNumber);

    auto min = std::get<float>(floatSpec._min);
    auto max = [&] {
        if (std::holds_alternative<MaxWorldRadiusSize>(floatSpec._max)) {
            auto worldSize = simulationFacade->getWorldSize();
            return toFloat(std::max(worldSize.x, worldSize.y));
        } else {
            return std::get<float>(floatSpec._max);
        }
    }();

    if (valueType == ColorDependence::ColorMatrix) {
        AlienImGui::InputFloatColorMatrix(
            AlienImGui::InputFloatColorMatrixParameters()
                .name(parameterSpec._name)
                .min(min)
                .max(max)
                .logarithmic(floatSpec._logarithmic)
                .format(floatSpec._format)
                .textWidth(RightColumnWidth)
                .tooltip(parameterSpec._description)
                .defaultValue(toVector<MAX_COLORS, MAX_COLORS>(*reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(origValue)))
                .disabledValue(
                    disabledValue != nullptr ? std::make_optional(toVector<MAX_COLORS, MAX_COLORS>(*reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(disabledValue)))
                                         : std::optional<std::vector<std::vector<float>>>()),
            *reinterpret_cast<float(*)[MAX_COLORS][MAX_COLORS]>(value),
            enabledValue);

    } else {
        float tempValue;
        float tempOrigValue;
        if (floatSpec._getterSetter.has_value()) {
            auto [getter, setter] = floatSpec._getterSetter.value();
            tempValue = getter(parameters, orderNumber);
            tempOrigValue = getter(origParameters, orderNumber);
            value = &tempValue;
            origValue = &tempOrigValue;
        }

        if (AlienImGui::SliderFloat(
                AlienImGui::SliderFloatParameters()
                    .name(parameterSpec._name)
                    .textWidth(RightColumnWidth)
                    .min(min)
                    .max(max)
                    .logarithmic(floatSpec._logarithmic)
                    .format(floatSpec._format)
                    .infinity(floatSpec._infinity)
                    .disabled(!enabled)
                    .disabledValue(disabledValue)
                    .defaultValue(origValue)
                    .defaultEnabledValue(origEnabledValue)
                    .colorDependence(valueType == ColorDependence::ColorVector)
                    .tooltip(parameterSpec._description),
                value,
            enabledValue,
            pinnedValue)) {

            if (floatSpec._getterSetter.has_value()) {
                auto [getter, setter] = floatSpec._getterSetter.value();
                setter(tempValue, parameters, orderNumber);
            }
        }
    }
}

void SpecificationGuiService::createWidgetsForFloat2Spec(
    ParameterSpec const& parameterSpec,
    bool enabled,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    SimulationFacade const& simulationFacade,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& float2Spec = std::get<Float2Spec>(parameterSpec._reference);

    auto [value, disabledValue, enabledValue, pinnedValue, valueType] = evaluationService.getRef(float2Spec._member, parameters, orderNumber);
    auto [origValue, origDisabledValue, origEnabledValue, origPinnedValue, origValueType] =
        evaluationService.getRef(float2Spec._member, origParameters, orderNumber);

    RealVector2D min = std::get<RealVector2D>(float2Spec._min);
    RealVector2D max = [&] {
        if (std::holds_alternative<WorldSize>(float2Spec._max)) {
            return toRealVector2D(simulationFacade->getWorldSize());
        } else {
            return std::get<RealVector2D>(float2Spec._max);
        }
    }();

    auto getMousePickerEnabledFunc = [&]() { return SimulationInteractionController::get().isPositionSelectionMode(); };
    auto setMousePickerEnabledFunc = [&](bool value) { SimulationInteractionController::get().setPositionSelectionMode(value); };
    auto getMousePickerPositionFunc = [&]() { return SimulationInteractionController::get().getPositionSelectionData(); };

    AlienImGui::SliderFloat2(
        AlienImGui::SliderFloat2Parameters()
            .name(parameterSpec._name)
            .textWidth(RightColumnWidth)
            .min(min)
            .max(max)
            .defaultValue(*origValue)
            .format(float2Spec._format)
            .getMousePickerEnabledFunc(float2Spec._mousePicker ? std::make_optional(getMousePickerEnabledFunc) : std::nullopt)
            .setMousePickerEnabledFunc(float2Spec._mousePicker ? std::make_optional(setMousePickerEnabledFunc) : std::nullopt)
            .getMousePickerPositionFunc(float2Spec._mousePicker ? std::make_optional(getMousePickerPositionFunc) : std::nullopt)
            .tooltip(parameterSpec._description),
        value->x,
        value->y);
}

void SpecificationGuiService::createWidgetsForChar64Spec(
    ParameterSpec const& parameterSpec,
    bool enabled,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& char64Spec = std::get<Char64Spec>(parameterSpec._reference);

    auto [value, disabledValue, enabledValue, pinnedValue, valueType] = evaluationService.getRef(char64Spec._member, parameters, orderNumber);
    auto [origValue, origDisabledValue, origEnabledValue, origPinnedValue, origValueType] =
        evaluationService.getRef(char64Spec._member, origParameters, orderNumber);

    AlienImGui::InputText(
        AlienImGui::InputTextParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origValue).tooltip(parameterSpec._description),
        *value,
        sizeof(Char64) / sizeof(char));
}

void SpecificationGuiService::createWidgetsForAlternativeSpec(
    ParameterSpec const& parameterSpec,
    bool enabled,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    SimulationFacade const& simulationFacade,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto alternativeSpec = std::get<AlternativeSpec>(parameterSpec._reference);

    auto [value, disabledValue, enabledValue, pinnedValue, valueType] = evaluationService.getRef(alternativeSpec._member, parameters, orderNumber);
    auto [origValue, origDisabledValue, origEnabledValue, origPinnedValue, origValueType] =
        evaluationService.getRef(alternativeSpec._member, origParameters, orderNumber);

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
            .disabled(!enabled)
            .tooltip(parameterSpec._description),
        *value,
        enabledValue);

    auto const& parametersForAlternative = alternativeSpec._alternatives.at(*value).second;
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
    auto containsWidgets = std::any_of(parametersForAlternative.begin(), parametersForAlternative.end(), [&](auto const& parameterSpec) {
        return evaluationService.isVisible(parameterSpec, locationType);
    });

    if (containsWidgets) {
        ImGui::Dummy(ImVec2(scale(22), 0));
        ImGui::SameLine();
        ImGui::BeginGroup();
        if (enabled) {
            enabled = enabledValue != nullptr ? *enabledValue : true;
        }
        createWidgetsForParameterGroup(alternativeSpec._alternatives.at(*value).second, enabled, parameters, origParameters, simulationFacade, orderNumber);
        ImGui::EndGroup();
    }
}

void SpecificationGuiService::createWidgetsForColorPickerSpec(
    ParameterSpec const& parameterSpec,
    bool enabled,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& colorPickerSpec = std::get<ColorSpec>(parameterSpec._reference);

    auto [value, disabledValue, enabledValue, pinnedValue, valueType] = evaluationService.getRef(colorPickerSpec._member, parameters, orderNumber);
    auto [origValue, origDisabledValue, origEnabledValue, origPinnedValue, origValueType] =
        evaluationService.getRef(colorPickerSpec._member, origParameters, orderNumber);
   
    AlienImGui::ColorButtonWithPicker(
        AlienImGui::ColorButtonWithPickerParameters().name(parameterSpec._name).textWidth(RightColumnWidth).defaultValue(*origValue), *value);
}

void SpecificationGuiService::createWidgetsForColorTransitionRulesSpec(
    ParameterSpec const& parameterSpec,
    bool enabled,
    SimulationParameters& parameters,
    SimulationParameters& origParameters,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);

    auto [value, disabledValue, enabledValue, pinnedValue, valueType] = evaluationService.getRef(colorTransitionRulesSpec._member, parameters, orderNumber);
    auto [origValue, origDisabledValue, origEnabledValue, origPinnedValue, origValueType] =
        evaluationService.getRef(colorTransitionRulesSpec._member, origParameters, orderNumber);

    for (int color = 0; color < MAX_COLORS; ++color) {
        ImGui::PushID(color);
        auto widgetParameters = AlienImGui::InputColorTransitionParameters()
                                    .textWidth(RightColumnWidth)
                                    .color(color)
                                    .defaultTargetColor(origValue[color].targetColor)
                                    .defaultTransitionAge(origValue[color].duration)
                                    .logarithmic(true)
                                    .infinity(true);
        if (0 == color) {
            widgetParameters.name(parameterSpec._name).tooltip(parameterSpec._description);
        }
        AlienImGui::InputColorTransition(widgetParameters, color, value[color].targetColor, value[color].duration);
        ImGui::PopID();
    }
}
