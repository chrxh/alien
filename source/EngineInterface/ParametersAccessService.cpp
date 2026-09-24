#include "ParametersAccessService.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <ranges>
#include <stdexcept>
#include <span>

#include <boost/algorithm/string.hpp>

#include <Base/Definitions.h>
#include <Base/StringHelper.h>

#include "LocationHelper.h"

std::vector<ParameterGroupSpec const*> ParametersAccessService::getGroups(LocationType locationType) const
{
    std::vector<ParameterGroupSpec const*> result;
    for (auto const& groupSpec : SimulationParameters::getSpec()._groups) {
        if (SpecificationEvaluationService::get().isVisible(groupSpec, locationType)) {
            result.emplace_back(&groupSpec);
        }
    }
    return result;
}

ParameterGroupSpec const* ParametersAccessService::findGroup(std::string const& name) const
{
    for (auto const& groupSpec : SimulationParameters::getSpec()._groups) {
        if (boost::algorithm::iequals(groupSpec._name, name)) {
            return &groupSpec;
        }
    }
    return nullptr;
}

namespace
{
    int getSelectedAlternativeIndex(AlternativeSpec const& alternativeSpec, SimulationParameters& parameters, int orderNumber)
    {
        auto ref = SpecificationEvaluationService::get().getRef(alternativeSpec._member, parameters, orderNumber);
        if (!ref.value || ref.colorDependence != ColorDependence::None) {
            return -1;
        }
        return std::clamp(*ref.value, 0, toInt(alternativeSpec._alternatives.size()) - 1);
    }

    void collectParameters(
        std::vector<ParameterSpec> const& parameterSpecs,
        std::string const& pathPrefix,
        SimulationParameters& parameters,
        int orderNumber,
        std::vector<ParameterEntry>& result)
    {
        auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
        for (auto const& parameterSpec : parameterSpecs) {
            if (!SpecificationEvaluationService::get().isVisible(parameterSpec, locationType)) {
                continue;
            }
            auto path = pathPrefix + "." + parameterSpec._name;
            result.emplace_back(ParameterEntry{.path = path, .spec = &parameterSpec});

            if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
                auto const& alternativeSpec = std::get<AlternativeSpec>(parameterSpec._reference);
                auto index = getSelectedAlternativeIndex(alternativeSpec, parameters, orderNumber);
                if (index >= 0) {
                    auto const& [alternativeName, alternativeParameterSpecs] = alternativeSpec._alternatives.at(index);
                    collectParameters(alternativeParameterSpecs, path + "." + alternativeName, parameters, orderNumber, result);
                }
            }
        }
    }
}

std::vector<ParameterEntry> ParametersAccessService::getParameters(ParameterGroupSpec const& groupSpec, SimulationParameters const& parameters, int orderNumber)
    const
{
    std::vector<ParameterEntry> result;
    collectParameters(groupSpec._parameters, groupSpec._name, const_cast<SimulationParameters&>(parameters), orderNumber, result);
    return result;
}

std::optional<ParameterEntry> ParametersAccessService::findParameter(std::string const& path, LocationType locationType) const
{
    std::vector<std::string> segments;
    boost::algorithm::split(segments, path, boost::algorithm::is_any_of("."));
    if (segments.size() < 2) {
        return std::nullopt;
    }
    auto groupSpec = findGroup(boost::algorithm::trim_copy(segments.front()));
    if (!groupSpec) {
        return std::nullopt;
    }

    auto resultPath = groupSpec->_name;
    auto parameterSpecs = &groupSpec->_parameters;
    for (size_t i = 1; i < segments.size(); i += 2) {
        auto parameterName = boost::algorithm::trim_copy(segments.at(i));
        auto parameterIter = std::ranges::find_if(*parameterSpecs, [&](ParameterSpec const& parameterSpec) {
            return boost::algorithm::iequals(parameterSpec._name, parameterName)
                && SpecificationEvaluationService::get().isVisible(parameterSpec, locationType);
        });
        if (parameterIter == parameterSpecs->end()) {
            return std::nullopt;
        }
        resultPath += "." + parameterIter->_name;
        if (i + 1 == segments.size()) {
            return ParameterEntry{.path = resultPath, .spec = &*parameterIter};
        }

        if (!std::holds_alternative<AlternativeSpec>(parameterIter->_reference)) {
            return std::nullopt;
        }
        auto const& alternatives = std::get<AlternativeSpec>(parameterIter->_reference)._alternatives;
        auto alternativeName = boost::algorithm::trim_copy(segments.at(i + 1));
        auto alternativeIter =
            std::ranges::find_if(alternatives, [&](auto const& alternative) { return boost::algorithm::iequals(alternative.first, alternativeName); });
        if (alternativeIter == alternatives.end()) {
            return std::nullopt;
        }
        resultPath += "." + alternativeIter->first;
        parameterSpecs = &alternativeIter->second;
    }
    return std::nullopt;
}

ParameterType ParametersAccessService::getType(ParameterEntry const& entry) const
{
    auto const& reference = entry.spec->_reference;
    if (std::holds_alternative<BoolSpec>(reference)) {
        return ParameterType::Bool;
    } else if (std::holds_alternative<IntSpec>(reference)) {
        return ParameterType::Int;
    } else if (std::holds_alternative<FloatSpec>(reference)) {
        return ParameterType::Float;
    } else if (std::holds_alternative<Float2Spec>(reference)) {
        return ParameterType::Float2;
    } else if (std::holds_alternative<Char64Spec>(reference)) {
        return ParameterType::Text;
    } else if (std::holds_alternative<AlternativeSpec>(reference)) {
        return ParameterType::Alternative;
    } else if (std::holds_alternative<ColorSpec>(reference)) {
        return ParameterType::Color;
    } else {
        return ParameterType::ColorTransitionRule;
    }
}

namespace
{
    int getArraySize(ColorDependence colorDependence)
    {
        if (colorDependence == ColorDependence::ColorVector) {
            return MAX_COLORS;
        } else if (colorDependence == ColorDependence::ColorMatrix) {
            return MAX_COLORS * MAX_COLORS;
        } else {
            return 1;
        }
    }

    template <typename T>
    ParameterValue toParameterValue(ValueRef<T> const& ref, std::function<ParameterScalar(T const&)> const& convert)
    {
        ParameterValue result{.colorDependence = ref.colorDependence};
        if (ref.enabled) {
            result.enabled = *ref.enabled;
        }
        if (ref.pinned) {
            result.pinned = *ref.pinned;
        }
        auto source = ref.enabled && !*ref.enabled && ref.disabledValue ? ref.disabledValue : ref.value;
        if (source) {
            for (auto const& element : std::span(source, getArraySize(ref.colorDependence))) {
                result.values.emplace_back(convert(element));
            }
        }
        return result;
    }

    template <typename T>
    ParameterScalar identity(T const& value)
    {
        return value;
    }
}

ParameterValue ParametersAccessService::getValue(ParameterEntry const& entry, SimulationParameters const& parameters, int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto& parametersRef = const_cast<SimulationParameters&>(parameters);
    auto const& reference = entry.spec->_reference;

    if (std::holds_alternative<BoolSpec>(reference)) {
        return toParameterValue<bool>(evaluationService.getRef(std::get<BoolSpec>(reference)._member, parametersRef, orderNumber), identity<bool>);
    } else if (std::holds_alternative<IntSpec>(reference)) {
        return toParameterValue<int>(evaluationService.getRef(std::get<IntSpec>(reference)._member, parametersRef, orderNumber), identity<int>);
    } else if (std::holds_alternative<FloatSpec>(reference)) {
        auto const& floatSpec = std::get<FloatSpec>(reference);
        auto result = toParameterValue<float>(evaluationService.getRef(floatSpec._member, parametersRef, orderNumber), identity<float>);
        if (floatSpec._getterSetter.has_value()) {
            auto const& getter = floatSpec._getterSetter->first;
            result.colorDependence = ColorDependence::None;
            result.values = {getter(parameters, orderNumber)};
        }
        return result;
    } else if (std::holds_alternative<Float2Spec>(reference)) {
        return toParameterValue<RealVector2D>(
            evaluationService.getRef(std::get<Float2Spec>(reference)._member, parametersRef, orderNumber), identity<RealVector2D>);
    } else if (std::holds_alternative<Char64Spec>(reference)) {
        return toParameterValue<Char64>(evaluationService.getRef(std::get<Char64Spec>(reference)._member, parametersRef, orderNumber), [](Char64 const& value) {
            return ParameterScalar(std::string(value));
        });
    } else if (std::holds_alternative<AlternativeSpec>(reference)) {
        auto const& alternatives = std::get<AlternativeSpec>(reference)._alternatives;
        return toParameterValue<int>(evaluationService.getRef(std::get<AlternativeSpec>(reference)._member, parametersRef, orderNumber), [&](int const& value) {
            return ParameterScalar(alternatives.at(std::clamp(value, 0, toInt(alternatives.size()) - 1)).first);
        });
    } else if (std::holds_alternative<ColorSpec>(reference)) {
        return toParameterValue<FloatColorRGB>(
            evaluationService.getRef(std::get<ColorSpec>(reference)._member, parametersRef, orderNumber), identity<FloatColorRGB>);
    } else {
        return toParameterValue<ColorTransitionRule>(
            evaluationService.getRef(std::get<ColorTransitionRulesSpec>(reference)._member, parametersRef, orderNumber), identity<ColorTransitionRule>);
    }
}

namespace
{
    template <typename T>
    T const& getScalar(ParameterScalar const& scalar, ParameterEntry const& entry)
    {
        if (auto result = std::get_if<T>(&scalar)) {
            return *result;
        }
        throw std::invalid_argument("The value for '" + entry.path + "' has the wrong type.");
    }

    void checkNumValues(ParameterEntry const& entry, ParameterValue const& value, int expectedNumValues)
    {
        if (toInt(value.values.size()) != expectedNumValues) {
            throw std::invalid_argument(
                "The parameter '" + entry.path + "' expects " + std::to_string(expectedNumValues) + " value(s) but got " + std::to_string(value.values.size())
                + ".");
        }
    }

    template <typename T>
    void applyFlags(ValueRef<T> const& ref, ParameterEntry const& entry, ParameterValue const& value)
    {
        if (value.enabled.has_value()) {
            if (!ref.enabled) {
                throw std::invalid_argument("The parameter '" + entry.path + "' cannot be enabled or disabled.");
            }
            *ref.enabled = *value.enabled;
        }
        if (value.pinned.has_value()) {
            if (!ref.pinned) {
                throw std::invalid_argument("The parameter '" + entry.path + "' cannot be pinned.");
            }
            *ref.pinned = *value.pinned;
        }
    }

    template <typename T>
    void applyParameterValue(
        ValueRef<T> const& ref,
        ParameterEntry const& entry,
        ParameterValue const& value,
        std::function<void(T&, ParameterScalar const&)> const& assign)
    {
        if (!ref.value) {
            throw std::invalid_argument("The parameter '" + entry.path + "' is not available at this location.");
        }
        checkNumValues(entry, value, getArraySize(ref.colorDependence));
        for (auto const& [target, scalar] : std::views::zip(std::span(ref.value, value.values.size()), value.values)) {
            assign(target, scalar);
        }
        applyFlags(ref, entry, value);
    }

    template <typename T>
    std::function<void(T&, ParameterScalar const&)> assignScalar(ParameterEntry const& entry)
    {
        return [&entry](T& target, ParameterScalar const& scalar) { target = getScalar<T>(scalar, entry); };
    }
}

void ParametersAccessService::setValue(ParameterEntry const& entry, SimulationParameters& parameters, int orderNumber, ParameterValue const& value) const
{
    auto& evaluationService = SpecificationEvaluationService::get();
    auto const& reference = entry.spec->_reference;

    if (std::holds_alternative<BoolSpec>(reference)) {
        applyParameterValue<bool>(
            evaluationService.getRef(std::get<BoolSpec>(reference)._member, parameters, orderNumber), entry, value, assignScalar<bool>(entry));
    } else if (std::holds_alternative<IntSpec>(reference)) {
        applyParameterValue<int>(
            evaluationService.getRef(std::get<IntSpec>(reference)._member, parameters, orderNumber), entry, value, assignScalar<int>(entry));
    } else if (std::holds_alternative<FloatSpec>(reference)) {
        auto const& floatSpec = std::get<FloatSpec>(reference);
        auto ref = evaluationService.getRef(floatSpec._member, parameters, orderNumber);
        if (floatSpec._getterSetter.has_value()) {
            auto const& [getter, setter] = floatSpec._getterSetter.value();
            checkNumValues(entry, value, 1);
            auto newValue = getScalar<float>(value.values.front(), entry);
            if (getter(parameters, orderNumber) != newValue) {
                setter(newValue, parameters, orderNumber);
            }
            applyFlags(ref, entry, value);
        } else {
            applyParameterValue<float>(ref, entry, value, assignScalar<float>(entry));
        }
    } else if (std::holds_alternative<Float2Spec>(reference)) {
        applyParameterValue<RealVector2D>(
            evaluationService.getRef(std::get<Float2Spec>(reference)._member, parameters, orderNumber), entry, value, assignScalar<RealVector2D>(entry));
    } else if (std::holds_alternative<Char64Spec>(reference)) {
        applyParameterValue<Char64>(
            evaluationService.getRef(std::get<Char64Spec>(reference)._member, parameters, orderNumber),
            entry,
            value,
            [&entry](Char64& target, ParameterScalar const& scalar) { StringHelper::copy(target, sizeof(Char64), getScalar<std::string>(scalar, entry)); });
    } else if (std::holds_alternative<AlternativeSpec>(reference)) {
        auto const& alternatives = std::get<AlternativeSpec>(reference)._alternatives;
        applyParameterValue<int>(
            evaluationService.getRef(std::get<AlternativeSpec>(reference)._member, parameters, orderNumber),
            entry,
            value,
            [&](int& target, ParameterScalar const& scalar) {
                auto const& name = getScalar<std::string>(scalar, entry);
                auto iter = std::ranges::find_if(alternatives, [&](auto const& alternative) { return boost::algorithm::iequals(alternative.first, name); });
                if (iter == alternatives.end()) {
                    std::vector<std::string> names;
                    for (auto const& alternativeName : alternatives | std::views::keys) {
                        names.emplace_back("'" + alternativeName + "'");
                    }
                    throw std::invalid_argument(
                        "'" + name + "' is not a valid option for '" + entry.path + "'. Valid options are " + boost::algorithm::join(names, ", ") + ".");
                }
                target = toInt(std::distance(alternatives.begin(), iter));
            });
    } else if (std::holds_alternative<ColorSpec>(reference)) {
        applyParameterValue<FloatColorRGB>(
            evaluationService.getRef(std::get<ColorSpec>(reference)._member, parameters, orderNumber), entry, value, assignScalar<FloatColorRGB>(entry));
    } else {
        applyParameterValue<ColorTransitionRule>(
            evaluationService.getRef(std::get<ColorTransitionRulesSpec>(reference)._member, parameters, orderNumber),
            entry,
            value,
            assignScalar<ColorTransitionRule>(entry));
    }
}

namespace
{
    template <typename T>
    void copyRef(ValueRef<T> const& sourceRef, ValueRef<T> const& targetRef)
    {
        if (sourceRef.value && targetRef.value) {
            std::memcpy(targetRef.value, sourceRef.value, sizeof(T) * getArraySize(sourceRef.colorDependence));
        }
        if (sourceRef.enabled && targetRef.enabled) {
            *targetRef.enabled = *sourceRef.enabled;
        }
        if (sourceRef.pinned && targetRef.pinned) {
            *targetRef.pinned = *sourceRef.pinned;
        }
    }

    void copyParameters(std::vector<ParameterSpec> const& parameterSpecs, SimulationParameters& source, SimulationParameters& target, int orderNumber)
    {
        auto& evaluationService = SpecificationEvaluationService::get();
        auto locationType = LocationHelper::getLocationType(orderNumber, target);
        for (auto const& parameterSpec : parameterSpecs) {
            if (!evaluationService.isVisible(parameterSpec, locationType)) {
                continue;
            }
            std::visit(
                [&](auto const& spec) {
                    copyRef(evaluationService.getRef(spec._member, source, orderNumber), evaluationService.getRef(spec._member, target, orderNumber));
                },
                parameterSpec._reference);

            if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
                for (auto const& alternativeParameterSpecs : std::get<AlternativeSpec>(parameterSpec._reference)._alternatives | std::views::values) {
                    copyParameters(alternativeParameterSpecs, source, target, orderNumber);
                }
            }
        }
    }
}

bool ParametersAccessService::copyGroup(ParameterGroupSpec const& groupSpec, SimulationParameters const& source, SimulationParameters& target) const
{
    auto haveSameLocations = source.numLayers == target.numLayers && source.numSources == target.numSources
        && std::equal(source.layerOrderNumbers, source.layerOrderNumbers + source.numLayers, target.layerOrderNumbers)
        && std::equal(source.sourceOrderNumbers, source.sourceOrderNumbers + source.numSources, target.sourceOrderNumbers);
    if (!haveSameLocations) {
        return false;
    }

    auto& sourceRef = const_cast<SimulationParameters&>(source);
    if (groupSpec._expertToggle) {
        auto& evaluationService = SpecificationEvaluationService::get();
        *evaluationService.getExpertToggleRef(groupSpec._expertToggle, target) = *evaluationService.getExpertToggleRef(groupSpec._expertToggle, sourceRef);
    }
    for (int orderNumber = 0; orderNumber <= target.numLayers + target.numSources; ++orderNumber) {
        copyParameters(groupSpec._parameters, sourceRef, target, orderNumber);
    }
    return true;
}
