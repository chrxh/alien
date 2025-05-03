#include "ParametersValidationService.h"

#include <algorithm>
#include <ranges>

#include "SimulationParametersSpecification.h"
#include "SpecificationEvaluationService.h"

void ParametersValidationService::validateAndCorrect(ValidationConfig const& config, SimulationParameters& parameters) const
{
    auto const& parametersSpecs = SimulationParameters::getSpec();
    for (int i = 0; i <= parameters.numLayers + parameters.numSources; ++i) {
        for (auto const& groupSpec : parametersSpecs._groups) {
            validateAndCorrectIntern(config, groupSpec._parameters, parameters, i);
        }
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
}

void ParametersValidationService::validateAndCorrectIntern(
    ValidationConfig const& config,
    std::vector<ParameterSpec> const& parameterSpecs,
    SimulationParameters& parameters,
    int orderNumber) const
{
    auto& evaluationService = SpecificationEvaluationService::get();

    for (auto const& parameterSpec : parameterSpecs) {
        if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
            auto const& valueSpec = std::get<IntSpec>(parameterSpec._reference);
            auto ref = evaluationService.getRef(valueSpec._member, parameters, orderNumber);
            if (ref.value) {
                auto arraySize = getArraySize(ref.colorDependence);
                for (int i = 0; i < arraySize; ++i) {
                    ref.value[i] = std::clamp(ref.value[i], valueSpec._min, valueSpec._max);
                }
            }
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& valueSpec = std::get<FloatSpec>(parameterSpec._reference);
            auto ref = evaluationService.getRef(valueSpec._member, parameters, orderNumber);
            if (ref.value) {
                auto min = std::get<float>(valueSpec._min);
                auto max = [&] {
                    if (std::holds_alternative<MaxWorldRadiusSize>(valueSpec._max)) {
                        return toFloat(std::max(config.worldSize.x, config.worldSize.y));
                    } else {
                        return std::get<float>(valueSpec._max);
                    }
                }();
                auto arraySize = getArraySize(ref.colorDependence);
                for (int i = 0; i < arraySize; ++i) {
                    ref.value[i] = std::clamp(ref.value[i], min, max);
                }
                auto otherRef = evaluationService.getRef(valueSpec._greaterThan, parameters, orderNumber);
                if (otherRef.value) {
                    for (int i = 0; i < arraySize; ++i) {
                        ref.value[i] = std::max(ref.value[i], otherRef.value[i]);
                    }
                }
            }
        } else if (std::holds_alternative<Float2Spec>(parameterSpec._reference)) {
            auto const& valueSpec = std::get<Float2Spec>(parameterSpec._reference);
            auto ref = evaluationService.getRef(valueSpec._member, parameters, orderNumber);
            if (ref.value) {
                auto min = std::get<RealVector2D>(valueSpec._min);
                auto max = [&] {
                    if (std::holds_alternative<WorldSize>(valueSpec._max)) {
                        return toRealVector2D(config.worldSize);
                    } else {
                        return std::get<RealVector2D>(valueSpec._max);
                    }
                }();
                auto arraySize = getArraySize(ref.colorDependence);
                for (int i = 0; i < arraySize; ++i) {
                    ref.value[i].x = std::clamp(ref.value[i].x, min.x, max.x);
                    ref.value[i].y = std::clamp(ref.value[i].y, min.y, max.y);
                }
            }
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            auto const& valueSpec = std::get<AlternativeSpec>(parameterSpec._reference);
            auto ref = evaluationService.getRef(valueSpec._member, parameters, orderNumber);
            if (ref.value) {
                auto max = toInt(valueSpec._alternatives.size());
                *ref.value = std::clamp(*ref.value, 0, max);
            }
            for (auto const& parameterSpecs : valueSpec._alternatives | std::views::values) {
                validateAndCorrectIntern(config, parameterSpecs, parameters, orderNumber);
            }
        } else if (std::holds_alternative<ColorSpec>(parameterSpec._reference)) {
            auto const& valueSpec = std::get<ColorSpec>(parameterSpec._reference);
            auto ref = evaluationService.getRef(valueSpec._member, parameters, orderNumber);
            if (ref.value) {
                auto arraySize = getArraySize(ref.colorDependence);
                for (int i = 0; i < arraySize; ++i) {
                    ref.value[i].r = std::clamp(ref.value[i].r, 0.0f, 1.0f);
                    ref.value[i].g = std::clamp(ref.value[i].g, 0.0f, 1.0f);
                    ref.value[i].b = std::clamp(ref.value[i].b, 0.0f, 1.0f);
                }
            }
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            auto const& valueSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            auto ref = evaluationService.getRef(valueSpec._member, parameters, orderNumber);
            if (ref.value) {
                auto arraySize = getArraySize(ref.colorDependence);
                for (int i = 0; i < arraySize; ++i) {
                    ref.value[i].duration = std::max(ref.value[i].duration, 0);
                    ref.value[i].targetColor = std::clamp(ref.value[i].targetColor, 0, MAX_COLORS - 1);
                }
            }
        }
    }
}
