#include "SpecificationEvaluationService.h"

#include <algorithm>

#include "CellTypeStrings.h"
#include "LocationHelper.h"

ValueRef<bool> SpecificationEvaluationService::getRef(BoolMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    // Single value
    if (locationType == LocationType::Base && std::holds_alternative<BoolMember>(member)) {
        return ValueRef{.value = &(parameters.**std::get<BoolMember>(member)).value};
    } else if (locationType != LocationType::Source && std::holds_alternative<BoolBaseLayerMember>(member)) {
        switch (LocationHelper::getLocationType(orderNumber, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = &(parameters.**std::get<BoolBaseLayerMember>(member)).baseValue};
        case LocationType::Layer: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            return ValueRef{
                .value = &(parameters.**std::get<BoolBaseLayerMember>(member)).layerValues[index].value,
                .disabledValue = &(parameters.**std::get<BoolBaseLayerMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<BoolBaseLayerMember>(member)).layerValues[index].enabled};
        }
        }
    } else if (locationType == LocationType::Layer && std::holds_alternative<BoolLayerMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<BoolLayerMember>(member)).layerValues[index]};
    }

    // Color matrix
    else if (locationType == LocationType::Base && std::holds_alternative<ColorMatrixBoolMember>(member)) {
        return ValueRef{.value = reinterpret_cast<bool*>((parameters.**std::get<ColorMatrixBoolMember>(member)).value), .colorDependence = ColorDependence::ColorMatrix};
    }

    return {};
}

ValueRef<int> SpecificationEvaluationService::getRef(IntMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    // Single value
    if (locationType == LocationType::Base && std::holds_alternative<IntMember>(member)) {
        return ValueRef{.value = &(parameters.**std::get<IntMember>(member)).value};
    } else if (locationType == LocationType::Base && std::holds_alternative<IntEnableableMember>(member)) {
        return ValueRef{
            .value = &(parameters.**std::get<IntEnableableMember>(member)).value,
            .disabledValue = &(parameters.**std::get<IntEnableableMember>(member)).value,
            .enabled = &(parameters.**std::get<IntEnableableMember>(member)).enabled};
    } else if (locationType == LocationType::Layer && std::holds_alternative<IntLayerMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<IntLayerMember>(member)).layerValues[index]};
    }

    // Color vector
    else if (locationType == LocationType::Base && std::holds_alternative<ColorVectorIntMember>(member)) {
        return ValueRef{.value = (parameters.**std::get<ColorVectorIntMember>(member)).value, .colorDependence = ColorDependence::ColorVector};
    }

    // Color matrix
    else if (locationType == LocationType::Base && std::holds_alternative<ColorMatrixIntMember>(member)) {
        return ValueRef{.value = reinterpret_cast<int*>((parameters.**std::get<ColorMatrixIntMember>(member)).value), .colorDependence = ColorDependence::ColorMatrix};
    }
    return {};
}

ValueRef<float> SpecificationEvaluationService::getRef(FloatMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    // Single value
    if (locationType == LocationType::Base && std::holds_alternative<FloatMember>(member)) {
        return ValueRef{.value = &(parameters.**std::get<FloatMember>(member)).value};
    } else if (locationType != LocationType::Source && std::holds_alternative<FloatBaseLayerMember>(member)) {
        switch (LocationHelper::getLocationType(orderNumber, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = &(parameters.**std::get<FloatBaseLayerMember>(member)).baseValue};
        case LocationType::Layer: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            return ValueRef{
                .value = &(parameters.**std::get<FloatBaseLayerMember>(member)).layerValues[index].value,
                .disabledValue = &(parameters.**std::get<FloatBaseLayerMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<FloatBaseLayerMember>(member)).layerValues[index].enabled};
        }
        }
    } else if (locationType == LocationType::Base && std::holds_alternative<FloatPinMember>(member)) {
        return ValueRef<float>{
            .pinned = &(parameters.**std::get<FloatPinMember>(member)).pinned};
    } else if (locationType == LocationType::Layer && std::holds_alternative<FloatLayerMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<FloatLayerMember>(member)).layerValues[index]};
    } else if (locationType == LocationType::Source &&  std::holds_alternative<FloatSourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<FloatSourceMember>(member)).sourceValues[index]};
    } else if (locationType == LocationType::Source && std::holds_alternative<FloatEnableableSourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{
            .value = &(parameters.**std::get<FloatEnableableSourceMember>(member)).sourceValues[index].value,
            .disabledValue = &(parameters.**std::get<FloatEnableableSourceMember>(member)).sourceValues[index].value,
            .enabled = &(parameters.**std::get<FloatEnableableSourceMember>(member)).sourceValues[index].enabled};
    } else if (locationType == LocationType::Source && std::holds_alternative<FloatPinnableSourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{
            .value = &(parameters.**std::get<FloatPinnableSourceMember>(member)).sourceValues[index].value,
            .pinned = &(parameters.**std::get<FloatPinnableSourceMember>(member)).sourceValues[index].pinned,
        };
    }
    
    // Color vector
    else if (locationType == LocationType::Base && std::holds_alternative<ColorVectorFloatMember>(member)) {
        return ValueRef{.value = (parameters.**std::get<ColorVectorFloatMember>(member)).value, .colorDependence = ColorDependence::ColorVector};
    } else if (locationType != LocationType::Source && std::holds_alternative<ColorVectorFloatBaseLayerMember>(member)) {
        switch (LocationHelper::getLocationType(orderNumber, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = (parameters.**std::get<ColorVectorFloatBaseLayerMember>(member)).baseValue, .colorDependence = ColorDependence::ColorVector};
        case LocationType::Layer: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            return ValueRef{
                .value = (parameters.**std::get<ColorVectorFloatBaseLayerMember>(member)).layerValues[index].value,
                .disabledValue = (parameters.**std::get<ColorVectorFloatBaseLayerMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<ColorVectorFloatBaseLayerMember>(member)).layerValues[index].enabled,
                .colorDependence = ColorDependence::ColorVector};
        }
        }
    }

    // Color matrix
    else if (locationType == LocationType::Base && std::holds_alternative<ColorMatrixFloatMember>(member)) {
        return ValueRef{.value = reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatMember>(member)).value), .colorDependence = ColorDependence::ColorMatrix};
    } else if (locationType != LocationType::Source && std::holds_alternative<ColorMatrixFloatBaseLayerMember>(member)) {
        switch (LocationHelper::getLocationType(orderNumber, parameters)) {
        case LocationType::Base:
            return ValueRef{
                .value = reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatBaseLayerMember>(member)).baseValue),
                .colorDependence = ColorDependence::ColorMatrix};
        case LocationType::Layer: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            return ValueRef{
                .value = reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatBaseLayerMember>(member)).layerValues[index].value),
                .disabledValue = reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatBaseLayerMember>(member)).baseValue),
                .enabled = &(parameters.**std::get<ColorMatrixFloatBaseLayerMember>(member)).layerValues[index].enabled,
                .colorDependence = ColorDependence::ColorMatrix};
        }
        }
    }

    return {};
}

ValueRef<RealVector2D> SpecificationEvaluationService::getRef(Float2MemberVariant const& member, SimulationParameters& parameters, int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    if (locationType == LocationType::Layer && std::holds_alternative<Float2LayerMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<Float2LayerMember>(member)).layerValues[index]};
    } else if (locationType == LocationType::Source && std::holds_alternative<Float2SourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<Float2SourceMember>(member)).sourceValues[index]};
    }

    return {};
}

ValueRef<Char64> SpecificationEvaluationService::getRef(Char64MemberVariant const& member, SimulationParameters& parameters, int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    if (locationType == LocationType::Base && std::holds_alternative<Char64Member>(member)) {
        return ValueRef{.value = &(parameters.**std::get<Char64Member>(member)).value};
    } else if (locationType == LocationType::Layer && std::holds_alternative<Char64LayerMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<Char64LayerMember>(member)).layerValues[index]};
    } else if (locationType == LocationType::Source && std::holds_alternative<Char64SourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<Char64SourceMember>(member)).sourceValues[index]};
    }

    return {};
}

ValueRef<int> SpecificationEvaluationService::getRef(AlternativeMemberVariant const& member, SimulationParameters& parameters, int orderNumber)
    const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    if (locationType == LocationType::Base && std::holds_alternative<IntMember>(member)) {
        return ValueRef{.value = &(parameters.**std::get<IntMember>(member)).value};
    } else if (locationType == LocationType::Layer && std::holds_alternative<IntLayerMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<IntLayerMember>(member)).layerValues[index]};
    } else if (locationType == LocationType::Source && std::holds_alternative<IntSourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return ValueRef{.value = &(parameters.**std::get<IntSourceMember>(member)).sourceValues[index]};
    }

    return {};
}

ValueRef<FloatColorRGB>
SpecificationEvaluationService::getRef(FloatColorRGBMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    if (locationType != LocationType::Source && std::holds_alternative<FloatColorRGBBaseLayerMember>(member)) {
        switch (LocationHelper::getLocationType(orderNumber, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = &(parameters.**std::get<FloatColorRGBBaseLayerMember>(member)).baseValue};
        case LocationType::Layer: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            return ValueRef{
                .value = &(parameters.**std::get<FloatColorRGBBaseLayerMember>(member)).layerValues[index].value,
                .disabledValue = &(parameters.**std::get<FloatColorRGBBaseLayerMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<FloatColorRGBBaseLayerMember>(member)).layerValues[index].enabled};
        }
        }
    }
    return {};
}

ValueRef<ColorTransitionRule> SpecificationEvaluationService::getRef(ColorTransitionRulesMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const
{
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    if (locationType != LocationType::Source && std::holds_alternative<ColorTransitionRulesBaseLayerMember>(member)) {
        switch (LocationHelper::getLocationType(orderNumber, parameters)) {
        case LocationType::Base:
            return ValueRef{
                .value = (parameters.**std::get<ColorTransitionRulesBaseLayerMember>(member)).baseValue, .colorDependence = ColorDependence::ColorVector};
        case LocationType::Layer: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            return ValueRef{
                .value = (parameters.**std::get<ColorTransitionRulesBaseLayerMember>(member)).layerValues[index].value,
                .disabledValue = (parameters.**std::get<ColorTransitionRulesBaseLayerMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<ColorTransitionRulesBaseLayerMember>(member)).layerValues[index].enabled,
                .colorDependence = ColorDependence::ColorVector};
        }
        }
    }
    return {};
}

bool* SpecificationEvaluationService::getExpertToggleRef(ExpertToggleMember const& member, SimulationParameters& parameters) const
{
    return &(parameters.**member).value;
}

bool SpecificationEvaluationService::isVisible(ParameterGroupSpec const& groupSpec, LocationType locationType) const
{
    return std::any_of(
        groupSpec._parameters.begin(), groupSpec._parameters.end(), [&](auto const& parameterSpec) { return isVisible(parameterSpec, locationType); });
}

bool SpecificationEvaluationService::isVisible(ParameterSpec const& parameterSpec, LocationType locationType) const
{
    if (!parameterSpec._visible) {
        return false;
    }
    if (locationType == LocationType::Base) {
        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            auto const& boolSpec = std::get<BoolSpec>(parameterSpec._reference);
            if (std::holds_alternative<BoolMember>(boolSpec._member) || std::holds_alternative<ColorMatrixBoolMember>(boolSpec._member)
                || std::holds_alternative<BoolBaseLayerMember>(boolSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<IntSpec>(parameterSpec._reference)) {
            auto const& intSpec = std::get<IntSpec>(parameterSpec._reference);
            if (std::holds_alternative<IntMember>(intSpec._member) || std::holds_alternative<IntEnableableMember>(intSpec._member)
                || std::holds_alternative<ColorVectorIntMember>(intSpec._member) || std::holds_alternative<ColorMatrixIntMember>(intSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatMember>(floatSpec._member) || std::holds_alternative<FloatPinMember>(floatSpec._member)
                || std::holds_alternative<ColorVectorFloatMember>(floatSpec._member) || std::holds_alternative<ColorMatrixFloatMember>(floatSpec._member)
                || std::holds_alternative<FloatBaseLayerMember>(floatSpec._member) || std::holds_alternative<ColorVectorFloatBaseLayerMember>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatBaseLayerMember>(floatSpec._member)) {
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
            if (std::holds_alternative<FloatColorRGBBaseLayerMember>(colorPickerSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorTransitionRulesBaseLayerMember>(colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }
    if (locationType == LocationType::Layer) {
        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            auto const& boolSpec = std::get<BoolSpec>(parameterSpec._reference);
            if (std::holds_alternative<BoolBaseLayerMember>(boolSpec._member) || std::holds_alternative<BoolLayerMember>(boolSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorVectorFloatBaseLayerMember>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatBaseLayerMember>(floatSpec._member)
                || std::holds_alternative<FloatLayerMember>(floatSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<Float2Spec>(parameterSpec._reference)) {
            auto const& float2Spec = std::get<Float2Spec>(parameterSpec._reference);
            if (std::holds_alternative<Float2LayerMember>(float2Spec._member)) {
                return true;
            }
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            auto const& char64Spec = std::get<Char64Spec>(parameterSpec._reference);
            if (std::holds_alternative<Char64LayerMember>(char64Spec._member)) {
                return true;
            }
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            auto const& alternativeSpec = std::get<AlternativeSpec>(parameterSpec._reference);
            if (std::holds_alternative<IntLayerMember>(alternativeSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._reference)) {
            auto const& colorPickerSpec = std::get<ColorPickerSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatColorRGBBaseLayerMember>(colorPickerSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorTransitionRulesBaseLayerMember>(colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }
    if (locationType == LocationType::Source) {
        if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatSourceMember>(floatSpec._member)
                || std::holds_alternative<FloatEnableableSourceMember>(floatSpec._member)
                || std::holds_alternative<FloatPinnableSourceMember>(floatSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<Float2Spec>(parameterSpec._reference)) {
            auto const& float2Spec = std::get<Float2Spec>(parameterSpec._reference);
            if (std::holds_alternative<Float2SourceMember>(float2Spec._member)) {
                return true;
            }
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            auto const& char64Spec = std::get<Char64Spec>(parameterSpec._reference);
            if (std::holds_alternative<Char64SourceMember>(char64Spec._member)) {
                return true;
            }
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            auto const& alternativeSpec = std::get<AlternativeSpec>(parameterSpec._reference);
            if (std::holds_alternative<IntSourceMember>(alternativeSpec._member)) {
                return true;
            }
        }
    }

    return false;
}
