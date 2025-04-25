#include "SpecificationEvaluationService.h"

#include <algorithm>

#include "CellTypeStrings.h"
#include "LocationHelper.h"

ValueRef<bool> SpecificationEvaluationService::getRef(BoolMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<BoolMember>(member)) {
        return ValueRef{.value = &(parameters.**std::get<BoolMember>(member)).value};
    } else if (std::holds_alternative<BoolBaseZoneMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = &(parameters.**std::get<BoolBaseZoneMember>(member)).baseValue};
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return ValueRef{
                .value = &(parameters.**std::get<BoolBaseZoneMember>(member)).zoneValues[index].value,
                .disabledValue = &(parameters.**std::get<BoolBaseZoneMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<BoolBaseZoneMember>(member)).zoneValues[index].enabled};
        }
        }
    } else if (std::holds_alternative<BoolZoneMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{.value = &(parameters.**std::get<BoolZoneMember>(member)).zoneValues[index]};
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixBoolMember>(member)) {
        return ValueRef{.value = reinterpret_cast<bool*>((parameters.**std::get<ColorMatrixBoolMember>(member)).value)};
    }

    return {};
}

ValueRef<int> SpecificationEvaluationService::getRef(IntMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<IntMember>(member)) {
        return ValueRef{.value = &(parameters.**std::get<IntMember>(member)).value};
    } else if (std::holds_alternative<IntEnableableMember>(member)) {
        return ValueRef{
            .value = &(parameters.**std::get<IntEnableableMember>(member)).value, .enabled = &(parameters.**std::get<IntEnableableMember>(member)).enabled};
    } else if (std::holds_alternative<IntZoneMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{.value = &(parameters.**std::get<IntZoneMember>(member)).zoneValues[index]};
    }

    // Color vector
    else if (std::holds_alternative<ColorVectorIntMember>(member)) {
        return ValueRef{.value = (parameters.**std::get<ColorVectorIntMember>(member)).value};
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixIntMember>(member)) {
        return ValueRef{.value = reinterpret_cast<int*>((parameters.**std::get<ColorMatrixIntMember>(member)).value)};
    }
    return {};
}

ValueRef<float> SpecificationEvaluationService::getRef(FloatMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<FloatMember>(member)) {
        return ValueRef{.value = &(parameters.**std::get<FloatMember>(member)).value};
    } else if (std::holds_alternative<FloatBaseZoneMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = &(parameters.**std::get<FloatBaseZoneMember>(member)).baseValue};
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return ValueRef{
                .value = &(parameters.**std::get<FloatBaseZoneMember>(member)).zoneValues[index].value,
                .disabledValue = &(parameters.**std::get<FloatBaseZoneMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<FloatBaseZoneMember>(member)).zoneValues[index].enabled};
        }
        }
    } else if (std::holds_alternative<FloatPinMember>(member)) {
        return ValueRef<float>{
            .pinned = &(parameters.**std::get<FloatPinMember>(member)).pinned};
    } else if (std::holds_alternative<FloatZoneMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{.value = &(parameters.**std::get<FloatZoneMember>(member)).zoneValues[index]};
    } else if (std::holds_alternative<FloatEnableableSourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{
            .value = &(parameters.**std::get<FloatEnableableSourceMember>(member)).sourceValues[index].value,
            .disabledValue = &(parameters.**std::get<FloatEnableableSourceMember>(member)).sourceValues[index].value,
            .enabled = &(parameters.**std::get<FloatEnableableSourceMember>(member)).sourceValues[index].enabled};
    } else if (std::holds_alternative<FloatPinnableSourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{
            .value = &(parameters.**std::get<FloatPinnableSourceMember>(member)).sourceValues[index].value,
            .pinned = &(parameters.**std::get<FloatPinnableSourceMember>(member)).sourceValues[index].pinned,
        };
    }
    
    // Color vector
    else if (std::holds_alternative<ColorVectorFloatMember>(member)) {
        return ValueRef{.value = (parameters.**std::get<ColorVectorFloatMember>(member)).value};
    } else if (std::holds_alternative<ColorVectorFloatBaseZoneMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = (parameters.**std::get<ColorVectorFloatBaseZoneMember>(member)).baseValue};
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return ValueRef{
                .value = (parameters.**std::get<ColorVectorFloatBaseZoneMember>(member)).zoneValues[index].value,
                .disabledValue = (parameters.**std::get<ColorVectorFloatBaseZoneMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<ColorVectorFloatBaseZoneMember>(member)).zoneValues[index].enabled};
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixFloatMember>(member)) {
        return ValueRef{.value = reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatMember>(member)).value)};
    } else if (std::holds_alternative<ColorMatrixFloatBaseZoneMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatBaseZoneMember>(member)).baseValue)};
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return ValueRef{
                .value = reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatBaseZoneMember>(member)).zoneValues[index].value),
                .disabledValue = reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatBaseZoneMember>(member)).baseValue),
                .enabled = &(parameters.**std::get<ColorMatrixFloatBaseZoneMember>(member)).zoneValues[index].enabled};
        }
        }
    }

    return {};
}

ValueRef<RealVector2D> SpecificationEvaluationService::getRef(Float2MemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<Float2ZoneMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{.value = &(parameters.**std::get<Float2ZoneMember>(member)).zoneValues[index]};
    } else if (std::holds_alternative<Float2SourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{.value = &(parameters.**std::get<Float2SourceMember>(member)).sourceValues[index]};
    }

    return {};
}

ValueRef<char> SpecificationEvaluationService::getRef(Char64MemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<Char64Member>(member)) {
        return ValueRef{.value = (parameters.**std::get<Char64Member>(member)).value};
    } else if (std::holds_alternative<Char64ZoneMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{.value = (parameters.**std::get<Char64ZoneMember>(member)).zoneValues[index]};
    } else if (std::holds_alternative<Char64SourceMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{.value = (parameters.**std::get<Char64SourceMember>(member)).sourceValues[index]};
    }

    return {};
}

ValueRef<int> SpecificationEvaluationService::getRef(AlternativeMemberVariant const& member, SimulationParameters& parameters, int locationIndex)
    const
{
    // Single value
    if (std::holds_alternative<IntMember>(member)) {
        return ValueRef{.value = &(parameters.**std::get<IntMember>(member)).value};
    } else if (std::holds_alternative<IntZoneMember>(member)) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return ValueRef{.value = &(parameters.**std::get<IntZoneMember>(member)).zoneValues[index]};
    }

    return {};
}

ValueRef<FloatColorRGB>
SpecificationEvaluationService::getRef(FloatColorRGBMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<FloatColorRGBBaseZoneMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = &(parameters.**std::get<FloatColorRGBBaseZoneMember>(member)).baseValue};
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return ValueRef{
                .value = &(parameters.**std::get<FloatColorRGBBaseZoneMember>(member)).zoneValues[index].value,
                .disabledValue = &(parameters.**std::get<FloatColorRGBBaseZoneMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<FloatColorRGBBaseZoneMember>(member)).zoneValues[index].enabled};
        }
        }
    }
    return {};
}

ValueRef<ColorTransitionRules> SpecificationEvaluationService::getRef(ColorTransitionRulesMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<ColorTransitionRulesBaseZoneMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return ValueRef{.value = &(parameters.**std::get<ColorTransitionRulesBaseZoneMember>(member)).baseValue};
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return ValueRef{
                .value = &(parameters.**std::get<ColorTransitionRulesBaseZoneMember>(member)).zoneValues[index].value,
                .disabledValue = &(parameters.**std::get<ColorTransitionRulesBaseZoneMember>(member)).baseValue,
                .enabled = &(parameters.**std::get<ColorTransitionRulesBaseZoneMember>(member)).zoneValues[index].enabled};
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
                || std::holds_alternative<BoolBaseZoneMember>(boolSpec._member)) {
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
                || std::holds_alternative<FloatBaseZoneMember>(floatSpec._member) || std::holds_alternative<ColorVectorFloatBaseZoneMember>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatBaseZoneMember>(floatSpec._member)) {
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
            if (std::holds_alternative<FloatColorRGBBaseZoneMember>(colorPickerSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorTransitionRulesBaseZoneMember>(colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }
    if (locationType == LocationType::Zone) {
        if (std::holds_alternative<BoolSpec>(parameterSpec._reference)) {
            auto const& boolSpec = std::get<BoolSpec>(parameterSpec._reference);
            if (std::holds_alternative<BoolBaseZoneMember>(boolSpec._member) || std::holds_alternative<BoolZoneMember>(boolSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorVectorFloatBaseZoneMember>(floatSpec._member)
                || std::holds_alternative<ColorMatrixFloatBaseZoneMember>(floatSpec._member)
                || std::holds_alternative<FloatZoneMember>(floatSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<Float2Spec>(parameterSpec._reference)) {
            auto const& float2Spec = std::get<Float2Spec>(parameterSpec._reference);
            if (std::holds_alternative<Float2ZoneMember>(float2Spec._member)) {
                return true;
            }
        } else if (std::holds_alternative<Char64Spec>(parameterSpec._reference)) {
            auto const& char64Spec = std::get<Char64Spec>(parameterSpec._reference);
            if (std::holds_alternative<Char64ZoneMember>(char64Spec._member)) {
                return true;
            }
        } else if (std::holds_alternative<AlternativeSpec>(parameterSpec._reference)) {
            auto const& alternativeSpec = std::get<AlternativeSpec>(parameterSpec._reference);
            if (std::holds_alternative<IntZoneMember>(alternativeSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorPickerSpec>(parameterSpec._reference)) {
            auto const& colorPickerSpec = std::get<ColorPickerSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatColorRGBBaseZoneMember>(colorPickerSpec._member)) {
                return true;
            }
        } else if (std::holds_alternative<ColorTransitionRulesSpec>(parameterSpec._reference)) {
            auto const& colorTransitionRulesSpec = std::get<ColorTransitionRulesSpec>(parameterSpec._reference);
            if (std::holds_alternative<ColorTransitionRulesBaseZoneMember>(colorTransitionRulesSpec._member)) {
                return true;
            }
        }
    }
    if (locationType == LocationType::Source) {
        if (std::holds_alternative<FloatSpec>(parameterSpec._reference)) {
            auto const& floatSpec = std::get<FloatSpec>(parameterSpec._reference);
            if (std::holds_alternative<FloatEnableableSourceMember>(floatSpec._member)
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
        }
    }

    return false;
}
