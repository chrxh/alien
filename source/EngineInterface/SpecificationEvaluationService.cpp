#include <boost/variant.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include "CellTypeStrings.h"
#include "LocationHelper.h"
#include "ParametersEditService.h"
#include "SpecificationEvaluationService.h"

bool* SpecificationEvaluationService::getBoolRef(BoolMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<BoolMember>(memberSpec)) {
        return &(parameters.**std::get<BoolMember>(memberSpec));
    } else if (std::holds_alternative<BoolZoneValuesMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<BoolZoneValuesMember>(memberSpec));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<BoolZoneValuesMember>(memberSpec));
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixBoolMember>(memberSpec)) {
        return reinterpret_cast<bool*>(parameters.**std::get<ColorMatrixBoolMember>(memberSpec));
    }

    // NEW
    // Single value
    if (std::holds_alternative<BoolMemberNew>(memberSpec)) {
        return &(parameters.**std::get<BoolMemberNew>(memberSpec)).value;
    } else if (std::holds_alternative<BoolZoneValuesMemberNew>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.**std::get<BoolZoneValuesMemberNew>(memberSpec)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.**std::get<BoolZoneValuesMemberNew>(memberSpec)).zoneValues[index].value;
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixBoolMemberNew>(memberSpec)) {
        return reinterpret_cast<bool*>((parameters.**std::get<ColorMatrixBoolMemberNew>(memberSpec)).value);
    }

    return nullptr;
}

int* SpecificationEvaluationService::getIntRef(IntMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<IntMember>(memberSpec)) {
        return &(parameters.**std::get<IntMember>(memberSpec));
    }

    // Color vector
    else if (std::holds_alternative<ColorVectorIntMember>(memberSpec)) {
        return parameters.**std::get<ColorVectorIntMember>(memberSpec);
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixIntMember>(memberSpec)) {
        return reinterpret_cast<int*>(parameters.**std::get<ColorMatrixIntMember>(memberSpec));
    }

    // NEW
    // Single value
    if (std::holds_alternative<IntMemberNew>(memberSpec)) {
        return &(parameters.**std::get<IntMemberNew>(memberSpec)).value;
    }

    // Color vector
    else if (std::holds_alternative<ColorVectorIntMemberNew>(memberSpec)) {
        return (parameters.**std::get<ColorVectorIntMemberNew>(memberSpec)).value;
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixIntMemberNew>(memberSpec)) {
        return reinterpret_cast<int*>((parameters.**std::get<ColorMatrixIntMemberNew>(memberSpec)).value);
    }
    return nullptr;
}

float* SpecificationEvaluationService::getFloatRef(FloatMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<FloatMember>(memberSpec)) {
        return &(parameters.**std::get<FloatMember>(memberSpec));
    } else if (std::holds_alternative<FloatZoneValuesMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<FloatZoneValuesMember>(memberSpec));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<FloatZoneValuesMember>(memberSpec));
        }
        }
    }

    // Color vector
    else if (std::holds_alternative<ColorVectorFloatMember>(memberSpec)) {
        return parameters.**std::get<ColorVectorFloatMember>(memberSpec);
    } else if (std::holds_alternative<ColorVectorFloatZoneValuesMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return parameters.baseValues.**std::get<ColorVectorFloatZoneValuesMember>(memberSpec);
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return parameters.zone[index].values.**std::get<ColorVectorFloatZoneValuesMember>(memberSpec);
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixFloatMember>(memberSpec)) {
        return reinterpret_cast<float*>(parameters.**std::get<ColorMatrixFloatMember>(memberSpec));
    } else if (std::holds_alternative<ColorMatrixFloatZoneValuesMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return reinterpret_cast<float*>(parameters.baseValues.**std::get<ColorMatrixFloatZoneValuesMember>(memberSpec));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return reinterpret_cast<float*>(parameters.zone[index].values.**std::get<ColorMatrixFloatZoneValuesMember>(memberSpec));
        }
        }
    }

    // NEW
    // Single value
    if (std::holds_alternative<FloatMemberNew>(memberSpec)) {
        return &(parameters.**std::get<FloatMemberNew>(memberSpec)).value;
    } else if (std::holds_alternative<FloatZoneValuesMemberNew>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.**std::get<FloatZoneValuesMemberNew>(memberSpec)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.**std::get<FloatZoneValuesMemberNew>(memberSpec)).zoneValues[index].value;
        }
        }
    }
    
    // Color vector
    else if (std::holds_alternative<ColorVectorFloatMemberNew>(memberSpec)) {
        return (parameters.**std::get<ColorVectorFloatMemberNew>(memberSpec)).value;
    } else if (std::holds_alternative<ColorVectorFloatZoneValuesMemberNew>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return (parameters.**std::get<ColorVectorFloatZoneValuesMemberNew>(memberSpec)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return (parameters.**std::get<ColorVectorFloatZoneValuesMemberNew>(memberSpec)).zoneValues[index].value;
        }
        }
    }

    return nullptr;
}

char* SpecificationEvaluationService::getChar64Ref(Char64MemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<Char64Member>(memberSpec)) {
        return parameters.**std::get<Char64Member>(memberSpec);
    }

    // NEW
    if (std::holds_alternative<Char64MemberNew>(memberSpec)) {
        return (parameters.**std::get<Char64MemberNew>(memberSpec)).value;
    }

    return nullptr;
}

int* SpecificationEvaluationService::getAlternativeRef(AlternativeMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<IntMember>(memberSpec)) {
        return &(parameters.**std::get<IntMember>(memberSpec));
    }

    // NEW
    // Single value
    if (std::holds_alternative<IntMemberNew>(memberSpec)) {
        return &(parameters.**std::get<IntMemberNew>(memberSpec)).value;
    }

    return nullptr;
}

FloatColorRGB* SpecificationEvaluationService::getFloatColorRGBRef(ColorPickerMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<FloatColorRGBZoneMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<FloatColorRGBZoneMember>(memberSpec));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<FloatColorRGBZoneMember>(memberSpec));
        }
        }
    }

    // NEW
    if (std::holds_alternative<FloatColorRGBZoneMemberNew>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.**std::get<FloatColorRGBZoneMemberNew>(memberSpec)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.**std::get<FloatColorRGBZoneMemberNew>(memberSpec)).zoneValues[index].value;
        }
        }
    }
    return nullptr;
}

ColorTransitionRules*
SpecificationEvaluationService::getColorTransitionRulesRef(ColorTransitionRulesMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<ColorTransitionRulesZoneMember>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<ColorTransitionRulesZoneMember>(memberSpec));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<ColorTransitionRulesZoneMember>(memberSpec));
        }
        }
    }

    // NEW
    if (std::holds_alternative<ColorTransitionRulesZoneMemberNew>(memberSpec)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.**std::get<ColorTransitionRulesZoneMemberNew>(memberSpec)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.**std::get<ColorTransitionRulesZoneMemberNew>(memberSpec)).zoneValues[index].value;
        }
        }
    }
    return nullptr;
}

bool* SpecificationEvaluationService::getEnabledRef(EnabledSpec const& spec, SimulationParameters& parameters, int locationIndex) const
{
    auto locationType = LocationHelper::getLocationType(locationIndex, parameters);
    if (spec._base && locationType == LocationType::Base) {
        return &(parameters.**spec._base.get());
    }
    if (spec._zone && locationType == LocationType::Zone) {
        auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
        return &(parameters.zone[index].enabledValues.**spec._zone.get());
    }
    return nullptr;
}

//bool* SimulationParametersSpecificationEvaluationService::getPinnedValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const
//{
//    if (std::get<BaseValueSpec>(&spec)) {
//        auto baseValueSpec = std::get<BaseValueSpec>(spec);
//        if (baseValueSpec._pinnedAddress.has_value()) {
//            return reinterpret_cast<bool*>(reinterpret_cast<char*>(&parameters) + baseValueSpec._pinnedAddress.value());
//        }
//    }
//    return nullptr;
//}

bool* SpecificationEvaluationService::getExpertToggleRef(ExpertToggleMember const& expertToggle, SimulationParameters& parameters) const
{
    if (expertToggle) {
        return &(parameters.expertToggles.**expertToggle);
    }
    return nullptr;
}

