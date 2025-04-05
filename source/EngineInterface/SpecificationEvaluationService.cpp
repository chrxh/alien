#include <boost/variant.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include "CellTypeStrings.h"
#include "LocationHelper.h"
#include "ParametersEditService.h"
#include "SpecificationEvaluationService.h"

bool* SpecificationEvaluationService::getBoolRef(BoolMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<BoolMember>(member)) {
        return &(parameters.**std::get<BoolMember>(member));
    } else if (std::holds_alternative<BoolZoneValuesMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<BoolZoneValuesMember>(member));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<BoolZoneValuesMember>(member));
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixBoolMember>(member)) {
        return reinterpret_cast<bool*>(parameters.**std::get<ColorMatrixBoolMember>(member));
    }

    // NEW
    // Single value
    if (std::holds_alternative<BoolMemberNew>(member)) {
        return &(parameters.**std::get<BoolMemberNew>(member)).value;
    } else if (std::holds_alternative<BoolZoneValuesMemberNew>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.**std::get<BoolZoneValuesMemberNew>(member)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.**std::get<BoolZoneValuesMemberNew>(member)).zoneValues[index].value;
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixBoolMemberNew>(member)) {
        return reinterpret_cast<bool*>((parameters.**std::get<ColorMatrixBoolMemberNew>(member)).value);
    }

    return nullptr;
}

int* SpecificationEvaluationService::getIntRef(IntMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<IntMember>(member)) {
        return &(parameters.**std::get<IntMember>(member));
    }

    // Color vector
    else if (std::holds_alternative<ColorVectorIntMember>(member)) {
        return parameters.**std::get<ColorVectorIntMember>(member);
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixIntMember>(member)) {
        return reinterpret_cast<int*>(parameters.**std::get<ColorMatrixIntMember>(member));
    }

    // NEW
    // Single value
    if (std::holds_alternative<IntMemberNew>(member)) {
        return &(parameters.**std::get<IntMemberNew>(member)).value;
    }

    // Color vector
    else if (std::holds_alternative<ColorVectorIntMemberNew>(member)) {
        return (parameters.**std::get<ColorVectorIntMemberNew>(member)).value;
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixIntMemberNew>(member)) {
        return reinterpret_cast<int*>((parameters.**std::get<ColorMatrixIntMemberNew>(member)).value);
    }
    return nullptr;
}

float* SpecificationEvaluationService::getFloatRef(FloatMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<FloatMember>(member)) {
        return &(parameters.**std::get<FloatMember>(member));
    } else if (std::holds_alternative<FloatZoneValuesMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<FloatZoneValuesMember>(member));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<FloatZoneValuesMember>(member));
        }
        }
    }

    // Color vector
    else if (std::holds_alternative<ColorVectorFloatMember>(member)) {
        return parameters.**std::get<ColorVectorFloatMember>(member);
    } else if (std::holds_alternative<ColorVectorFloatZoneValuesMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return parameters.baseValues.**std::get<ColorVectorFloatZoneValuesMember>(member);
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return parameters.zone[index].values.**std::get<ColorVectorFloatZoneValuesMember>(member);
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixFloatMember>(member)) {
        return reinterpret_cast<float*>(parameters.**std::get<ColorMatrixFloatMember>(member));
    } else if (std::holds_alternative<ColorMatrixFloatZoneValuesMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return reinterpret_cast<float*>(parameters.baseValues.**std::get<ColorMatrixFloatZoneValuesMember>(member));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return reinterpret_cast<float*>(parameters.zone[index].values.**std::get<ColorMatrixFloatZoneValuesMember>(member));
        }
        }
    }

    // NEW
    // Single value
    if (std::holds_alternative<FloatMemberNew>(member)) {
        return &(parameters.**std::get<FloatMemberNew>(member)).value;
    } else if (std::holds_alternative<FloatZoneValuesMemberNew>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.**std::get<FloatZoneValuesMemberNew>(member)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.**std::get<FloatZoneValuesMemberNew>(member)).zoneValues[index].value;
        }
        }
    }
    
    // Color vector
    else if (std::holds_alternative<ColorVectorFloatMemberNew>(member)) {
        return (parameters.**std::get<ColorVectorFloatMemberNew>(member)).value;
    } else if (std::holds_alternative<ColorVectorFloatBaseZoneMemberNew>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return (parameters.**std::get<ColorVectorFloatBaseZoneMemberNew>(member)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return (parameters.**std::get<ColorVectorFloatBaseZoneMemberNew>(member)).zoneValues[index].value;
        }
        }
    }

    // Color matrix
    else if (std::holds_alternative<ColorMatrixFloatMemberNew>(member)) {
        return reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatMemberNew>(member)).value);
    } else if (std::holds_alternative<ColorMatrixFloatBaseZoneMemberNew>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatBaseZoneMemberNew>(member)).baseValue);
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return reinterpret_cast<float*>((parameters.**std::get<ColorMatrixFloatBaseZoneMemberNew>(member)).zoneValues[index].value);
        }
        }
    }

    return nullptr;
}

char* SpecificationEvaluationService::getChar64Ref(Char64MemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<Char64Member>(member)) {
        return parameters.**std::get<Char64Member>(member);
    }

    // NEW
    if (std::holds_alternative<Char64MemberNew>(member)) {
        return (parameters.**std::get<Char64MemberNew>(member)).value;
    }

    return nullptr;
}

int* SpecificationEvaluationService::getAlternativeRef(AlternativeMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<IntMember>(member)) {
        return &(parameters.**std::get<IntMember>(member));
    }

    // NEW
    // Single value
    if (std::holds_alternative<IntMemberNew>(member)) {
        return &(parameters.**std::get<IntMemberNew>(member)).value;
    }

    return nullptr;
}

FloatColorRGB* SpecificationEvaluationService::getFloatColorRGBRef(ColorPickerMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<FloatColorRGBZoneMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<FloatColorRGBZoneMember>(member));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<FloatColorRGBZoneMember>(member));
        }
        }
    }

    // NEW
    if (std::holds_alternative<FloatColorRGBBaseZoneMemberNew>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.**std::get<FloatColorRGBBaseZoneMemberNew>(member)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.**std::get<FloatColorRGBBaseZoneMemberNew>(member)).zoneValues[index].value;
        }
        }
    }
    return nullptr;
}

ColorTransitionRules*
SpecificationEvaluationService::getColorTransitionRulesRef(ColorTransitionRulesMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<ColorTransitionRulesZoneMember>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.baseValues.**std::get<ColorTransitionRulesZoneMember>(member));
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<ColorTransitionRulesZoneMember>(member));
        }
        }
    }

    // NEW
    if (std::holds_alternative<ColorTransitionRulesBaseZoneMemberNew>(member)) {
        switch (LocationHelper::getLocationType(locationIndex, parameters)) {
        case LocationType::Base:
            return &(parameters.**std::get<ColorTransitionRulesBaseZoneMemberNew>(member)).baseValue;
        case LocationType::Zone: {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.**std::get<ColorTransitionRulesBaseZoneMemberNew>(member)).zoneValues[index].value;
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

bool* SpecificationEvaluationService::getExpertToggleRef(BoolMemberNew const& member, SimulationParameters& parameters) const
{
    return &(parameters.**member).value;
}

