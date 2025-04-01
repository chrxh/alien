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

    return nullptr;
}

float* SpecificationEvaluationService::getFloatRef(FloatMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<FloatMember>(memberSpec)) {
        return &(parameters.**std::get<FloatMember>(memberSpec));
    } else if (std::holds_alternative<FloatZoneValuesMember>(memberSpec)) {
        if (locationIndex == 0) {
            return &(parameters.baseValues.**std::get<FloatZoneValuesMember>(memberSpec));
        } else {
            auto index = LocationHelper::findLocationArrayIndex(parameters, locationIndex);
            return &(parameters.zone[index].values.**std::get<FloatZoneValuesMember>(memberSpec));
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

    return nullptr;
}

char* SpecificationEvaluationService::getChar64Ref(Char64MemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    if (std::holds_alternative<Char64Member>(memberSpec)) {
        return parameters.**std::get<Char64Member>(memberSpec);
    }

    return nullptr;
}

int* SpecificationEvaluationService::getAlternativeRef(AlternativeMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const
{
    // Single value
    if (std::holds_alternative<IntMember>(memberSpec)) {
        return &(parameters.**std::get<IntMember>(memberSpec));
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

