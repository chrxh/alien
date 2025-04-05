#pragma once

#include "Base/Singleton.h"
#include "LocationHelper.h"
#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"
#include "SimulationParametersTypes.h"

class SpecificationEvaluationService
{
    MAKE_SINGLETON(SpecificationEvaluationService);

public:
    bool* getBoolRef(BoolMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    int* getIntRef(IntMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    float* getFloatRef(FloatMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    char* getChar64Ref(Char64MemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    int* getAlternativeRef(AlternativeMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    FloatColorRGB* getFloatColorRGBRef(ColorPickerMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ColorTransitionRules* getColorTransitionRulesRef(ColorTransitionRulesMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;

    bool* getEnabledRef(EnabledSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    bool* getExpertToggleRef(BoolMemberNew const& member, SimulationParameters& parameters) const;
};
