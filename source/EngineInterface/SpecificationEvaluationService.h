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
    bool* getBoolRef(BoolMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    int* getIntRef(IntMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    float* getFloatRef(FloatMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    char* getChar64Ref(Char64MemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    int* getAlternativeRef(AlternativeMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    FloatColorRGB* getFloatColorRGBRef(ColorPickerMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;
    ColorTransitionRules* getColorTransitionRulesRef(ColorTransitionRulesMemberSpec const& memberSpec, SimulationParameters& parameters, int locationIndex) const;

    bool* getEnabledRef(EnabledSpec const& spec, SimulationParameters& parameters, int locationIndex) const;

    //bool* getPinnedValueRef(ValueSpec const& spec, SimulationParameters& parameters, int locationIndex) const;
    bool* getExpertToggleRef(ExpertToggleMember const& expertToggle, SimulationParameters& parameters) const;
};
