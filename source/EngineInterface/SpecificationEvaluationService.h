#pragma once

#include "Base/Singleton.h"
#include "LocationHelper.h"
#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"
#include "SimulationParametersTypes.h"

template <typename T>
struct ValueRef
{
    T* value = nullptr;
    bool* enabled = nullptr;
    bool* pinned = nullptr;
};

class SpecificationEvaluationService
{
    MAKE_SINGLETON(SpecificationEvaluationService);

public:
    ValueRef<bool> getRef(BoolMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<int> getRef(IntMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<float> getRef(FloatMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<char> getRef(Char64MemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<int> getRef(AlternativeMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<FloatColorRGB> getRef(ColorPickerMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<ColorTransitionRules> getRef(ColorTransitionRulesMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;

    bool* getExpertToggleRef(ExpertToggleMemberNew const& member, SimulationParameters& parameters) const;
};
