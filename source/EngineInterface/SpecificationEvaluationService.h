#pragma once

#include "Base/Singleton.h"
#include "LocationHelper.h"
#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"
#include "SimulationParametersTypes.h"

enum class ValueType
{
    Single,
    ColorVector,
    ColorMatrix
};
template <typename T>
struct ValueRef
{
    T* value = nullptr;
    T* disabledValue = nullptr;
    bool* enabled = nullptr;
    bool* pinned = nullptr;
    ValueType valueType = ValueType::Single;
};

class SpecificationEvaluationService
{
    MAKE_SINGLETON(SpecificationEvaluationService);

public:
    ValueRef<bool> getRef(BoolMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<int> getRef(IntMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<float> getRef(FloatMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<RealVector2D> getRef(Float2MemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<char> getRef(Char64MemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<int> getRef(AlternativeMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<FloatColorRGB> getRef(FloatColorRGBMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;
    ValueRef<ColorTransitionRules> getRef(ColorTransitionRulesMemberVariant const& member, SimulationParameters& parameters, int locationIndex) const;

    bool* getExpertToggleRef(ExpertToggleMember const& member, SimulationParameters& parameters) const;

    bool isVisible(ParameterGroupSpec const& groupSpec, LocationType locationType) const;
    bool isVisible(ParameterSpec const& parameterSpec, LocationType locationType) const;
};
