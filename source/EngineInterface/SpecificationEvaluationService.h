#pragma once

#include "Base/Singleton.h"
#include "LocationHelper.h"
#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"
#include "SimulationParametersTypes.h"

enum class ColorDependence
{
    None,
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
    ColorDependence colorDependence = ColorDependence::None;
};

class SpecificationEvaluationService
{
    MAKE_SINGLETON(SpecificationEvaluationService);

public:
    ValueRef<bool> getRef(BoolMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const;
    ValueRef<int> getRef(IntMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const;
    ValueRef<float> getRef(FloatMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const;
    ValueRef<RealVector2D> getRef(Float2MemberVariant const& member, SimulationParameters& parameters, int orderNumber) const;
    ValueRef<Char64> getRef(Char64MemberVariant const& member, SimulationParameters& parameters, int orderNumber) const;
    ValueRef<int> getRef(AlternativeMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const;
    ValueRef<FloatColorRGB> getRef(FloatColorRGBMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const;
    ValueRef<ColorTransitionRule> getRef(ColorTransitionRulesMemberVariant const& member, SimulationParameters& parameters, int orderNumber) const;

    bool* getExpertToggleRef(ExpertToggleMember const& member, SimulationParameters& parameters) const;

    bool isVisible(ParameterGroupSpec const& groupSpec, LocationType locationType) const;
    bool isVisible(ParameterSpec const& parameterSpec, LocationType locationType) const;
};
