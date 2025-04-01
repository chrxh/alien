#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "Base/Definitions.h"

#include "Definitions.h"
#include "SimulationParametersTypes.h"
#include "Colors.h"

using _BoolMember = bool SimulationParameters::*;
using BoolMember = std::shared_ptr<_BoolMember>;

using _IntMember = int SimulationParameters::*;
using IntMember = std::shared_ptr<_IntMember>;

using _FloatMember = float SimulationParameters::*;
using FloatMember = std::shared_ptr<_FloatMember>;

using _ColorVectorIntMember = ColorVector<int> SimulationParameters::*;
using ColorVectorIntMember = std::shared_ptr<_ColorVectorIntMember >;

using _ColorVectorFloatMember = ColorVector<float> SimulationParameters::*;
using ColorVectorFloatMember = std::shared_ptr<_ColorVectorFloatMember>;

using _ColorMatrixBoolMember = ColorMatrix<bool> SimulationParameters::*;
using ColorMatrixBoolMember = std::shared_ptr<_ColorMatrixBoolMember>;

using _ColorMatrixIntMember = ColorMatrix<int> SimulationParameters::*;
using ColorMatrixIntMember = std::shared_ptr<_ColorMatrixIntMember>;

using _ColorMatrixFloatMember = ColorMatrix<float> SimulationParameters::*;
using ColorMatrixFloatMember = std::shared_ptr<_ColorMatrixFloatMember>;

using _BoolZoneValuesMember = bool SimulationParametersZoneValues::*;
using BoolZoneValuesMember = std::shared_ptr<_BoolZoneValuesMember>;

using _FloatZoneValuesMember = float SimulationParametersZoneValues::*;
using FloatZoneValuesMember = std::shared_ptr<_FloatZoneValuesMember>;

using _ColorVectorFloatZoneValuesMember = ColorVector<float> SimulationParametersZoneValues::*;
using ColorVectorFloatZoneValuesMember = std::shared_ptr<_ColorVectorFloatZoneValuesMember>;

using _ColorMatrixFloatZoneValuesMember = ColorMatrix<float> SimulationParametersZoneValues::*;
using ColorMatrixFloatZoneValuesMember = std::shared_ptr<_ColorMatrixFloatZoneValuesMember>;

using _Char64Member = Char64 SimulationParameters::*;
using Char64Member = std::shared_ptr<_Char64Member>;

using _FloatColorRGBZoneMember = FloatColorRGB SimulationParametersZoneValues::*;
using FloatColorRGBZoneMember = std::shared_ptr<_FloatColorRGBZoneMember>;

using _ColorTransitionRulesZoneMember = ColorTransitionRules SimulationParametersZoneValues::*;
using ColorTransitionRulesZoneMember = std::shared_ptr<_ColorTransitionRulesZoneMember>;

using FloatGetterSetter =
    std::pair<std::function<float(SimulationParameters const&, int)>, std::function<void(float, SimulationParameters&, int)>>;  // int for locationIndex

using _BaseEnabledMember = bool SimulationParameters::*;
using BaseEnabledMember = std::shared_ptr<_BaseEnabledMember>;

using _ZoneEnabledMember = bool SimulationParametersZoneEnabledValues::*;
using ZoneEnabledMember = std::shared_ptr<_ZoneEnabledMember>;

using _SourceEnabledMember = bool RadiationSource::*;
using SourceEnabledMember = std::shared_ptr<_SourceEnabledMember>;

using BoolMemberSpec = std::variant<std::monostate, BoolMember, BoolZoneValuesMember, ColorMatrixBoolMember>;

struct BoolSpec
{
    SETTER_SHARED_PTR(BoolSpec, BoolMember, member);
    SETTER_SHARED_PTR(BoolSpec, BoolZoneValuesMember, member);
    SETTER_SHARED_PTR(BoolSpec, ColorMatrixBoolMember, member);
    BoolMemberSpec _member = std::monostate();
};

using IntMemberSpec = std::variant<std::monostate, IntMember, ColorVectorIntMember, ColorMatrixIntMember>;

struct IntSpec
{
    SETTER_SHARED_PTR(IntSpec, IntMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorVectorIntMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorMatrixIntMember, member);
    IntMemberSpec _member = std::monostate();

    MEMBER(IntSpec, int, min, 0);
    MEMBER(IntSpec, int, max, 0);
    MEMBER(IntSpec, bool, logarithmic, false);
    MEMBER(IntSpec, bool, infinity, false);
};

using FloatMemberSpec = std::variant<
    std::monostate,
    FloatMember,
    ColorVectorFloatMember,
    ColorMatrixFloatMember,
    FloatZoneValuesMember,
    ColorVectorFloatZoneValuesMember,
    ColorMatrixFloatZoneValuesMember,
    FloatGetterSetter>;

struct FloatSpec
{
    SETTER_SHARED_PTR(FloatSpec, FloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatZoneValuesMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatZoneValuesMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatZoneValuesMember, member);
    SETTER(FloatSpec, FloatGetterSetter, member);
    FloatMemberSpec _member = std::monostate();

    MEMBER(FloatSpec, float, min, 0);
    MEMBER(FloatSpec, float, max, 0);
    MEMBER(FloatSpec, bool, logarithmic, false);
    MEMBER(FloatSpec, std::string, format, "%.3f");
    MEMBER(FloatSpec, bool, infinity, false);
};

using Char64MemberSpec = std::variant<std::monostate, Char64Member>;

struct Char64Spec
{
    SETTER_SHARED_PTR(Char64Spec, Char64Member, member);
    Char64MemberSpec _member = std::monostate();
};

using AlternativeMemberSpec = std::variant<std::monostate, IntMember>;

struct ParameterSpec;
struct AlternativeSpec
{
    SETTER_SHARED_PTR(AlternativeSpec, IntMember, member);
    AlternativeMemberSpec _member = std::monostate();

    using Alternatives = std::vector<std::pair<std::string, std::vector<ParameterSpec>>>;
    MEMBER(AlternativeSpec, Alternatives, alternatives, {});
};

using ColorPickerMemberSpec = std::variant<std::monostate, FloatColorRGBZoneMember>;

struct ColorPickerSpec
{
    SETTER_SHARED_PTR(ColorPickerSpec, FloatColorRGBZoneMember, member);
    ColorPickerMemberSpec _member = std::monostate();
};

using ColorTransitionRulesMemberSpec = std::variant<std::monostate, ColorTransitionRulesZoneMember>;

struct ColorTransitionRulesSpec
{
    SETTER_SHARED_PTR(ColorTransitionRulesSpec, ColorTransitionRulesZoneMember, member);
    ColorTransitionRulesMemberSpec _member = std::monostate();
};

struct EnabledSpec
{
    SETTER_SHARED_PTR(EnabledSpec, BaseEnabledMember, base);
    BaseEnabledMember _base;

    SETTER_SHARED_PTR(EnabledSpec, ZoneEnabledMember, zone);
    ZoneEnabledMember _zone;

    SETTER_SHARED_PTR(EnabledSpec, SourceEnabledMember, source);
    SourceEnabledMember _source;

};

using ReferenceSpec = std::variant<BoolSpec, IntSpec, FloatSpec, Char64Spec, AlternativeSpec, ColorPickerSpec, ColorTransitionRulesSpec>;

struct ParameterSpec
{
    MEMBER(ParameterSpec, std::string, name, std::string());
    MEMBER(ParameterSpec, ReferenceSpec, reference, FloatSpec());
    MEMBER(ParameterSpec, EnabledSpec, enabled, EnabledSpec());
    MEMBER(ParameterSpec, std::optional<std::string>, tooltip, std::nullopt);
};

struct ParameterGroupSpec
{
    MEMBER(ParameterGroupSpec, std::string, name, std::string());
    MEMBER(ParameterGroupSpec, std::vector<ParameterSpec>, parameters, {});
    MEMBER(ParameterGroupSpec, std::optional<size_t>, expertToggleAddress, std::nullopt);
    MEMBER(ParameterGroupSpec, std::optional<std::string>, tooltip, std::nullopt);
};

struct ParametersSpec
{
    MEMBER(ParametersSpec, std::vector<ParameterGroupSpec>, groups, {});
};
