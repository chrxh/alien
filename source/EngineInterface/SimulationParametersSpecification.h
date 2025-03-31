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

using BoolMemberSpec = std::variant<std::monostate, BoolMember, BoolZoneValuesMember, ColorMatrixBoolMember>;
using IntMemberSpec = std::variant<std::monostate, IntMember, ColorVectorIntMember, ColorMatrixIntMember>;
using FloatMemberSpec = std::variant<std::monostate, FloatMember, ColorVectorFloatMember, ColorMatrixFloatMember, FloatZoneValuesMember, ColorVectorFloatZoneValuesMember, ColorMatrixFloatZoneValuesMember, FloatGetterSetter>;
using Char64MemberSpec = std::variant<std::monostate, Char64Member>;
using AlternativeMemberSpec = std::variant<std::monostate, IntMember>;
using ColorPickerMemberSpec = std::variant<std::monostate, FloatColorRGBZoneMember>;
using ColorTransitionRulesMemberSpec = std::variant<std::monostate, ColorTransitionRulesZoneMember>;

struct BoolSpec
{
    SETTER_SHARED_PTR(BoolSpec, BoolMember, member);
    SETTER_SHARED_PTR(BoolSpec, BoolZoneValuesMember, member);
    SETTER_SHARED_PTR(BoolSpec, ColorMatrixBoolMember, member);
    BoolMemberSpec _member = std::monostate();
};

struct IntSpec
{
    SETTER_SHARED_PTR(IntSpec, IntMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorVectorIntMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorMatrixIntMember, member);
    IntMemberSpec _member = std::monostate();

    MEMBER_DECLARATION(IntSpec, int, min, 0);
    MEMBER_DECLARATION(IntSpec, int, max, 0);
    MEMBER_DECLARATION(IntSpec, bool, logarithmic, false);
    MEMBER_DECLARATION(IntSpec, bool, infinity, false);
};

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

    MEMBER_DECLARATION(FloatSpec, float, min, 0);
    MEMBER_DECLARATION(FloatSpec, float, max, 0);
    MEMBER_DECLARATION(FloatSpec, bool, logarithmic, false);
    MEMBER_DECLARATION(FloatSpec, std::string, format, "%.3f");
    MEMBER_DECLARATION(FloatSpec, bool, infinity, false);
};

struct Char64Spec
{
    SETTER_SHARED_PTR(Char64Spec, Char64Member, member);
    Char64MemberSpec _member = std::monostate();
};

struct ParameterSpec;
struct AlternativeSpec
{
    SETTER_SHARED_PTR(AlternativeSpec, IntMember, member);
    AlternativeMemberSpec _member = std::monostate();

    using Alternatives = std::vector<std::pair<std::string, std::vector<ParameterSpec>>>;
    MEMBER_DECLARATION(AlternativeSpec, Alternatives, alternatives, {});
};

struct ColorPickerSpec
{
    SETTER_SHARED_PTR(ColorPickerSpec, FloatColorRGBZoneMember, member);
    ColorPickerMemberSpec _member = std::monostate();
};

struct ColorTransitionRulesSpec
{
    SETTER_SHARED_PTR(ColorTransitionRulesSpec, ColorTransitionRulesZoneMember, member);
    ColorTransitionRulesMemberSpec _member = std::monostate();
};

using ReferenceSpec = std::variant<BoolSpec, IntSpec, FloatSpec, Char64Spec, AlternativeSpec, ColorPickerSpec, ColorTransitionRulesSpec>;

using BaseEnabledMember = std::shared_ptr<bool SimulationParameters::*>;
using ZoneEnabledMember = std::shared_ptr<bool SimulationParametersZoneEnabledValues::*>;
using SourceEnabledMember = std::shared_ptr<bool RadiationSource::*>;

using EnabledMember = std::variant<std::monostate, BaseEnabledMember>;
struct EnabledSpec
{
    MEMBER_DECLARATION(EnabledSpec, BaseEnabledMember, base, nullptr);
    MEMBER_DECLARATION(EnabledSpec, ZoneEnabledMember, zone, nullptr);
    MEMBER_DECLARATION(EnabledSpec, SourceEnabledMember, source, nullptr);
};

struct ParameterSpec
{
    MEMBER_DECLARATION(ParameterSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterSpec, ReferenceSpec, reference, FloatSpec());
    MEMBER_DECLARATION(ParameterSpec, EnabledSpec, enabled, EnabledSpec());
    MEMBER_DECLARATION(ParameterSpec, std::optional<std::string>, tooltip, std::nullopt);
};

struct ParameterGroupSpec
{
    MEMBER_DECLARATION(ParameterGroupSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterGroupSpec, std::vector<ParameterSpec>, parameters, {});
    MEMBER_DECLARATION(ParameterGroupSpec, std::optional<size_t>, expertToggleAddress, std::nullopt);
    MEMBER_DECLARATION(ParameterGroupSpec, std::optional<std::string>, tooltip, std::nullopt);
};

struct ParametersSpec
{
    MEMBER_DECLARATION(ParametersSpec, std::vector<ParameterGroupSpec>, groups, {});
};
