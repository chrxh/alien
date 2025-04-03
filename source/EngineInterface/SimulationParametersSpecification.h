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
using _BoolMemberNew = BaseParameter<bool> SimulationParameters::*;
using BoolMemberNew = std::shared_ptr<_BoolMemberNew>;

using _IntMember = int SimulationParameters::*;
using IntMember = std::shared_ptr<_IntMember>;
using _IntMemberNew = BaseParameter<int> SimulationParameters::*;
using IntMemberNew = std::shared_ptr<_IntMemberNew>;

using _FloatMember = float SimulationParameters::*;
using FloatMember = std::shared_ptr<_FloatMember>;
using _FloatMemberNew = BaseParameter<float> SimulationParameters::*;
using FloatMemberNew = std::shared_ptr<_FloatMemberNew>;

using _ColorVectorIntMember = ColorVector<int> SimulationParameters::*;
using ColorVectorIntMember = std::shared_ptr<_ColorVectorIntMember >;
using _ColorVectorIntMemberNew = BaseParameter<ColorVector<int>> SimulationParameters::*;
using ColorVectorIntMemberNew = std::shared_ptr<_ColorVectorIntMemberNew>;

using _ColorVectorFloatMember = ColorVector<float> SimulationParameters::*;
using ColorVectorFloatMember = std::shared_ptr<_ColorVectorFloatMember>;
using _ColorVectorFloatMemberNew = BaseParameter<ColorVector<float>> SimulationParameters::*;
using ColorVectorFloatMemberNew = std::shared_ptr<_ColorVectorFloatMemberNew>;

using _ColorMatrixBoolMember = ColorMatrix<bool> SimulationParameters::*;
using ColorMatrixBoolMember = std::shared_ptr<_ColorMatrixBoolMember>;
using _ColorMatrixBoolMemberNew = BaseParameter<ColorMatrix<bool>> SimulationParameters::*;
using ColorMatrixBoolMemberNew = std::shared_ptr<_ColorMatrixBoolMemberNew>;

using _ColorMatrixIntMember = ColorMatrix<int> SimulationParameters::*;
using ColorMatrixIntMember = std::shared_ptr<_ColorMatrixIntMember>;
using _ColorMatrixIntMemberNew = BaseParameter<ColorMatrix<int>> SimulationParameters::*;
using ColorMatrixIntMemberNew = std::shared_ptr<_ColorMatrixIntMemberNew>;

using _ColorMatrixFloatMember = ColorMatrix<float> SimulationParameters::*;
using ColorMatrixFloatMember = std::shared_ptr<_ColorMatrixFloatMember>;
using _ColorMatrixFloatMemberNew = BaseParameter<ColorMatrix<float>> SimulationParameters::*;
using ColorMatrixFloatMemberNew = std::shared_ptr<_ColorMatrixFloatMemberNew>;

using _Char64Member = Char64 SimulationParameters::*;
using Char64Member = std::shared_ptr<_Char64Member>;
using _Char64MemberNew = BaseParameter<Char64> SimulationParameters::*;
using Char64MemberNew = std::shared_ptr<_Char64MemberNew>;

using _BoolZoneValuesMember = bool SimulationParametersZoneValues::*;
using BoolZoneValuesMember = std::shared_ptr<_BoolZoneValuesMember>;
using _BoolZoneValuesMemberNew = BaseZoneParameter<bool> SimulationParameters::*;
using BoolZoneValuesMemberNew = std::shared_ptr<_BoolZoneValuesMemberNew>;

using _FloatZoneValuesMember = float SimulationParametersZoneValues::*;
using FloatZoneValuesMember = std::shared_ptr<_FloatZoneValuesMember>;
using _FloatZoneValuesMemberNew = BaseZoneParameter<float> SimulationParameters::*;
using FloatZoneValuesMemberNew = std::shared_ptr<_FloatZoneValuesMemberNew>;

using _ColorVectorFloatZoneValuesMember = ColorVector<float> SimulationParametersZoneValues::*;
using ColorVectorFloatZoneValuesMember = std::shared_ptr<_ColorVectorFloatZoneValuesMember>;
using _ColorVectorFloatBaseZoneMemberNew = BaseZoneParameter<ColorVector<float>> SimulationParameters::*;
using ColorVectorFloatBaseZoneMemberNew = std::shared_ptr<_ColorVectorFloatBaseZoneMemberNew>;

using _ColorMatrixFloatZoneValuesMember = ColorMatrix<float> SimulationParametersZoneValues::*;
using ColorMatrixFloatZoneValuesMember = std::shared_ptr<_ColorMatrixFloatZoneValuesMember>;
using _ColorMatrixFloatBaseZoneMemberNew = BaseZoneParameter<ColorMatrix<float>> SimulationParameters::*;
using ColorMatrixFloatBaseZoneMemberNew = std::shared_ptr<_ColorMatrixFloatBaseZoneMemberNew>;

using _FloatColorRGBZoneMember = FloatColorRGB SimulationParametersZoneValues::*;
using FloatColorRGBZoneMember = std::shared_ptr<_FloatColorRGBZoneMember>;
using _FloatColorRGBBaseZoneMemberNew = BaseZoneParameter<FloatColorRGB> SimulationParameters::*;
using FloatColorRGBBaseZoneMemberNew = std::shared_ptr<_FloatColorRGBBaseZoneMemberNew>;

using _ColorTransitionRulesZoneMember = ColorTransitionRules SimulationParametersZoneValues::*;
using ColorTransitionRulesZoneMember = std::shared_ptr<_ColorTransitionRulesZoneMember>;
using _ColorTransitionRulesBaseZoneMemberNew = BaseZoneParameter<ColorTransitionRules> SimulationParameters::*;
using ColorTransitionRulesBaseZoneMemberNew = std::shared_ptr<_ColorTransitionRulesBaseZoneMemberNew>;

using FloatGetterSetter =
    std::pair<std::function<float(SimulationParameters const&, int)>, std::function<void(float, SimulationParameters&, int)>>;  // int for locationIndex

using _BaseEnabledMember = bool SimulationParameters::*;
using BaseEnabledMember = std::shared_ptr<_BaseEnabledMember>;

using _ZoneEnabledMember = bool SimulationParametersZoneEnabledValues::*;
using ZoneEnabledMember = std::shared_ptr<_ZoneEnabledMember>;

using _SourceEnabledMember = bool RadiationSource::*;
using SourceEnabledMember = std::shared_ptr<_SourceEnabledMember>;

using _ExpertToggleMember = bool ExpertToggles::*;
using ExpertToggleMember = std::shared_ptr<_ExpertToggleMember>;

using BoolMemberSpec = std::variant<std::monostate, BoolMember, BoolZoneValuesMember, ColorMatrixBoolMember, BoolMemberNew, BoolZoneValuesMemberNew, ColorMatrixBoolMemberNew>;

struct PinnableBaseValueSpecNew
{
    SETTER_SHARED_PTR(PinnableBaseValueSpecNew, BoolMemberNew, pinnedMember);
    BoolMemberNew _pinnedMember;

    MEMBER(PinnableBaseValueSpecNew, FloatGetterSetter, getterSetter, {});
};

struct BoolSpec
{
    SETTER_SHARED_PTR(BoolSpec, BoolMember, member);
    SETTER_SHARED_PTR(BoolSpec, BoolZoneValuesMember, member);
    SETTER_SHARED_PTR(BoolSpec, ColorMatrixBoolMember, member);
    SETTER_SHARED_PTR(BoolSpec, BoolMemberNew, member);
    SETTER_SHARED_PTR(BoolSpec, BoolZoneValuesMemberNew, member);
    SETTER_SHARED_PTR(BoolSpec, ColorMatrixBoolMemberNew, member);
    BoolMemberSpec _member = std::monostate();
};

using IntMemberSpec = std::variant<std::monostate, IntMember, ColorVectorIntMember, ColorMatrixIntMember, IntMemberNew, ColorVectorIntMemberNew, ColorMatrixIntMemberNew>;

struct IntSpec
{
    SETTER_SHARED_PTR(IntSpec, IntMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorVectorIntMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorMatrixIntMember, member);
    SETTER_SHARED_PTR(IntSpec, IntMemberNew, member);
    SETTER_SHARED_PTR(IntSpec, ColorVectorIntMemberNew, member);
    SETTER_SHARED_PTR(IntSpec, ColorMatrixIntMemberNew, member);
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
    FloatMemberNew,
    ColorVectorFloatMemberNew,
    ColorMatrixFloatMemberNew,
    FloatZoneValuesMemberNew,
    ColorVectorFloatBaseZoneMemberNew,
    ColorMatrixFloatBaseZoneMemberNew,
    PinnableBaseValueSpecNew>;

struct FloatSpec
{
    SETTER_SHARED_PTR(FloatSpec, FloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatZoneValuesMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatZoneValuesMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatZoneValuesMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, FloatZoneValuesMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatBaseZoneMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatBaseZoneMemberNew, member);
    SETTER(FloatSpec, PinnableBaseValueSpecNew, member);
    FloatMemberSpec _member = std::monostate();

    MEMBER(FloatSpec, float, min, 0);
    MEMBER(FloatSpec, float, max, 0);
    MEMBER(FloatSpec, bool, logarithmic, false);
    MEMBER(FloatSpec, std::string, format, "%.3f");
    MEMBER(FloatSpec, bool, infinity, false);
};

using Char64MemberSpec = std::variant<std::monostate, Char64Member, Char64MemberNew>;

struct Char64Spec
{
    SETTER_SHARED_PTR(Char64Spec, Char64Member, member);
    SETTER_SHARED_PTR(Char64Spec, Char64MemberNew, member);
    Char64MemberSpec _member = std::monostate();
};

using AlternativeMemberSpec = std::variant<std::monostate, IntMember, IntMemberNew>;

struct ParameterSpec;
struct AlternativeSpec
{
    SETTER_SHARED_PTR(AlternativeSpec, IntMember, member);
    SETTER_SHARED_PTR(AlternativeSpec, IntMemberNew, member);
    AlternativeMemberSpec _member = std::monostate();

    using Alternatives = std::vector<std::pair<std::string, std::vector<ParameterSpec>>>;
    MEMBER(AlternativeSpec, Alternatives, alternatives, {});
};

using ColorPickerMemberSpec = std::variant<std::monostate, FloatColorRGBZoneMember, FloatColorRGBBaseZoneMemberNew>;

struct ColorPickerSpec
{
    SETTER_SHARED_PTR(ColorPickerSpec, FloatColorRGBZoneMember, member);
    SETTER_SHARED_PTR(ColorPickerSpec, FloatColorRGBBaseZoneMemberNew, member);
    ColorPickerMemberSpec _member = std::monostate();
};

using ColorTransitionRulesMemberSpec = std::variant<std::monostate, ColorTransitionRulesZoneMember, ColorTransitionRulesBaseZoneMemberNew>;

struct ColorTransitionRulesSpec
{
    SETTER_SHARED_PTR(ColorTransitionRulesSpec, ColorTransitionRulesZoneMember, member);
    SETTER_SHARED_PTR(ColorTransitionRulesSpec, ColorTransitionRulesBaseZoneMemberNew, member);
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
    MEMBER(ParameterSpec, bool, visible, true);
    MEMBER(ParameterSpec, std::optional<std::string>, tooltip, std::nullopt);
};

struct ParameterGroupSpec
{
    MEMBER(ParameterGroupSpec, std::string, name, std::string());
    MEMBER(ParameterGroupSpec, std::vector<ParameterSpec>, parameters, {});
    MEMBER(ParameterGroupSpec, std::optional<std::string>, tooltip, std::nullopt);

    SETTER_SHARED_PTR(ParameterGroupSpec, ExpertToggleMember, expertToggle);
    ExpertToggleMember _expertToggle;
};

struct ParametersSpec
{
    MEMBER(ParametersSpec, std::vector<ParameterGroupSpec>, groups, {});
};
