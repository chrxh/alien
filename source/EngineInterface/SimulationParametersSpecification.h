#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "Base/Definitions.h"

#include "Definitions.h"
#include "SimulationParametersTypes.h"
#include "Colors.h"

using _BoolMember = BaseParameter<bool> SimulationParameters::*;
using BoolMember = std::shared_ptr<_BoolMember>;
using _BoolZoneMember = ZoneParameter<bool> SimulationParameters::*;
using BoolZoneMember = std::shared_ptr<_BoolZoneMember>;
using _ColorMatrixBoolMember = BaseParameter<ColorMatrix<bool>> SimulationParameters::*;
using ColorMatrixBoolMember = std::shared_ptr<_ColorMatrixBoolMember>;
using _BoolBaseZoneMember = BaseZoneParameter<bool> SimulationParameters::*;
using BoolBaseZoneMember = std::shared_ptr<_BoolBaseZoneMember>;
using _ExpertToggleMember = ExpertToggle SimulationParameters::*;
using ExpertToggleMember = std::shared_ptr<_ExpertToggleMember>;
using BoolMemberVariant = std::variant<std::monostate, BoolMember, BoolBaseZoneMember, ColorMatrixBoolMember, BoolZoneMember>;

using _IntMember = BaseParameter<int> SimulationParameters::*;
using IntMember = std::shared_ptr<_IntMember>;
using _IntEnableableMember = EnableableBaseParameter<int> SimulationParameters::*;
using IntEnableableMember = std::shared_ptr<_IntEnableableMember>;
using _ColorVectorIntMember = BaseParameter<ColorVector<int>> SimulationParameters::*;
using ColorVectorIntMember = std::shared_ptr<_ColorVectorIntMember>;
using _ColorMatrixIntMember = BaseParameter<ColorMatrix<int>> SimulationParameters::*;
using ColorMatrixIntMember = std::shared_ptr<_ColorMatrixIntMember>;
using IntMemberVariant = std::variant<std::monostate, IntMember, IntEnableableMember, ColorVectorIntMember, ColorMatrixIntMember>;
using AlternativeMemberVariant = std::variant<std::monostate, IntMember>;

using _FloatMember = BaseParameter<float> SimulationParameters::*;
using FloatMember = std::shared_ptr<_FloatMember>;
using _FloatPinMember = PinBaseParameter SimulationParameters::*;
using FloatPinMember = std::shared_ptr<_FloatPinMember>;
using _ColorVectorFloatMember = BaseParameter<ColorVector<float>> SimulationParameters::*;
using ColorVectorFloatMember = std::shared_ptr<_ColorVectorFloatMember>;
using _ColorMatrixFloatMember = BaseParameter<ColorMatrix<float>> SimulationParameters::*;
using ColorMatrixFloatMember = std::shared_ptr<_ColorMatrixFloatMember>;
using _FloatBaseZoneMember = BaseZoneParameter<float> SimulationParameters::*;
using FloatBaseZoneMember = std::shared_ptr<_FloatBaseZoneMember>;
using _ColorVectorFloatBaseZoneMember = BaseZoneParameter<ColorVector<float>> SimulationParameters::*;
using ColorVectorFloatBaseZoneMember = std::shared_ptr<_ColorVectorFloatBaseZoneMember>;
using _ColorMatrixFloatBaseZoneMember = BaseZoneParameter<ColorMatrix<float>> SimulationParameters::*;
using ColorMatrixFloatBaseZoneMember = std::shared_ptr<_ColorMatrixFloatBaseZoneMember>;
using FloatGetterSetter =
    std::pair<std::function<float(SimulationParameters const&, int)>, std::function<void(float, SimulationParameters&, int)>>;  // int for locationIndex
using FloatMemberVariant = std::variant<
    std::monostate,
    FloatMember,
    FloatPinMember,
    ColorVectorFloatMember,
    ColorMatrixFloatMember,
    FloatBaseZoneMember,
    ColorVectorFloatBaseZoneMember,
    ColorMatrixFloatBaseZoneMember>;

using _Float2ZoneMember = ZoneParameter<RealVector2D> SimulationParameters::*;
using Float2ZoneMember = std::shared_ptr<_Float2ZoneMember>;
using Float2MemberVariant = std::variant<std::monostate, Float2ZoneMember>;

using _Char64Member = BaseParameter<Char64> SimulationParameters::*;
using Char64Member = std::shared_ptr<_Char64Member>;
using _Char64ZoneMember = ZoneParameter<Char64> SimulationParameters::*;
using Char64ZoneMember = std::shared_ptr<_Char64ZoneMember>;
using Char64MemberVariant = std::variant<std::monostate, Char64Member, Char64ZoneMember>;

using _FloatColorRGBBaseZoneMember = BaseZoneParameter<FloatColorRGB> SimulationParameters::*;
using FloatColorRGBBaseZoneMember = std::shared_ptr<_FloatColorRGBBaseZoneMember>;
using FloatColorRGBMemberVariant = std::variant<std::monostate, FloatColorRGBBaseZoneMember>;

using _ColorTransitionRulesBaseZoneMember = BaseZoneParameter<ColorTransitionRules> SimulationParameters::*;
using ColorTransitionRulesBaseZoneMember = std::shared_ptr<_ColorTransitionRulesBaseZoneMember>;
using ColorTransitionRulesMemberVariant = std::variant<std::monostate, ColorTransitionRulesBaseZoneMember>;

struct BoolSpec
{
    SETTER_SHARED_PTR(BoolSpec, BoolMember, member);
    SETTER_SHARED_PTR(BoolSpec, BoolBaseZoneMember, member);
    SETTER_SHARED_PTR(BoolSpec, ColorMatrixBoolMember, member);
    SETTER_SHARED_PTR(BoolSpec, BoolZoneMember, member);
    BoolMemberVariant _member = std::monostate();
};

struct IntSpec
{
    SETTER_SHARED_PTR(IntSpec, IntMember, member);
    SETTER_SHARED_PTR(IntSpec, IntEnableableMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorVectorIntMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorMatrixIntMember, member);
    IntMemberVariant _member = std::monostate();

    MEMBER(IntSpec, int, min, 0);
    MEMBER(IntSpec, int, max, 0);
    MEMBER(IntSpec, bool, logarithmic, false);
    MEMBER(IntSpec, bool, infinity, false);
};

struct FloatSpec
{
    SETTER_SHARED_PTR(FloatSpec, FloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatPinMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatBaseZoneMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatBaseZoneMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatBaseZoneMember, member);
    FloatMemberVariant _member = std::monostate();

    MEMBER(FloatSpec, std::optional<FloatGetterSetter>, getterSetter, std::nullopt);

    MEMBER(FloatSpec, float, min, 0);
    MEMBER(FloatSpec, float, max, 0);
    MEMBER(FloatSpec, bool, logarithmic, false);
    MEMBER(FloatSpec, std::string, format, "%.3f");
    MEMBER(FloatSpec, bool, infinity, false);
};

struct WorldSize
{};
using Min2Variant = std::variant<RealVector2D>;
using Max2Variant = std::variant<RealVector2D, WorldSize>;

struct Float2Spec
{
    SETTER_SHARED_PTR(Float2Spec, Float2ZoneMember, member);
    Float2MemberVariant _member = std::monostate();

    MEMBER(Float2Spec, Min2Variant, min, RealVector2D());
    MEMBER(Float2Spec, Max2Variant, max, RealVector2D());
};

struct Char64Spec
{
    SETTER_SHARED_PTR(Char64Spec, Char64Member, member);
    SETTER_SHARED_PTR(Char64Spec, Char64ZoneMember, member);
    Char64MemberVariant _member = std::monostate();
};

struct ParameterSpec;
struct AlternativeSpec
{
    SETTER_SHARED_PTR(AlternativeSpec, IntMember, member);
    AlternativeMemberVariant _member = std::monostate();

    using Alternatives = std::vector<std::pair<std::string, std::vector<ParameterSpec>>>;
    MEMBER(AlternativeSpec, Alternatives, alternatives, {});
};

struct ColorPickerSpec
{
    SETTER_SHARED_PTR(ColorPickerSpec, FloatColorRGBBaseZoneMember, member);
    FloatColorRGBMemberVariant _member = std::monostate();
};

struct ColorTransitionRulesSpec
{
    SETTER_SHARED_PTR(ColorTransitionRulesSpec, ColorTransitionRulesBaseZoneMember, member);
    ColorTransitionRulesMemberVariant _member = std::monostate();
};

using ReferenceSpec = std::variant<BoolSpec, IntSpec, FloatSpec, Float2Spec, Char64Spec, AlternativeSpec, ColorPickerSpec, ColorTransitionRulesSpec>;

struct ParameterSpec
{
    MEMBER(ParameterSpec, std::string, name, std::string());
    MEMBER(ParameterSpec, ReferenceSpec, reference, FloatSpec());
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
