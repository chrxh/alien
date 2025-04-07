#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "Base/Definitions.h"

#include "Definitions.h"
#include "SimulationParametersTypes.h"
#include "Colors.h"

using _BoolMemberNew = BaseParameter<bool> SimulationParameters::*;
using BoolMemberNew = std::shared_ptr<_BoolMemberNew>;

using _ExpertToggleMemberNew = ExpertToggle SimulationParameters::*;
using ExpertToggleMemberNew = std::shared_ptr<_ExpertToggleMemberNew>;

using _IntMemberNew = BaseParameter<int> SimulationParameters::*;
using IntMemberNew = std::shared_ptr<_IntMemberNew>;

using _IntEnableableMemberNew = EnableableBaseParameter<int> SimulationParameters::*;
using IntEnableableMemberNew = std::shared_ptr<_IntEnableableMemberNew>;

using _FloatMemberNew = BaseParameter<float> SimulationParameters::*;
using FloatMemberNew = std::shared_ptr<_FloatMemberNew>;
using _FloatPinMemberNew = PinBaseParameter SimulationParameters::*;
using FloatPinMemberNew = std::shared_ptr<_FloatPinMemberNew>;

using _ColorVectorIntMemberNew = BaseParameter<ColorVector<int>> SimulationParameters::*;
using ColorVectorIntMemberNew = std::shared_ptr<_ColorVectorIntMemberNew>;

using _ColorVectorFloatMemberNew = BaseParameter<ColorVector<float>> SimulationParameters::*;
using ColorVectorFloatMemberNew = std::shared_ptr<_ColorVectorFloatMemberNew>;

using _ColorMatrixBoolMemberNew = BaseParameter<ColorMatrix<bool>> SimulationParameters::*;
using ColorMatrixBoolMemberNew = std::shared_ptr<_ColorMatrixBoolMemberNew>;

using _ColorMatrixIntMemberNew = BaseParameter<ColorMatrix<int>> SimulationParameters::*;
using ColorMatrixIntMemberNew = std::shared_ptr<_ColorMatrixIntMemberNew>;

using _ColorMatrixFloatMemberNew = BaseParameter<ColorMatrix<float>> SimulationParameters::*;
using ColorMatrixFloatMemberNew = std::shared_ptr<_ColorMatrixFloatMemberNew>;

using _Char64MemberNew = BaseParameter<Char64> SimulationParameters::*;
using Char64MemberNew = std::shared_ptr<_Char64MemberNew>;

using _BoolZoneValuesMemberNew = BaseZoneParameter<bool> SimulationParameters::*;
using BoolZoneValuesMemberNew = std::shared_ptr<_BoolZoneValuesMemberNew>;

using _FloatZoneValuesMemberNew = BaseZoneParameter<float> SimulationParameters::*;
using FloatZoneValuesMemberNew = std::shared_ptr<_FloatZoneValuesMemberNew>;

using _ColorVectorFloatBaseZoneMemberNew = BaseZoneParameter<ColorVector<float>> SimulationParameters::*;
using ColorVectorFloatBaseZoneMemberNew = std::shared_ptr<_ColorVectorFloatBaseZoneMemberNew>;

using _ColorMatrixFloatBaseZoneMemberNew = BaseZoneParameter<ColorMatrix<float>> SimulationParameters::*;
using ColorMatrixFloatBaseZoneMemberNew = std::shared_ptr<_ColorMatrixFloatBaseZoneMemberNew>;

using _FloatColorRGBBaseZoneMemberNew = BaseZoneParameter<FloatColorRGB> SimulationParameters::*;
using FloatColorRGBBaseZoneMemberNew = std::shared_ptr<_FloatColorRGBBaseZoneMemberNew>;

using _ColorTransitionRulesBaseZoneMemberNew = BaseZoneParameter<ColorTransitionRules> SimulationParameters::*;
using ColorTransitionRulesBaseZoneMemberNew = std::shared_ptr<_ColorTransitionRulesBaseZoneMemberNew>;

using FloatGetterSetter =
    std::pair<std::function<float(SimulationParameters const&, int)>, std::function<void(float, SimulationParameters&, int)>>;  // int for locationIndex

using BoolMemberVariant = std::variant<std::monostate, BoolMemberNew, BoolZoneValuesMemberNew, ColorMatrixBoolMemberNew>;

struct BoolSpec
{
    SETTER_SHARED_PTR(BoolSpec, BoolMemberNew, member);
    SETTER_SHARED_PTR(BoolSpec, BoolZoneValuesMemberNew, member);
    SETTER_SHARED_PTR(BoolSpec, ColorMatrixBoolMemberNew, member);
    BoolMemberVariant _member = std::monostate();
};

using IntMemberVariant = std::variant<std::monostate, IntMemberNew, IntEnableableMemberNew, ColorVectorIntMemberNew, ColorMatrixIntMemberNew>;

struct IntSpec
{
    SETTER_SHARED_PTR(IntSpec, IntMemberNew, member);
    SETTER_SHARED_PTR(IntSpec, IntEnableableMemberNew, member);
    SETTER_SHARED_PTR(IntSpec, ColorVectorIntMemberNew, member);
    SETTER_SHARED_PTR(IntSpec, ColorMatrixIntMemberNew, member);
    IntMemberVariant _member = std::monostate();

    MEMBER(IntSpec, int, min, 0);
    MEMBER(IntSpec, int, max, 0);
    MEMBER(IntSpec, bool, logarithmic, false);
    MEMBER(IntSpec, bool, infinity, false);
};

using FloatMemberVariant = std::variant<
    std::monostate,
    FloatMemberNew,
    FloatPinMemberNew,
    ColorVectorFloatMemberNew,
    ColorMatrixFloatMemberNew,
    FloatZoneValuesMemberNew,
    ColorVectorFloatBaseZoneMemberNew,
    ColorMatrixFloatBaseZoneMemberNew>;

struct FloatSpec
{
    SETTER_SHARED_PTR(FloatSpec, FloatMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, FloatPinMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, FloatZoneValuesMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatBaseZoneMemberNew, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatBaseZoneMemberNew, member);
    FloatMemberVariant _member = std::monostate();

    MEMBER(FloatSpec, std::optional<FloatGetterSetter>, getterSetter, std::nullopt);

    MEMBER(FloatSpec, float, min, 0);
    MEMBER(FloatSpec, float, max, 0);
    MEMBER(FloatSpec, bool, logarithmic, false);
    MEMBER(FloatSpec, std::string, format, "%.3f");
    MEMBER(FloatSpec, bool, infinity, false);
};

using Char64MemberVariant = std::variant<std::monostate, Char64MemberNew>;

struct Char64Spec
{
    SETTER_SHARED_PTR(Char64Spec, Char64MemberNew, member);
    Char64MemberVariant _member = std::monostate();
};

using AlternativeMemberVariant = std::variant<std::monostate, IntMemberNew>;

struct ParameterSpec;
struct AlternativeSpec
{
    SETTER_SHARED_PTR(AlternativeSpec, IntMemberNew, member);
    AlternativeMemberVariant _member = std::monostate();

    using Alternatives = std::vector<std::pair<std::string, std::vector<ParameterSpec>>>;
    MEMBER(AlternativeSpec, Alternatives, alternatives, {});
};

using ColorPickerMemberVariant = std::variant<std::monostate, FloatColorRGBBaseZoneMemberNew>;

struct ColorPickerSpec
{
    SETTER_SHARED_PTR(ColorPickerSpec, FloatColorRGBBaseZoneMemberNew, member);
    ColorPickerMemberVariant _member = std::monostate();
};

using ColorTransitionRulesMemberVariant = std::variant<std::monostate, ColorTransitionRulesBaseZoneMemberNew>;

struct ColorTransitionRulesSpec
{
    SETTER_SHARED_PTR(ColorTransitionRulesSpec, ColorTransitionRulesBaseZoneMemberNew, member);
    ColorTransitionRulesMemberVariant _member = std::monostate();
};

using ReferenceSpec = std::variant<BoolSpec, IntSpec, FloatSpec, Char64Spec, AlternativeSpec, ColorPickerSpec, ColorTransitionRulesSpec>;

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

    SETTER_SHARED_PTR(ParameterGroupSpec, ExpertToggleMemberNew, expertToggle);
    ExpertToggleMemberNew _expertToggle;
};

struct ParametersSpec
{
    MEMBER(ParametersSpec, std::vector<ParameterGroupSpec>, groups, {});
};
