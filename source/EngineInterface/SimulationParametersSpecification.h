#pragma once

#include <string>
#include <variant>
#include <vector>
#include <optional>
#include <memory>

#include "Base/Definitions.h"

#include "Definitions.h"
#include "SimulationParametersTypes.h"
#include "Colors.h"

struct SimulationParameters;

using _BoolMember = BaseParameter<bool> SimulationParameters::*;
using BoolMember = std::shared_ptr<_BoolMember>;
using _BoolLayerMember = LayerParameter<bool> SimulationParameters::*;
using BoolLayerMember = std::shared_ptr<_BoolLayerMember>;
using _ColorMatrixBoolMember = BaseParameter<ColorMatrix<bool>> SimulationParameters::*;
using ColorMatrixBoolMember = std::shared_ptr<_ColorMatrixBoolMember>;
using _BoolBaseLayerMember = BaseLayerParameter<bool> SimulationParameters::*;
using BoolBaseLayerMember = std::shared_ptr<_BoolBaseLayerMember>;
using _ExpertToggleMember = ExpertToggle SimulationParameters::*;
using ExpertToggleMember = std::shared_ptr<_ExpertToggleMember>;
using BoolMemberVariant = std::variant<std::monostate, BoolMember, BoolBaseLayerMember, ColorMatrixBoolMember, BoolLayerMember>;

using _IntMember = BaseParameter<int> SimulationParameters::*;
using IntMember = std::shared_ptr<_IntMember>;
using _IntEnableableMember = EnableableBaseParameter<int> SimulationParameters::*;
using IntEnableableMember = std::shared_ptr<_IntEnableableMember>;
using _ColorVectorIntMember = BaseParameter<ColorVector<int>> SimulationParameters::*;
using ColorVectorIntMember = std::shared_ptr<_ColorVectorIntMember>;
using _ColorMatrixIntMember = BaseParameter<ColorMatrix<int>> SimulationParameters::*;
using ColorMatrixIntMember = std::shared_ptr<_ColorMatrixIntMember>;
using _IntLayerMember = LayerParameter<int> SimulationParameters::*;
using IntLayerMember = std::shared_ptr<_IntLayerMember>;
using _IntSourceMember = SourceParameter<int> SimulationParameters::*;
using IntSourceMember = std::shared_ptr<_IntSourceMember>;
using IntMemberVariant = std::variant<std::monostate, IntMember, IntEnableableMember, ColorVectorIntMember, ColorMatrixIntMember, IntLayerMember, IntSourceMember>;
using AlternativeMemberVariant = std::variant<std::monostate, IntMember, IntLayerMember, IntSourceMember>;

using _FloatMember = BaseParameter<float> SimulationParameters::*;
using FloatMember = std::shared_ptr<_FloatMember>;
using _FloatPinMember = PinBaseParameter SimulationParameters::*;
using FloatPinMember = std::shared_ptr<_FloatPinMember>;
using _ColorVectorFloatMember = BaseParameter<ColorVector<float>> SimulationParameters::*;
using ColorVectorFloatMember = std::shared_ptr<_ColorVectorFloatMember>;
using _ColorMatrixFloatMember = BaseParameter<ColorMatrix<float>> SimulationParameters::*;
using ColorMatrixFloatMember = std::shared_ptr<_ColorMatrixFloatMember>;
using _FloatBaseLayerMember = BaseLayerParameter<float> SimulationParameters::*;
using FloatBaseLayerMember = std::shared_ptr<_FloatBaseLayerMember>;
using _ColorVectorFloatBaseLayerMember = BaseLayerParameter<ColorVector<float>> SimulationParameters::*;
using ColorVectorFloatBaseLayerMember = std::shared_ptr<_ColorVectorFloatBaseLayerMember>;
using _ColorMatrixFloatBaseLayerMember = BaseLayerParameter<ColorMatrix<float>> SimulationParameters::*;
using ColorMatrixFloatBaseLayerMember = std::shared_ptr<_ColorMatrixFloatBaseLayerMember>;
using _FloatLayerMember = LayerParameter<float> SimulationParameters::*;
using FloatLayerMember = std::shared_ptr<_FloatLayerMember >;
using _FloatSourceMember = SourceParameter<float> SimulationParameters::*;
using FloatSourceMember = std::shared_ptr<_FloatSourceMember>;
using _FloatEnableableSourceMember = EnableableSourceParameter<float> SimulationParameters::*;
using FloatEnableableSourceMember = std::shared_ptr<_FloatEnableableSourceMember >;
using _FloatPinnableSourceMember = PinnableSourceParameter<float> SimulationParameters::*;
using FloatPinnableSourceMember = std::shared_ptr<_FloatPinnableSourceMember >;
using FloatMemberVariant = std::variant<
    std::monostate,
    FloatMember,
    FloatPinMember,
    ColorVectorFloatMember,
    ColorMatrixFloatMember,
    FloatBaseLayerMember,
    ColorVectorFloatBaseLayerMember,
    ColorMatrixFloatBaseLayerMember,
    FloatLayerMember,
    FloatSourceMember,
    FloatEnableableSourceMember,
    FloatPinnableSourceMember>;
using FloatGetterSetter =
    std::pair<std::function<float(SimulationParameters const&, int)>, std::function<void(float, SimulationParameters&, int)>>;  // int for orderNumber

using _Float2LayerMember = LayerParameter<RealVector2D> SimulationParameters::*;
using Float2LayerMember = std::shared_ptr<_Float2LayerMember>;
using _Float2SourceMember = SourceParameter<RealVector2D> SimulationParameters::*;
using Float2SourceMember = std::shared_ptr<_Float2SourceMember>;
using Float2MemberVariant = std::variant<std::monostate, Float2LayerMember, Float2SourceMember>;

using _Char64Member = BaseParameter<Char64> SimulationParameters::*;
using Char64Member = std::shared_ptr<_Char64Member>;
using _Char64LayerMember = LayerParameter<Char64> SimulationParameters::*;
using Char64LayerMember = std::shared_ptr<_Char64LayerMember>;
using _Char64SourceMember = SourceParameter<Char64> SimulationParameters::*;
using Char64SourceMember = std::shared_ptr<_Char64SourceMember>;
using Char64MemberVariant = std::variant<std::monostate, Char64Member, Char64LayerMember, Char64SourceMember>;

using _FloatColorRGBBaseLayerMember = BaseLayerParameter<FloatColorRGB> SimulationParameters::*;
using FloatColorRGBBaseLayerMember = std::shared_ptr<_FloatColorRGBBaseLayerMember>;
using FloatColorRGBMemberVariant = std::variant<std::monostate, FloatColorRGBBaseLayerMember>;

using _ColorTransitionRulesBaseLayerMember = BaseLayerParameter<ColorVector<ColorTransitionRule>> SimulationParameters::*;
using ColorTransitionRulesBaseLayerMember = std::shared_ptr<_ColorTransitionRulesBaseLayerMember>;
using ColorTransitionRulesMemberVariant = std::variant<std::monostate, ColorTransitionRulesBaseLayerMember>;

struct BoolSpec
{
    SETTER_SHARED_PTR(BoolSpec, BoolMember, member);
    SETTER_SHARED_PTR(BoolSpec, ColorMatrixBoolMember, member);
    SETTER_SHARED_PTR(BoolSpec, BoolBaseLayerMember, member);
    SETTER_SHARED_PTR(BoolSpec, BoolLayerMember, member);
    BoolMemberVariant _member = std::monostate();
};

struct IntSpec
{
    SETTER_SHARED_PTR(IntSpec, IntMember, member);
    SETTER_SHARED_PTR(IntSpec, IntEnableableMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorVectorIntMember, member);
    SETTER_SHARED_PTR(IntSpec, ColorMatrixIntMember, member);
    SETTER_SHARED_PTR(IntSpec, IntLayerMember, member);
    SETTER_SHARED_PTR(IntSpec, IntSourceMember, member);
    IntMemberVariant _member = std::monostate();

    MEMBER(IntSpec, int, min, 0);
    MEMBER(IntSpec, int, max, 0);
    MEMBER(IntSpec, bool, logarithmic, false);
    MEMBER(IntSpec, bool, infinity, false);
};

struct MaxWorldRadiusSize
{};
using FloatMinVariant = std::variant<float>;
using FloatMaxVariant = std::variant<float, MaxWorldRadiusSize>;

struct FloatSpec
{
    SETTER_SHARED_PTR(FloatSpec, FloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatPinMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatBaseLayerMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatBaseLayerMember, member);
    SETTER_SHARED_PTR(FloatSpec, ColorMatrixFloatBaseLayerMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatLayerMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatSourceMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatEnableableSourceMember, member);
    SETTER_SHARED_PTR(FloatSpec, FloatPinnableSourceMember, member);
    FloatMemberVariant _member = std::monostate();

    MEMBER(FloatSpec, std::optional<FloatGetterSetter>, getterSetter, std::nullopt);

    SETTER_SHARED_PTR(FloatSpec, ColorVectorFloatBaseLayerMember, greaterThan);
    FloatMemberVariant _greaterThan = std::monostate();
    MEMBER(FloatSpec, FloatMinVariant, min, 0.0f);
    MEMBER(FloatSpec, FloatMaxVariant, max, 0.0f);
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
    SETTER_SHARED_PTR(Float2Spec, Float2LayerMember, member);
    SETTER_SHARED_PTR(Float2Spec, Float2SourceMember, member);
    Float2MemberVariant _member = std::monostate();

    MEMBER(Float2Spec, Min2Variant, min, RealVector2D());
    MEMBER(Float2Spec, Max2Variant, max, RealVector2D());
    MEMBER(Float2Spec, std::string, format, "%.3f");
    MEMBER(Float2Spec, bool, mousePicker, false);
};

struct Char64Spec
{
    SETTER_SHARED_PTR(Char64Spec, Char64Member, member);
    SETTER_SHARED_PTR(Char64Spec, Char64LayerMember, member);
    SETTER_SHARED_PTR(Char64Spec, Char64SourceMember, member);
    Char64MemberVariant _member = std::monostate();
};

struct ParameterSpec;
struct AlternativeSpec
{
    SETTER_SHARED_PTR(AlternativeSpec, IntMember, member);
    SETTER_SHARED_PTR(AlternativeSpec, IntLayerMember, member);
    SETTER_SHARED_PTR(AlternativeSpec, IntSourceMember, member);
    AlternativeMemberVariant _member = std::monostate();

    using Alternatives = std::vector<std::pair<std::string, std::vector<ParameterSpec>>>;
    MEMBER(AlternativeSpec, Alternatives, alternatives, {});
};

struct ColorSpec
{
    SETTER_SHARED_PTR(ColorSpec, FloatColorRGBBaseLayerMember, member);
    FloatColorRGBMemberVariant _member = std::monostate();
};

struct ColorTransitionRulesSpec
{
    SETTER_SHARED_PTR(ColorTransitionRulesSpec, ColorTransitionRulesBaseLayerMember, member);
    ColorTransitionRulesMemberVariant _member = std::monostate();
};

using ReferenceSpec = std::variant<BoolSpec, IntSpec, FloatSpec, Float2Spec, Char64Spec, AlternativeSpec, ColorSpec, ColorTransitionRulesSpec>;

struct ParameterSpec
{
    MEMBER(ParameterSpec, std::string, name, std::string());
    MEMBER(ParameterSpec, ReferenceSpec, reference, FloatSpec());
    MEMBER(ParameterSpec, bool, visible, true);
    MEMBER(ParameterSpec, std::optional<std::string>, description, std::nullopt);
};

struct ParameterGroupSpec
{
    MEMBER(ParameterGroupSpec, std::string, name, std::string());
    MEMBER(ParameterGroupSpec, std::vector<ParameterSpec>, parameters, {});
    MEMBER(ParameterGroupSpec, std::optional<std::string>, description, std::nullopt);

    SETTER_SHARED_PTR(ParameterGroupSpec, ExpertToggleMember, expertToggle);
    ExpertToggleMember _expertToggle;
};

struct ParametersSpec
{
    MEMBER(ParametersSpec, std::vector<ParameterGroupSpec>, groups, {});
};
