#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include <boost/variant.hpp>

#include "Base/Definitions.h"

#include "Definitions.h"
#include "SimulationParametersTypes.h"
#include "Colors.h"

using BoolMember = bool SimulationParameters::*;
using IntMember = int SimulationParameters::*;
using FloatMember = float SimulationParameters::*;
using ColorVectorIntMember = ColorVector<int> SimulationParameters::*;
using ColorVectorFloatMember = ColorVector<float> SimulationParameters::*;
using ColorMatrixBoolMember = ColorMatrix<bool> SimulationParameters::*;
using ColorMatrixIntMember = ColorMatrix<int> SimulationParameters::*;
using ColorMatrixFloatMember = ColorMatrix<float> SimulationParameters::*;

using BoolZoneValuesMember = bool SimulationParametersZoneValues::*;
using FloatZoneValuesMember = float SimulationParametersZoneValues::*;
using ColorVectorFloatZoneValuesMember = ColorVector<float> SimulationParametersZoneValues::*;
using ColorMatrixFloatZoneValuesMember = ColorMatrix<float> SimulationParametersZoneValues::*;

using Char64Member = Char64 SimulationParameters::*;
using FloatColorRGBZoneMember = FloatColorRGB SimulationParametersZoneValues::*;
using ColorTransitionRulesZoneMember = ColorTransitionRules SimulationParametersZoneValues::*;

using FloatGetterSetter =
    std::pair<std::function<float(SimulationParameters const&, int)>, std::function<void(float, SimulationParameters&, int)>>;  // int for locationIndex

using BoolMemberSpec = boost::variant<std::monostate, BoolMember, BoolZoneValuesMember, ColorMatrixBoolMember>;
using IntMemberSpec = boost::variant<std::monostate, IntMember, ColorVectorIntMember, ColorMatrixIntMember>;
using FloatMemberSpec = boost::variant<
    std::monostate,
    FloatMember,
    ColorVectorFloatMember,
    ColorMatrixFloatMember,
    FloatZoneValuesMember,
    ColorVectorFloatZoneValuesMember,
    ColorMatrixFloatZoneValuesMember,
    FloatGetterSetter>;
using Char64MemberSpec = boost::variant<std::monostate, Char64Member>;
using AlternativeMemberSpec = boost::variant<std::monostate, IntMember>;
using ColorPickerMemberSpec = boost::variant<std::monostate, FloatColorRGBZoneMember>;
using ColorTransitionRulesMemberSpec = boost::variant<std::monostate, ColorTransitionRulesZoneMember>;

struct BoolSpec
{
    MEMBER_DECLARATION(BoolSpec, BoolMemberSpec, member, std::monostate());
};

struct IntSpec
{
    MEMBER_DECLARATION(IntSpec, IntMemberSpec, member, std::monostate());
    MEMBER_DECLARATION(IntSpec, int, min, 0);
    MEMBER_DECLARATION(IntSpec, int, max, 0);
    MEMBER_DECLARATION(IntSpec, bool, logarithmic, false);
    MEMBER_DECLARATION(IntSpec, bool, infinity, false);
};

struct FloatSpec
{
    MEMBER_DECLARATION(FloatSpec, FloatMemberSpec, member, std::monostate());
    MEMBER_DECLARATION(FloatSpec, float, min, 0);
    MEMBER_DECLARATION(FloatSpec, float, max, 0);
    MEMBER_DECLARATION(FloatSpec, bool, logarithmic, false);
    MEMBER_DECLARATION(FloatSpec, std::string, format, "%.3f");
    MEMBER_DECLARATION(FloatSpec, bool, infinity, false);
};

struct Char64Spec
{
    MEMBER_DECLARATION(Char64Spec, Char64MemberSpec, member, std::monostate());
};

struct ParameterSpec;
struct AlternativeSpec
{
    MEMBER_DECLARATION(
        AlternativeSpec,
        std::shared_ptr<AlternativeMemberSpec>,
        member,
        nullptr);  // Workaround: shared_ptr to avoid aligned delete on memory coming from an unaligned allocation

    using Alternatives = std::vector<std::pair<std::string, std::vector<ParameterSpec>>>;
    MEMBER_DECLARATION(AlternativeSpec, Alternatives, alternatives, {});
};

struct ColorPickerSpec
{
    MEMBER_DECLARATION(ColorPickerSpec, ColorPickerMemberSpec, member, std::monostate());
};

struct ColorTransitionRulesSpec
{
    MEMBER_DECLARATION(ColorTransitionRulesSpec, ColorTransitionRulesMemberSpec, member, std::monostate());
};

using ReferenceSpec = boost::variant<BoolSpec, IntSpec, FloatSpec, Char64Spec, AlternativeSpec, ColorPickerSpec, ColorTransitionRulesSpec>;


struct BaseValueSpec
{
    MEMBER_DECLARATION(BaseValueSpec, std::optional<size_t>, valueAddress, std::nullopt);
    MEMBER_DECLARATION(BaseValueSpec, std::optional<std::function<float(SimulationParameters const&, int)>>, valueGetter, std::nullopt);  // int for locationIndex
    MEMBER_DECLARATION(BaseValueSpec, std::optional<std::function<void(float, SimulationParameters&, int)>>, valueSetter, std::nullopt);  // int for locationIndex
    MEMBER_DECLARATION(BaseValueSpec, std::optional<size_t>, pinnedAddress, std::nullopt);
    MEMBER_DECLARATION(BaseValueSpec, std::optional<size_t>, enabledValueAddress, std::nullopt);
};

struct BaseZoneValueSpec
{
    MEMBER_DECLARATION(BaseZoneValueSpec, std::optional<size_t>, valueAddress, std::nullopt);
    MEMBER_DECLARATION(BaseZoneValueSpec, std::optional<size_t>, enabledBaseValueAddress, std::nullopt);
    MEMBER_DECLARATION(BaseZoneValueSpec, std::optional<size_t>, enabledZoneValueAddress, std::nullopt);
};

using ValueSpec = boost::variant<BaseValueSpec, BaseZoneValueSpec>;

struct ParameterSpec
{
    MEMBER_DECLARATION(ParameterSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterSpec, ValueSpec, value, ValueSpec());
    MEMBER_DECLARATION(ParameterSpec, ReferenceSpec, reference, FloatSpec());
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
