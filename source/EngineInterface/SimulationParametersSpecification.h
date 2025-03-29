#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "Base/Definitions.h"

#include "Definitions.h"
#include "SimulationParametersTypes.h"
#include "Colors.h"

struct BoolSpec
{
};

struct IntSpec
{
    MEMBER_DECLARATION(IntSpec, int, min, 0);
    MEMBER_DECLARATION(IntSpec, int, max, 0);
    MEMBER_DECLARATION(IntSpec, bool, logarithmic, false);
    MEMBER_DECLARATION(IntSpec, bool, infinity, false);
};

struct FloatSpec
{
    MEMBER_DECLARATION(FloatSpec, float, min, 0);
    MEMBER_DECLARATION(FloatSpec, float, max, 0);
    MEMBER_DECLARATION(FloatSpec, bool, logarithmic, false);
    MEMBER_DECLARATION(FloatSpec, std::string, format, "%.3f");
    MEMBER_DECLARATION(FloatSpec, bool, infinity, false);
};

struct Char64Spec
{
};

struct ParameterSpec;
struct AlternativeSpec
{
    using Alternatives = std::vector<std::pair<std::string, std::vector<ParameterSpec>>>;
    MEMBER_DECLARATION(AlternativeSpec, Alternatives, alternatives, {});
};

struct ColorPickerSpec
{
};

struct ColorTransitionSpec
{
};

using TypeSpec = std::variant<BoolSpec, IntSpec, FloatSpec, Char64Spec, AlternativeSpec, ColorPickerSpec, ColorTransitionSpec>;

using Char64Member = Char64 SimulationParameters::*;
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
using ColorVectorFloatZoneMember = ColorVector<float> SimulationParametersZoneValues::*;
using ColorMatrixFloatZoneValuesMember = ColorMatrix<float> SimulationParametersZoneValues::*;
using ColorTransitionRulesMember = ColorTransitionRules SimulationParametersZoneValues::*;

using FloatGetterSetter =
    std::pair<std::function<float(SimulationParameters const&, int)>, std::function<void(float, SimulationParameters&, int)>>;  // int for locationIndex

using MemberSpec = std::variant<
    std::monostate,
    Char64Member,
    BoolMember,
    IntMember,
    FloatMember,
    ColorVectorIntMember,
    ColorVectorFloatMember,
    ColorMatrixBoolMember,
    ColorMatrixIntMember,
    ColorMatrixFloatMember,
    BoolZoneValuesMember,
    FloatZoneValuesMember,
    ColorVectorFloatZoneMember,
    ColorMatrixFloatZoneValuesMember,
    ColorTransitionRulesMember,
    FloatGetterSetter>;

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

using ValueSpec = std::variant<BaseValueSpec, BaseZoneValueSpec>;

enum class ColorDependence
{
    None,
    Vector,
    Matrix
};
struct ParameterSpec
{
    MEMBER_DECLARATION(ParameterSpec, MemberSpec, member, std::monostate());
    MEMBER_DECLARATION(ParameterSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterSpec, ValueSpec, value, ValueSpec());
    MEMBER_DECLARATION(ParameterSpec, TypeSpec, type, FloatSpec());
    MEMBER_DECLARATION(ParameterSpec, ColorDependence, colorDependence, ColorDependence::None);
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
