#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "Base/Definitions.h"

#include "Definitions.h"


struct BoolSpec
{
};

struct FloatSpec
{
    MEMBER_DECLARATION(FloatSpec, float, min, 0);
    MEMBER_DECLARATION(FloatSpec, float, max, 0);
    MEMBER_DECLARATION(FloatSpec, bool, logarithmic, false);
    MEMBER_DECLARATION(FloatSpec, std::string, format, "%.3f");
    MEMBER_DECLARATION(FloatSpec, bool, infinity, false);
    MEMBER_DECLARATION(FloatSpec, std::optional<std::function<float(SimulationParameters const&, int)>>, valueGetter, std::nullopt);  // int for locationIndex
    MEMBER_DECLARATION(FloatSpec, std::optional<std::function<void(float, SimulationParameters&, int)>>, valueSetter, std::nullopt);  // int for locationIndex
    MEMBER_DECLARATION(FloatSpec, std::optional<size_t>, pinnedAddress, std::nullopt);
};

struct Char64Spec
{
};

struct ParameterSpec;
struct SwitcherSpec
{
    using Alternatives = std::vector<std::pair<std::string, std::vector<ParameterSpec>>>;
    MEMBER_DECLARATION(SwitcherSpec, Alternatives, alternatives, {});
};

struct ColorSpec
{
};

using TypeSpec = std::variant<BoolSpec, FloatSpec, Char64Spec, SwitcherSpec, ColorSpec>;

struct ParameterSpec
{
    MEMBER_DECLARATION(ParameterSpec, bool, visibleInBase, true);
    MEMBER_DECLARATION(ParameterSpec, bool, visibleInZone, false);
    MEMBER_DECLARATION(ParameterSpec, bool, visibleInSource, false);
    MEMBER_DECLARATION(ParameterSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterSpec, std::optional<size_t>, valueAddress, std::nullopt);
    MEMBER_DECLARATION(ParameterSpec, std::optional<std::string>, tooltip, std::nullopt);
    MEMBER_DECLARATION(ParameterSpec, std::optional<size_t>, valueActivatedAddress, std::nullopt);
    MEMBER_DECLARATION(ParameterSpec, TypeSpec, type, FloatSpec());
    MEMBER_DECLARATION(ParameterSpec, bool, colorDependence, false);
};

struct ParameterGroupSpec
{
    MEMBER_DECLARATION(ParameterGroupSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterGroupSpec, std::vector<ParameterSpec>, parameters, {});
    MEMBER_DECLARATION(ParameterGroupSpec, std::optional<size_t>, expertSettingAddress, std::nullopt);
};

struct ParametersSpec
{
    MEMBER_DECLARATION(ParametersSpec, std::vector<ParameterGroupSpec>, groups, {});
};
