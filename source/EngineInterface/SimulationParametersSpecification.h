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
    MEMBER_DECLARATION(ParameterSpec, int, valueAddress, 0);
    MEMBER_DECLARATION(ParameterSpec, std::optional<std::string>, tooltip, std::nullopt);
    MEMBER_DECLARATION(ParameterSpec, std::optional<int>, valueActivatedAddress, std::nullopt);
    MEMBER_DECLARATION(ParameterSpec, TypeSpec, type, FloatSpec());
};

struct ParameterGroupSpec
{
    MEMBER_DECLARATION(ParameterGroupSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterGroupSpec, std::vector<ParameterSpec>, parameters, {});
    MEMBER_DECLARATION(ParameterGroupSpec, bool, expertSetting, false);  // name field must match with ExpertSettingSpec::name
};

struct ExpertSettingSpec
{
    MEMBER_DECLARATION(ExpertSettingSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ExpertSettingSpec, int, enabledAddress, 0);
};

struct ParametersSpec
{
    MEMBER_DECLARATION(ParametersSpec, std::vector<ParameterGroupSpec>, groups, {});
    MEMBER_DECLARATION(ParametersSpec, std::vector<ExpertSettingSpec>, expertSettings, {});
};
