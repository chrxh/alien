#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "Base/Definitions.h"

#include "Definitions.h"

struct BoolParameterSpec
{
    MEMBER_DECLARATION(BoolParameterSpec, bool, visibleInBase, true);
    MEMBER_DECLARATION(BoolParameterSpec, bool, visibleInZone, false);
    MEMBER_DECLARATION(BoolParameterSpec, bool, visibleInSource, false);
    MEMBER_DECLARATION(BoolParameterSpec, std::string, name, std::string());
    MEMBER_DECLARATION(BoolParameterSpec, int, refTypeId, 0);
    MEMBER_DECLARATION(BoolParameterSpec, int, valueAddress, 0);
    MEMBER_DECLARATION(BoolParameterSpec, std::optional<std::string>, tooltip, std::nullopt);
    MEMBER_DECLARATION(BoolParameterSpec, std::optional<int>, valueActivationAddress, std::nullopt);
};

struct FloatParameterSpec
{
    MEMBER_DECLARATION(FloatParameterSpec, bool, visibleInBase, true);
    MEMBER_DECLARATION(FloatParameterSpec, bool, visibleInZone, false);
    MEMBER_DECLARATION(FloatParameterSpec, bool, visibleInSource, false);
    MEMBER_DECLARATION(FloatParameterSpec, std::string, name, std::string());
    MEMBER_DECLARATION(FloatParameterSpec, int, valueAddress, 0);
    MEMBER_DECLARATION(FloatParameterSpec, std::optional<std::string>, tooltip, std::nullopt);
    MEMBER_DECLARATION(FloatParameterSpec, std::optional<int>, valueActivationAddress, std::nullopt);

    MEMBER_DECLARATION(FloatParameterSpec, float, min, 0);
    MEMBER_DECLARATION(FloatParameterSpec, float, max, 0);
    MEMBER_DECLARATION(FloatParameterSpec, bool, logarithmic, false);
    MEMBER_DECLARATION(FloatParameterSpec, std::string, format, "%.3f");
    MEMBER_DECLARATION(FloatParameterSpec, bool, infinity, false);
};

struct Char64ParameterSpec
{
    MEMBER_DECLARATION(Char64ParameterSpec, bool, visibleInBase, true);
    MEMBER_DECLARATION(Char64ParameterSpec, bool, visibleInZone, false);
    MEMBER_DECLARATION(Char64ParameterSpec, bool, visibleInSource, false);
    MEMBER_DECLARATION(Char64ParameterSpec, std::string, name, std::string());
    MEMBER_DECLARATION(Char64ParameterSpec, int, refTypeId, 0);
    MEMBER_DECLARATION(Char64ParameterSpec, int, valueAddress, 0);
    MEMBER_DECLARATION(Char64ParameterSpec, std::optional<std::string>, tooltip, std::nullopt);
    MEMBER_DECLARATION(Char64ParameterSpec, std::optional<int>, valueActivationAddress, std::nullopt);
};

using ParameterSpec = std::variant<BoolParameterSpec, FloatParameterSpec, Char64ParameterSpec>;

struct ParameterAlternativeSpec
{
    MEMBER_DECLARATION(ParameterAlternativeSpec, bool, visibleInBase, true);
    MEMBER_DECLARATION(ParameterAlternativeSpec, bool, visibleInZone, false);
    MEMBER_DECLARATION(ParameterAlternativeSpec, bool, visibleInSource, false);
    MEMBER_DECLARATION(ParameterAlternativeSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterAlternativeSpec, int, valueAddress, 0);
    MEMBER_DECLARATION(ParameterAlternativeSpec, std::optional<std::string>, tooltip, std::nullopt);

    MEMBER_DECLARATION(ParameterAlternativeSpec, std::vector<std::vector<ParameterSpec>>, parameter, {});
};

using ParameterOrAlternativeSpec = std::variant<ParameterSpec, ParameterAlternativeSpec>;

struct ParameterGroupSpec
{
    MEMBER_DECLARATION(ParameterGroupSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterGroupSpec, std::vector<ParameterOrAlternativeSpec>, parameters, {});
    MEMBER_DECLARATION(ParameterGroupSpec, std::optional<int>, featureAddress, std::nullopt);
};

struct FeatureSpec
{
    MEMBER_DECLARATION(FeatureSpec, std::string, name, std::string());
    MEMBER_DECLARATION(FeatureSpec, int, featureAddress, 0);
};

struct ParametersSpec
{
    MEMBER_DECLARATION(ParametersSpec, std::vector<ParameterGroupSpec>, groups, {});
    MEMBER_DECLARATION(ParametersSpec, std::vector<FeatureSpec>, features, {});
};
