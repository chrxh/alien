#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "Base/Definitions.h"

#include "Definitions.h"

struct AbstractParameterSpec
{
    MEMBER_DECLARATION(AbstractParameterSpec, bool, visibleInBase, true);
    MEMBER_DECLARATION(AbstractParameterSpec, bool, visibleInSpot, false);
    MEMBER_DECLARATION(AbstractParameterSpec, bool, visibleInSource, false);
};

struct FloatParameterSpec : public AbstractParameterSpec
{
    MEMBER_DECLARATION(FloatParameterSpec, std::string, name, std::string());
    MEMBER_DECLARATION(FloatParameterSpec, int, valueAddress, 0);
    MEMBER_DECLARATION(FloatParameterSpec, float, min, 0);
    MEMBER_DECLARATION(FloatParameterSpec, float, max, 0);
    MEMBER_DECLARATION(FloatParameterSpec, bool, infinity, false);
    MEMBER_DECLARATION(FloatParameterSpec, std::optional<int>, valueActivationAddress, std::nullopt);
};
using ParameterSpec = std::variant<FloatParameterSpec>;

struct ParameterAlternativeSpec : public AbstractParameterSpec
{
    MEMBER_DECLARATION(ParameterAlternativeSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterAlternativeSpec, int, valueAddress, 0);
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
