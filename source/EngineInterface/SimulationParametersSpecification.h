#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

#include "Base/Definitions.h"

#include "Definitions.h"

enum class ParameterType
{
    Bool,
    Int,
    Float,
    ColorVectorInt,
    ColorVectorFloat,
    ColorMatrixInt,
    ColorMatrixFloat
};

enum class Visibility
{
    OnlyBase,
    BaseZoneAndZone,
    RadiationSource
};


struct ParameterSpec
{
    MEMBER_DECLARATION(ParameterSpec, Visibility, visibility, Visibility::OnlyBase);
    MEMBER_DECLARATION(ParameterSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterSpec, ParameterType, type, ParameterType::Float);
    MEMBER_DECLARATION(ParameterSpec, int, valueAddress, 0);
    MEMBER_DECLARATION(ParameterSpec, std::optional<int>, valueActivationAddress, std::nullopt);
};

struct ParameterAlternativeSpec
{
    MEMBER_DECLARATION(ParameterAlternativeSpec, Visibility, visibility, Visibility::OnlyBase);
    MEMBER_DECLARATION(ParameterAlternativeSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterAlternativeSpec, int, valueAddress, 0);
    MEMBER_DECLARATION(ParameterAlternativeSpec, std::vector<std::vector<ParameterSpec>>, parameter, {});
};

using ParameterOrAlternativeSpec = std::variant<ParameterSpec, ParameterAlternativeSpec>;

struct ParameterGroupSpec
{
    MEMBER_DECLARATION(ParameterGroupSpec, std::string, name, std::string());
    MEMBER_DECLARATION(ParameterGroupSpec, std::vector<ParameterOrAlternativeSpec>, parameter, {});
};

struct ParametersSpec
{
    MEMBER_DECLARATION(ParametersSpec, std::vector<ParameterGroupSpec>, groups, {});
};
