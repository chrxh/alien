#pragma once

#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <Base/Singleton.h>

#include "SimulationParameters.h"
#include "SimulationParametersSpecification.h"
#include "SpecificationEvaluationService.h"

enum class ParameterType
{
    Bool,
    Int,
    Float,
    Float2,
    Text,
    Alternative,
    Color,
    ColorTransitionRule
};

using ParameterScalar = std::variant<bool, int, float, RealVector2D, std::string, FloatColorRGB, ColorTransitionRule>;

struct ParameterValue
{
    ColorDependence colorDependence = ColorDependence::None;
    std::vector<ParameterScalar> values;
    std::optional<bool> enabled;
    std::optional<bool> pinned;

    bool operator==(ParameterValue const&) const = default;
};

struct ParameterEntry
{
    std::string path;
    ParameterSpec const* spec = nullptr;
};

class ParametersAccessService
{
    MAKE_SINGLETON(ParametersAccessService);

public:
    std::vector<ParameterGroupSpec const*> getGroups(LocationType locationType) const;
    ParameterGroupSpec const* findGroup(std::string const& name) const;

    std::vector<ParameterEntry> getParameters(ParameterGroupSpec const& groupSpec, SimulationParameters const& parameters, int orderNumber) const;

    std::optional<ParameterEntry> findParameter(std::string const& path, LocationType locationType) const;

    ParameterType getType(ParameterEntry const& entry) const;

    ParameterValue getValue(ParameterEntry const& entry, SimulationParameters const& parameters, int orderNumber) const;

    void setValue(ParameterEntry const& entry, SimulationParameters& parameters, int orderNumber, ParameterValue const& value) const;

    bool copyGroup(ParameterGroupSpec const& groupSpec, SimulationParameters const& source, SimulationParameters& target) const;
};
