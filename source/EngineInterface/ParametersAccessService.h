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

// Text holds the name of the selected alternative for ParameterType::Alternative
using ParameterScalar = std::variant<bool, int, float, RealVector2D, std::string, FloatColorRGB, ColorTransitionRule>;

struct ParameterValue
{
    ColorDependence colorDependence = ColorDependence::None;
    std::vector<ParameterScalar> values;  // 1, MAX_COLORS or MAX_COLORS * MAX_COLORS (row-major) entries
    std::optional<bool> enabled;
    std::optional<bool> pinned;

    bool operator==(ParameterValue const&) const = default;
};

struct ParameterEntry
{
    std::string path;
    ParameterSpec const* spec = nullptr;
};

// Addresses simulation parameters by paths of the form "<group>.<parameter>[.<alternative>.<parameter>]" (case-insensitive) as used in the settings files.
// A location is given by its order number (0 = base).
class ParametersAccessService
{
    MAKE_SINGLETON(ParametersAccessService);

public:
    std::vector<ParameterGroupSpec const*> getGroups(LocationType locationType) const;
    ParameterGroupSpec const* findGroup(std::string const& name) const;

    // Contains the sub-parameters of the selected alternatives only
    std::vector<ParameterEntry> getParameters(ParameterGroupSpec const& groupSpec, SimulationParameters const& parameters, int orderNumber) const;

    std::optional<ParameterEntry> findParameter(std::string const& path, LocationType locationType) const;

    ParameterType getType(ParameterEntry const& entry) const;

    // Returns the effective value, i.e. the base value for a layer that does not override it
    ParameterValue getValue(ParameterEntry const& entry, SimulationParameters const& parameters, int orderNumber) const;

    // Throws std::invalid_argument if the value does not match the parameter
    void setValue(ParameterEntry const& entry, SimulationParameters& parameters, int orderNumber, ParameterValue const& value) const;

    // Returns false without copying if the locations of both parameters differ
    bool copyGroup(ParameterGroupSpec const& groupSpec, SimulationParameters const& source, SimulationParameters& target) const;
};
