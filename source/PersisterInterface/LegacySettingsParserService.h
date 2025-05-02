#pragma once

#include <optional>

#include <boost/property_tree/ptree_fwd.hpp>

#include "Base/Singleton.h"
#include "EngineInterface/SimulationParameters.h"

template <typename T>
struct LegacyProperty
{
    bool existent = false;
    T parameter;
};

template <typename T>
struct LegacySpotProperty
{
    bool existent = false;
    bool active = false;
    T parameter;
};

struct LegacyFeatures
{
    LegacyProperty<bool> advancedMuscleControl;
};

struct LegacyParametersForBase
{
};

struct LegacyParametersForSpot
{
};

struct LegacyParameters
{
    LegacyParametersForBase base;
    LegacyParametersForSpot spots[MAX_LAYERS];
};

class LegacySettingsParserService
{
    MAKE_SINGLETON(LegacySettingsParserService);

public:
    void searchAndApplyLegacyParameters(
        std::string const& programVersion,
        boost::property_tree::ptree& tree,
        SimulationParameters& parameters);

private:
    void updateParametersAndFeaturesForLegacyFiles(
        std::string const& programVersion,
        LegacyFeatures const& legacyFeatures,
        LegacyParameters const& legacyParameters,
        SimulationParameters& parameters);
};
