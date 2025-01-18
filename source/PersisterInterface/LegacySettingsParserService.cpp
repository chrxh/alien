#include "LegacySettingsParserService.h"

#include <set>

#include "Base/StringHelper.h"
#include "Base/VersionParserService.h"

#include "ParameterParser.h"

namespace
{
    template <typename T>
    bool contains(SimulationParameters const& parameters, ColorVector<T> SimulationParametersZoneValues::*parameter, std::set<T> const& values)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (!values.contains((parameters.baseValues.*parameter)[i])) {
                return false;
            }
            for (int j = 0; j < parameters.numZones; ++j) {
                if (!values.contains((parameters.zone[j].values.*parameter)[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    template <typename T>
    bool equals(ColorVector<T> const& parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            if (parameter[i] != value) {
                return false;
            }
        }
        return true;
    }

    template <typename T>
    bool equals(ColorMatrix<T> const& parameter, T const& value)
    {
        for (int i = 0; i < MAX_COLORS; ++i) {
            for (int j = 0; j < MAX_COLORS; ++i) {
                if (parameter[i][j] != value) {
                    return false;
                }
            }
        }
        return true;
    }

    template<typename T>
    void readLegacyParameterForBase(LegacyProperty<T>& result, boost::property_tree::ptree& tree, std::string const& node)
    {
        T defaultDummy;
        result.existent = !ParameterParser::encodeDecode(tree, result.parameter, defaultDummy, node, ParserTask::Decode);
    }

    template <typename T>
    void readLegacyParameterForSpot(LegacySpotProperty<T>& result, boost::property_tree::ptree& tree, std::string const& node)
    {
        T defaultDummy;
        result.existent = !ParameterParser::encodeDecodeWithEnabled(tree, result.parameter, result.active, defaultDummy, node, ParserTask::Decode);
    }

    LegacyParametersForBase readLegacyParametersForBase(boost::property_tree::ptree& tree, std::string const& nodeBase)
    {
        LegacyParametersForBase result;
        return result;
    }

    LegacyParametersForSpot readLegacyParametersForSpot(boost::property_tree::ptree& tree, std::string const& nodeBase)
    {
        LegacyParametersForSpot result;
        return result;
    }
}

void LegacySettingsParserService::searchAndApplyLegacyParameters(
    std::string const& programVersion,
    boost::property_tree::ptree& tree,
    SimulationParameters& parameters)
{
    LegacyFeatures legacyFeatures;
    readLegacyParameterForBase(legacyFeatures.advancedMuscleControl, tree, "simulation parameters.features.additional muscle control");

    LegacyParameters legacyParameters;
    legacyParameters.base = readLegacyParametersForBase(tree, "simulation parameters.");
    for (int i = 0; i < parameters.numZones; ++i) {
        legacyParameters.spots[i] = readLegacyParametersForSpot(tree, "simulation parameters.spots." + std::to_string(i) + ".");
    }
    updateParametersAndFeaturesForLegacyFiles(programVersion, legacyFeatures, legacyParameters, parameters);
}

void LegacySettingsParserService::updateParametersAndFeaturesForLegacyFiles(
    std::string const& programVersion,
    LegacyFeatures const& legacyFeatures,
    LegacyParameters const& legacyParameters,
    SimulationParameters& parameters)
{
}
