#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"
#include "SymbolMap.h"
#include "Settings.h"
#include "SimulationParameters.h"
#include "GeneralSettings.h"
#include "Descriptions.h"

struct DeserializedSimulation
{
    uint64_t timestep;
    Settings settings;
    SymbolMap symbolMap;
    ClusteredDataDescription content;
};

class Serializer
{
public:
    static bool serializeSimulationToFiles(std::string const& filename, DeserializedSimulation const& data);
    static bool deserializeSimulationFromFiles(DeserializedSimulation& data, std::string const& filename);

    static bool serializeSimulationToStrings(
        std::string& content,
        std::string& timestepAndSettings,
        std::string& symbolMap,
        DeserializedSimulation const& data);
    static bool deserializeSimulationFromStrings(
        DeserializedSimulation& data,
        std::string const& content,
        std::string const& timestepAndSettings,
        std::string const& symbolMap);

/*
    static bool serializeSimulationToSingleString(std::string& output, DeserializedSimulation const& data);
    static bool deserializeSimulationFromSingleString(DeserializedSimulation& data, std::string const& input);
*/

    static bool serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content);
    static bool deserializeContentFromFile(ClusteredDataDescription& content, std::string const& filenam);

    static bool serializeSymbolsToFile(std::string const& filename, SymbolMap const& symbolMap);
    static bool deserializeSymbolsFromFile(SymbolMap& symbolMap, std::string const& filename);

private:
    static void serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream);
    static void serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings, std::ostream& stream);
    static void serializeSymbolMap(SymbolMap const symbols, std::ostream& stream);

    static bool deserializeDataDescription(ClusteredDataDescription& data, std::string const& filename);
    static void deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream);
    static void DEPREACATED_deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream);
    static void deserializeTimestepAndSettings(uint64_t& timestep, Settings& settings, std::istream& stream);
    static void deserializeSymbolMap(SymbolMap& symbolMap, std::istream& stream);
};
