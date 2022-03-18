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
    static bool serializeSimulationToFile(std::string const& filename, DeserializedSimulation const& data);
    static bool deserializeSimulationFromFile(std::string const& filename, DeserializedSimulation& data);

    static bool serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content);
    static bool deserializeContentFromFile(std::string const& filename, ClusteredDataDescription& content);

    static bool serializeSymbolsToFile(std::string const& filename, SymbolMap const& symbolMap);
    static bool deserializeSymbolsFromFile(std::string const& filename, SymbolMap& symbolMap);

private:
    static void serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream);
    static void serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings, std::ostream& stream);
    static void serializeSymbolMap(SymbolMap const symbols, std::ostream& stream);

    static void deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream);
    static void deserializeTimestepAndSettings(uint64_t& timestep, Settings& settings, std::istream& stream);
    static void deserializeSymbolMap(SymbolMap& symbolMap, std::istream& stream);
};
