#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"
#include "Settings.h"
#include "SimulationParameters.h"
#include "GeneralSettings.h"
#include "Descriptions.h"

struct DeserializedSimulation
{
    uint64_t timestep;
    Settings settings;
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
        DeserializedSimulation const& data);
    static bool deserializeSimulationFromStrings(
        DeserializedSimulation& data,
        std::string const& content,
        std::string const& timestepAndSettings);

    static bool serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content);
    static bool deserializeContentFromFile(ClusteredDataDescription& content, std::string const& filenam);

private:
    static void serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream);
    static void serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings, std::ostream& stream);

    static bool deserializeDataDescription(ClusteredDataDescription& data, std::string const& filename);
    static void deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream);
    static void deserializeTimestepAndSettings(uint64_t& timestep, Settings& settings, std::istream& stream);
};
