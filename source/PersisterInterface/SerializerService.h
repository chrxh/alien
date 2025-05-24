#pragma once

#include <filesystem>

#include "Base/Definitions.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/StatisticsHistory.h"

#include "DeserializedSimulation.h"
#include "SerializedSimulation.h"
#include "Definitions.h"
#include "SettingsForSerialization.h"
#include "Base/Singleton.h"

class SerializerService
{
    MAKE_SINGLETON(SerializerService);

public:
    bool serializeSimulationToFiles(std::filesystem::path const& filename, DeserializedSimulation const& data);
    bool deserializeSimulationFromFiles(DeserializedSimulation& data, std::filesystem::path const& filename);
    bool deleteSimulation(std::filesystem::path const& filename);

    bool serializeSimulationToStrings(SerializedSimulation& output, DeserializedSimulation const& input);
    bool deserializeSimulationFromStrings(DeserializedSimulation& output, SerializedSimulation const& input);

    bool serializeGenomeToFile(std::filesystem::path const& filename, std::vector<uint8_t> const& genome);
    bool deserializeGenomeFromFile(std::vector<uint8_t>& genome, std::filesystem::path const& filename);

    bool serializeGenomeToString(std::string& output, std::vector<uint8_t> const& input);
    bool deserializeGenomeFromString(std::vector<uint8_t>& output, std::string const& input);

    bool serializeSimulationParametersToFile(std::filesystem::path const& filename, SimulationParameters const& parameters);
    bool deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::filesystem::path const& filename);

    bool serializeStatisticsToFile(std::filesystem::path const& filename, StatisticsHistoryData const& statistics);

    bool serializeContentToFile(std::filesystem::path const& filename, CollectionDescription const& content);
    bool deserializeContentFromFile(CollectionDescription& content, std::filesystem::path const& filename);

private:
    void serializeDescription(CollectionDescription const& data, std::ostream& stream);
    bool deserializeDescription(CollectionDescription& data, std::filesystem::path const& filename);
    void deserializeDescription(CollectionDescription& data, std::istream& stream);

    void serializeSettings(SettingsForSerialization const& settings, std::ostream& stream);
    void deserializeSettings(SettingsForSerialization& settings, std::istream& stream);

    void serializeSimulationParameters(SimulationParameters const& parameters, std::ostream& stream);
    void deserializeSimulationParameters(SimulationParameters& parameters, std::istream& stream);

    void serializeStatistics(StatisticsHistoryData const& statistics, std::ostream& stream);
    void deserializeStatistics(StatisticsHistoryData& statistics, std::istream& stream);

    bool wrapGenome(CollectionDescription& output, std::vector<uint8_t> const& input);
    bool unwrapGenome(std::vector<uint8_t>& output, CollectionDescription const& input);
};
