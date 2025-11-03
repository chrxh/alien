#pragma once

#include <filesystem>

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/StatisticsHistory.h>

#include "Definitions.h"
#include "DeserializedSimulation.h"
#include "SerializedSimulation.h"
#include "SettingsForSerialization.h"

class SerializerService
{
    MAKE_SINGLETON(SerializerService);

public:
    bool serializeSimulationToFiles(std::filesystem::path const& filename, DeserializedSimulation const& data) const;
    bool deserializeSimulationFromFiles(DeserializedSimulation& data, std::filesystem::path const& filename) const;
    bool deleteSimulation(std::filesystem::path const& filename) const;

    bool serializeSimulationToStrings(SerializedSimulation& output, DeserializedSimulation const& input) const;
    bool deserializeSimulationFromStrings(DeserializedSimulation& output, SerializedSimulation const& input) const;

    bool serializeGenomeToFile(std::filesystem::path const& filename, GenomeDescription const& genome) const;
    bool deserializeGenomeFromFile(GenomeDescription& genome, std::filesystem::path const& filename) const;

    bool serializeGenomeToString(std::string& output, std::vector<uint8_t> const& input) const;
    bool deserializeGenomeFromString(std::vector<uint8_t>& output, std::string const& input) const;

    bool serializeSimulationParametersToFile(std::filesystem::path const& filename, SimulationParameters const& parameters) const;
    bool deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::filesystem::path const& filename) const;

    bool serializeStatisticsToFile(std::filesystem::path const& filename, StatisticsHistoryData const& statistics) const;

    bool serializeContentToFile(std::filesystem::path const& filename, Description const& content) const;
    bool deserializeContentFromFile(Description& content, std::filesystem::path const& filename) const;

private:
    void serializeDescription(Description const& description, std::ostream& stream) const;
    bool deserializeDescription(Description& description, std::filesystem::path const& filename) const;
    void deserializeDescription(Description& description, std::istream& stream) const;

    void serializeSettings(SettingsForSerialization const& settings, std::ostream& stream) const;
    void deserializeSettings(SettingsForSerialization& settings, std::istream& stream) const;

    void serializeSimulationParameters(SimulationParameters const& parameters, std::ostream& stream) const;
    void deserializeSimulationParameters(SimulationParameters& parameters, std::istream& stream) const;

    void serializeStatistics(StatisticsHistoryData const& statistics, std::ostream& stream) const;
    void deserializeStatistics(StatisticsHistoryData& statistics, std::istream& stream) const;

    bool wrapGenome(Description& output, GenomeDescription const& input) const;
    bool unwrapGenome(GenomeDescription& output, Description& input) const;
};
