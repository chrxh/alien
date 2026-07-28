#pragma once

#include <filesystem>
#include <string>

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/StatisticsHistory.h>

#include "Definitions.h"
#include "DeserializedSimulation.h"
#include "SettingsForSerialization.h"

class SerializerService
{
    MAKE_SINGLETON(SerializerService);

public:
    bool serializeSimulationToFiles(std::filesystem::path const& filename, DeserializedSimulation const& data) const;
    bool deserializeSimulationFromFiles(DeserializedSimulation& data, std::filesystem::path const& filename) const;
    bool deleteSimulation(std::filesystem::path const& filename) const;

    bool serializeSimulationToString(std::string& output, DeserializedSimulation const& input) const;
    bool deserializeSimulationFromString(DeserializedSimulation& output, std::string const& input) const;

    bool serializeGenomeToFile(std::filesystem::path const& filename, GenomeDesc const& genome) const;
    bool deserializeGenomeFromFile(GenomeDesc& genome, std::filesystem::path const& filename) const;

    bool serializeGenomeToString(std::string& output, std::vector<uint8_t> const& input) const;
    bool deserializeGenomeFromString(std::vector<uint8_t>& output, std::string const& input) const;

    bool serializeSimulationParametersToFile(std::filesystem::path const& filename, SimulationParameters const& parameters) const;
    bool deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::filesystem::path const& filename) const;

    bool serializeContentToFile(std::filesystem::path const& filename, Desc const& content) const;
    bool deserializeContentFromFile(Desc& content, std::filesystem::path const& filename) const;

private:
    void serializeDescription(Desc const& description, std::ostream& stream) const;
    bool deserializeDescription(Desc& description, std::filesystem::path const& filename) const;
    void deserializeDescription(Desc& description, std::istream& stream) const;

    void serializeSimulation(DeserializedSimulation const& data, std::ostream& stream) const;
    void deserializeSimulation(DeserializedSimulation& data, std::istream& stream) const;

    void serializeSettings(SimulationParameters const& parameters, std::ostream& stream) const;
    void deserializeSettings(SimulationParameters& parameters, std::istream& stream) const;

    bool wrapGenome(Desc& output, GenomeDesc const& input) const;
    bool unwrapGenome(GenomeDesc& output, Desc& input) const;
};
