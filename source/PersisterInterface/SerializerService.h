#pragma once

#include <filesystem>
#include <string>

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Descs.h>
#include <EngineInterface/StatisticsHistory.h>

#include "Definitions.h"

class SerializerService
{
    MAKE_SINGLETON(SerializerService);

public:
    bool serializeSimulationToFiles(std::filesystem::path const& filename, SimulationDesc const& data) const;
    bool deserializeSimulationFromFiles(SimulationDesc& data, std::filesystem::path const& filename) const;
    bool deleteSimulation(std::filesystem::path const& filename) const;

    bool serializeSimulationToString(std::string& output, SimulationDesc const& input) const;
    bool deserializeSimulationFromString(SimulationDesc& output, std::string const& input) const;

    bool serializeGenomeToFile(std::filesystem::path const& filename, GenomeDesc const& genome) const;
    bool deserializeGenomeFromFile(GenomeDesc& genome, std::filesystem::path const& filename) const;

    bool serializeGenomeToString(std::string& output, std::vector<uint8_t> const& input) const;
    bool deserializeGenomeFromString(std::vector<uint8_t>& output, std::string const& input) const;

    bool serializeSimulationParametersToFile(std::filesystem::path const& filename, SimulationParameters const& parameters) const;
    bool deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::filesystem::path const& filename) const;

    bool serializeContentToFile(std::filesystem::path const& filename, ContentDesc const& content) const;
    bool deserializeContentFromFile(ContentDesc& content, std::filesystem::path const& filename) const;

private:
    void serializeDescription(ContentDesc const& description, std::ostream& stream) const;
    bool deserializeDescription(ContentDesc& description, std::filesystem::path const& filename) const;
    void deserializeDescription(ContentDesc& description, std::istream& stream) const;

    void serializeSimulation(SimulationDesc const& data, std::ostream& stream) const;
    void deserializeSimulation(SimulationDesc& data, std::istream& stream) const;

    void serializeSettings(SimulationParameters const& parameters, std::ostream& stream) const;
    void deserializeSettings(SimulationParameters& parameters, std::istream& stream) const;

    bool wrapGenome(ContentDesc& output, GenomeDesc const& input) const;
    bool unwrapGenome(GenomeDesc& output, ContentDesc& input) const;
};
