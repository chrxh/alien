#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"
#include "AuxiliaryData.h"
#include "Descriptions.h"
#include "StatisticsHistory.h"

struct DeserializedSimulation
{
    ClusteredDataDescription mainData;
    AuxiliaryData auxiliaryData;
    StatisticsHistoryData statistics;
};

struct SerializedSimulation
{
    std::string mainData;  //binary
    std::string auxiliaryData;  //JSON
    std::string statistics;  //CSV
};

class SerializerService
{
public:
    static bool serializeSimulationToFiles(std::string const& filename, DeserializedSimulation const& data);
    static bool deserializeSimulationFromFiles(DeserializedSimulation& data, std::string const& filename);

    static bool serializeSimulationToStrings(SerializedSimulation& output, DeserializedSimulation const& input);
    static bool deserializeSimulationFromStrings(DeserializedSimulation& output, SerializedSimulation const& input);

    static bool serializeGenomeToFile(std::string const& filename, std::vector<uint8_t> const& genome);
    static bool deserializeGenomeFromFile(std::vector<uint8_t>& genome, std::string const& filename);

    static bool serializeGenomeToString(std::string& output, std::vector<uint8_t> const& input);
    static bool deserializeGenomeFromString(std::vector<uint8_t>& output, std::string const& input);

    static bool serializeSimulationParametersToFile(std::string const& filename, SimulationParameters const& parameters);
    static bool deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::string const& filename);

    static bool serializeStatisticsToFile(std::string const& filename, StatisticsHistoryData const& statistics);

    static bool serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content);
    static bool deserializeContentFromFile(ClusteredDataDescription& content, std::string const& filename);

private:
    static void serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream);
    static bool deserializeDataDescription(ClusteredDataDescription& data, std::string const& filename);
    static void deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream);

    static void serializeAuxiliaryData(AuxiliaryData const& auxiliaryData, std::ostream& stream);
    static void deserializeAuxiliaryData(AuxiliaryData& auxiliaryData, std::istream& stream);

    static void serializeSimulationParameters(SimulationParameters const& parameters, std::ostream& stream);
    static void deserializeSimulationParameters(SimulationParameters& parameters, std::istream& stream);

    static void serializeStatistics(StatisticsHistoryData const& statistics, std::ostream& stream);
    static void deserializeStatistics(StatisticsHistoryData& statistics, std::istream& stream);

    static bool wrapGenome(ClusteredDataDescription& output, std::vector<uint8_t> const& input);
    static bool unwrapGenome(std::vector<uint8_t>& output, ClusteredDataDescription const& input);
};
