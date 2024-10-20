#pragma once

#include "Base/Definitions.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/StatisticsHistory.h"

#include "DeserializedSimulation.h"
#include "SerializedSimulation.h"
#include "Definitions.h"
#include "AuxiliaryData.h"
#include "Base/Singleton.h"

class SerializerService
{
    MAKE_SINGLETON(SerializerService);

public:
    bool serializeSimulationToFiles(std::string const& filename, DeserializedSimulation const& data);
    bool deserializeSimulationFromFiles(DeserializedSimulation& data, std::string const& filename);

    bool serializeSimulationToStrings(SerializedSimulation& output, DeserializedSimulation const& input);
    bool deserializeSimulationFromStrings(DeserializedSimulation& output, SerializedSimulation const& input);

    bool serializeGenomeToFile(std::string const& filename, std::vector<uint8_t> const& genome);
    bool deserializeGenomeFromFile(std::vector<uint8_t>& genome, std::string const& filename);

    bool serializeGenomeToString(std::string& output, std::vector<uint8_t> const& input);
    bool deserializeGenomeFromString(std::vector<uint8_t>& output, std::string const& input);

    bool serializeSimulationParametersToFile(std::string const& filename, SimulationParameters const& parameters);
    bool deserializeSimulationParametersFromFile(SimulationParameters& parameters, std::string const& filename);

    bool serializeStatisticsToFile(std::string const& filename, StatisticsHistoryData const& statistics);

    bool serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content);
    bool deserializeContentFromFile(ClusteredDataDescription& content, std::string const& filename);

private:
    void serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream);
    bool deserializeDataDescription(ClusteredDataDescription& data, std::string const& filename);
    void deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream);

    void serializeAuxiliaryData(AuxiliaryData const& auxiliaryData, std::ostream& stream);
    void deserializeAuxiliaryData(AuxiliaryData& auxiliaryData, std::istream& stream);

    void serializeSimulationParameters(SimulationParameters const& parameters, std::ostream& stream);
    void deserializeSimulationParameters(SimulationParameters& parameters, std::istream& stream);

    void serializeStatistics(StatisticsHistoryData const& statistics, std::ostream& stream);
    void deserializeStatistics(StatisticsHistoryData& statistics, std::istream& stream);

    bool wrapGenome(ClusteredDataDescription& output, std::vector<uint8_t> const& input);
    bool unwrapGenome(std::vector<uint8_t>& output, ClusteredDataDescription const& input);
};
