#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"
#include "AuxiliaryData.h"
#include "Descriptions.h"

struct DeserializedSimulation
{
    AuxiliaryData auxiliaryData;
    ClusteredDataDescription mainData;
};

struct SerializedSimulation
{
    std::string auxiliaryData;  //JSON
    std::string mainData;   //binary
};

class Serializer
{
public:
    static bool serializeSimulationToFiles(std::string const& filename, DeserializedSimulation const& data);
    static bool deserializeSimulationFromFiles(DeserializedSimulation& data, std::string const& filename);

    static bool serializeSimulationToStrings(SerializedSimulation& output, DeserializedSimulation const& input);
    static bool deserializeSimulationFromStrings(DeserializedSimulation& output, SerializedSimulation const& input);

    static bool serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content);
    static bool deserializeContentFromFile(ClusteredDataDescription& content, std::string const& filenam);

private:
    static void serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream);
    static void serializeAuxiliaryData(AuxiliaryData const& auxiliaryData, std::ostream& stream);

    static bool deserializeDataDescription(ClusteredDataDescription& data, std::string const& filename);
    static void deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream);
    static void deserializeAuxiliaryData(AuxiliaryData& auxiliaryData, std::istream& stream);
};
