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

class _Serializer
{
public:
    ENGINEINTERFACE_EXPORT bool serializeSimulationToFile(std::string const& filename, DeserializedSimulation const& data);
    ENGINEINTERFACE_EXPORT bool deserializeSimulationFromFile(std::string const& filename, DeserializedSimulation& data);

    ENGINEINTERFACE_EXPORT bool serializeContentToFile(std::string const& filename, ClusteredDataDescription const& content);
    ENGINEINTERFACE_EXPORT bool deserializeContentFromFile(std::string const& filename, ClusteredDataDescription& content);

    ENGINEINTERFACE_EXPORT bool serializeSymbolsToFile(std::string const& filename, SymbolMap const& symbolMap);
    ENGINEINTERFACE_EXPORT bool deserializeSymbolsFromFile(std::string const& filename, SymbolMap& symbolMap);

private:
    void serializeDataDescription(ClusteredDataDescription const& data, std::ostream& stream) const;
    void serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings, std::ostream& stream) const;
    void serializeSymbolMap(SymbolMap const symbols, std::ostream& stream) const;

    void deserializeDataDescription(ClusteredDataDescription& data, std::istream& stream) const;
    void deserializeTimestepAndSettings(uint64_t& timestep, Settings& settings, std::istream& stream) const;
    void deserializeSymbolMap(SymbolMap& symbolMap, std::istream& stream);
};
