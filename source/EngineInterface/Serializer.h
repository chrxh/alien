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
    DataDescription2 content;
};

class _Serializer
{
public:
    ENGINEINTERFACE_EXPORT bool serializeSimulationToFile(string const& filename, DeserializedSimulation const& data);
    ENGINEINTERFACE_EXPORT bool deserializeSimulationFromFile(string const& filename, DeserializedSimulation& data);

private:
    void serializeDataDescription(DataDescription2 const& data, std::ostream& stream) const;
    void serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings, std::ostream& stream) const;
    void serializeSymbolMap(SymbolMap const symbols, std::ostream& stream) const;

    void deserializeDataDescription(DataDescription2& data, std::istream& stream) const;
    void deserializeTimestepAndSettings(uint64_t& timestep, Settings& settings, std::istream& stream) const;
    void deserializeSymbolMap(SymbolMap& symbolMap, std::istream& stream);
};
