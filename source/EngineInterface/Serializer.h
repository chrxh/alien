#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"
#include "SymbolMap.h"
#include "Settings.h"
#include "SimulationParameters.h"
#include "GeneralSettings.h"
#include "Descriptions.h"

struct SerializedSimulation
{
    std::string timestepAndSettings;
    std::string symbolMap;
    std::string content;
};

struct DeserializedSimulation
{
    uint64_t timestep;
    Settings settings;
    SymbolMap symbolMap;
    DataDescription content;
};

class _Serializer
{
public:
    ENGINEINTERFACE_EXPORT bool loadSimulationDataFromFile2(string const& filename, SerializedSimulation& data);
    ENGINEINTERFACE_EXPORT bool saveSimulationDataToFile2(string const& filename, SerializedSimulation& data);

    ENGINEINTERFACE_EXPORT SerializedSimulation serializeSimulation2(DeserializedSimulation const& data);
    ENGINEINTERFACE_EXPORT DeserializedSimulation deserializeSimulation2(SerializedSimulation const& data);

private:
	ENGINEINTERFACE_EXPORT string serializeSymbolMap(SymbolMap const symbols) const;
    ENGINEINTERFACE_EXPORT SymbolMap deserializeSymbolMap(string const& data);

    ENGINEINTERFACE_EXPORT string serializeTimestepAndSettings(uint64_t timestep, Settings const& generalSettings) const;
    ENGINEINTERFACE_EXPORT std::pair<uint64_t, Settings> deserializeTimestepAndSettings(std::string const& data) const;

    ENGINEINTERFACE_EXPORT string serializeDataDescription(DataDescription const& desc) const;
    ENGINEINTERFACE_EXPORT DataDescription deserializeDataDescription(string const& data);

    ENGINEINTERFACE_EXPORT bool loadDataFromFile(std::string const& filename, std::string& data);
    ENGINEINTERFACE_EXPORT bool saveDataToFile(std::string const& filename, std::string const& data);
};
