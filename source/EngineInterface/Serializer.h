#pragma once

#include "Base/Definitions.h"

#include "Definitions.h"
#include "SymbolMap.h"
#include "SimulationParameters.h"
#include "GeneralSettings.h"
#include "Descriptions.h"

struct SerializedSimulation
{
    std::string generalSettings;
    std::string simulationParameters;
    std::string symbolMap;
    std::string content;
};

struct DeserializedSimulation
{
    uint32_t timestep;
    GeneralSettings generalSettings;
    SimulationParameters simulationParameters;
    SymbolMap symbolMap;
    DataDescription content;
};

class _Serializer
{
public:
    ENGINEINTERFACE_EXPORT bool loadSimulationDataFromFile(string const& filename, SerializedSimulation& data);
    ENGINEINTERFACE_EXPORT bool loadDataFromFile(std::string const& filename, std::string& data);

    ENGINEINTERFACE_EXPORT SerializedSimulation serializeSimulation(DeserializedSimulation const& data);
    ENGINEINTERFACE_EXPORT DeserializedSimulation deserializeSimulation(SerializedSimulation const& data);

	ENGINEINTERFACE_EXPORT string serializeDataDescription(DataDescription const& desc) const;
    ENGINEINTERFACE_EXPORT DataDescription deserializeDataDescription(string const& data);

	ENGINEINTERFACE_EXPORT string serializeSymbolMap(SymbolMap const symbols) const;
    ENGINEINTERFACE_EXPORT SymbolMap deserializeSymbolMap(string const& data);

	ENGINEINTERFACE_EXPORT string serializeSimulationParameters(SimulationParameters const& parameters) const;
    ENGINEINTERFACE_EXPORT SimulationParameters deserializeSimulationParameters(string const& data);

    ENGINEINTERFACE_EXPORT string serializeGeneralSettings(GeneralSettings const& generalSettings) const;
    ENGINEINTERFACE_EXPORT GeneralSettings deserializeGeneralSettings(std::string const& data) const;
};
