#pragma once

#include <memory>
#include <optional>

#include <Base/Singleton.h>

#include <EngineInterface/SettingsForSimulation.h>
#include <EngineInterface/SimulationParametersUpdateConfig.h>
#include <EngineInterface/StatisticsRawData.h>

struct SimulationData;
class _MaxAgeBalancer;
using MaxAgeBalancer = std::shared_ptr<_MaxAgeBalancer>;

class SimulationParametersUpdateService
{
    MAKE_SINGLETON(SimulationParametersUpdateService);

public:
    SimulationParameters integrateChanges(
        SimulationParameters const& currentParameters,
        SimulationParameters const& changedParameters,
        SimulationParametersUpdateConfig const& updateConfig) const;

    bool updateSimulationParametersAfterTimestep(
        SettingsForSimulation& settings,
        MaxAgeBalancer const& maxAgeBalancer,
        SimulationData const& simulationData,
        uint64_t timestep,
        StatisticsRawData const& statistics);  //returns true if parameters have been changed
};
