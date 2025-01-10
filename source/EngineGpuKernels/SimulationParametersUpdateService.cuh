#include <optional>

#include "Base/Singleton.h"

#include "EngineInterface/RawStatisticsData.h"
#include "EngineInterface/SettingsForSimulation.h"
#include "EngineInterface/SimulationParametersUpdateConfig.h"

#include "Definitions.cuh"

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
        RawStatisticsData const& statistics);  //returns true if parameters have been changed
};
