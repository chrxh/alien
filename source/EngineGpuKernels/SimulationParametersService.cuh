#include <optional>

#include "Base/Singleton.h"

#include "EngineInterface/RawStatisticsData.h"
#include "EngineInterface/Settings.h"
#include "EngineInterface/SimulationParametersUpdateConfig.h"

#include "Definitions.cuh"

class SimulationParametersService
{
    MAKE_SINGLETON(SimulationParametersService);

public:
    SimulationParameters integrateChanges(
        SimulationParameters const& currentParameters,
        SimulationParameters const& changedParameters,
        SimulationParametersUpdateConfig const& updateConfig) const;

    bool updateSimulationParametersAfterTimestep(
        Settings& settings,
        MaxAgeBalancer const& maxAgeBalancer,
        SimulationData const& simulationData,
        RawStatisticsData const& statistics);  //returns true if parameters have been changed
};
