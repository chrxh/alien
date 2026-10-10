#include <optional>

#include <Base/Interface/Singleton.h>

#include <Engine/Interface/SettingsForSimulation.h>
#include <Engine/Interface/SimulationParametersUpdateConfig.h>

#include <Engine/Kernels/Definitions.cuh>

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
        SimulationData const& simulationData,
        uint64_t timestep);  // Returns true if parameters have been changed
};
