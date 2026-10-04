#include <functional>
#include <optional>

#include <Base/Singleton.h>

#include <EngineInterface/SettingsForSimulation.h>
#include <EngineInterface/SimulationParametersUpdateConfig.h>

#include <EngineKernels/Definitions.cuh>

class SimulationParametersUpdateService
{
    MAKE_SINGLETON(SimulationParametersUpdateService);

public:
    SimulationParameters integrateChanges(
        SimulationParameters const& currentParameters,
        SimulationParameters const& changedParameters,
        SimulationParametersUpdateConfig const& updateConfig) const;

    // Returns true if parameters have been changed
    bool updateSimulationParametersAfterTimestep(SettingsForSimulation& settings, std::function<double()> const& readExternalEnergy, uint64_t timestep);
};
