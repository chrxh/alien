#include <optional>

#include "Base/Singleton.h"

#include "Definitions.cuh"

class SimulationParameterService
{
    MAKE_SINGLETON(SimulationParameterService);

public:
    SimulationParameters integrateChanges(SimulationParameters const& currentParameters, SimulationParameters const& changedParameters) const;
};
