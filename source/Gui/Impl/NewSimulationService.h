#pragma once

#include <optional>
#include <string>

#include <Base/Interface/Definitions.h>
#include <Base/Interface/Macros.h>
#include <Base/Interface/Singleton.h>

#include <Data/Interface/Descs.h>

class NewSimulationService
{
    MAKE_SINGLETON(NewSimulationService);

public:
    struct Parameters
    {
        MEMBER(Parameters, std::string, projectName, "");
        MEMBER(Parameters, IntVector2D, worldSize, IntVector2D());
        MEMBER(Parameters, float, externalEnergy, 0.0f);
        MEMBER(Parameters, bool, adoptSimulationParameters, true);
    };
    void createSimulation(Parameters const& parameters);

    std::optional<std::string> loadSimulation(SimulationDesc const& simulation);
};
