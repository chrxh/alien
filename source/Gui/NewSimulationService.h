#pragma once

#include <string>

#include <Base/Definitions.h>
#include <Base/Macros.h>
#include <Base/Singleton.h>

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
};
