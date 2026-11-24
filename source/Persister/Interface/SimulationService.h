#pragma once

#include <functional>
#include <optional>
#include <string>

#include <Base/Definitions.h>
#include <Base/Macros.h>
#include <Base/Singleton.h>

#include <Data/Descs.h>

class SimulationService
{
    MAKE_SINGLETON(SimulationService);

public:
    struct NewSimulationParameters
    {
        MEMBER(NewSimulationParameters, std::string, projectName, "");
        MEMBER(NewSimulationParameters, IntVector2D, worldSize, IntVector2D());
        MEMBER(NewSimulationParameters, float, externalEnergy, 0.0f);
        MEMBER(NewSimulationParameters, bool, adoptSimulationParameters, true);
    };
    void createSimulation(NewSimulationParameters const& parameters);
    std::optional<std::string> loadSimulation(SimulationDesc simulation);
    void replaceSimulation(std::function<void()> const& setupFunc);
};
