#pragma once

#include <chrono>

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "PersisterInterface/PersisterRequestId.h"

#include "Definitions.h"

class StartupController
{
    MAKE_SINGLETON(StartupController);

public:
    void setup(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade);

    void process();
    enum class State
    {
        StartLoadSimulation,
        LoadingSimulation,
        FadeOutLoadingScreen,
        FadeInUI,
        Ready
    };
    State getState();

    void activate();

private:
    void processLoadingScreen();

    SimulationFacade _simulationFacade;
    PersisterFacade _persisterFacade;

    PersisterRequestId _startupSimRequestId;
    State _state = State::StartLoadSimulation;
    TextureData _logo;
    std::optional<std::chrono::steady_clock::time_point> _startupTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _lastActivationTimepoint;
};
