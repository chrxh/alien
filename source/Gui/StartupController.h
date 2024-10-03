#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "PersisterInterface/PersisterRequestId.h"
#include "Definitions.h"

class _StartupController
{
public:
    _StartupController(
        SimulationController const& simController,
        PersisterController const& persisterController,
        TemporalControlWindow const& temporalControlWindow);

    void process();
    enum class State
    {
        StartLoadSimulation,
        LoadingSimulation,
        FadeOutLoadingScreen,
        LoadingControls,
        Ready
    };
    State getState();

    void activate();

private:
    void processLoadingScreen();

    void drawGrid(float yPos, float alpha);

    SimulationController _simController;
    TemporalControlWindow _temporalControlWindow;
    PersisterController _persisterController;

    PersisterRequestId _startupSimRequestId;
    State _state = State::StartLoadSimulation;
    TextureData _logo;
    std::optional<std::chrono::steady_clock::time_point> _startupTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _lastActivationTimepoint;
    float _lineDistance = 0;
};
