#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _StartupController
{
public:
    _StartupController(SimulationController const& simController, TemporalControlWindow const& temporalControlWindow, Viewport const& viewport);

    void process();
    enum class State
    {
        Unintialized,
        RequestLoading,
        LoadingSimulation,
        LoadingControls,
        FinishedLoading
    };
    State getState();

    void activate();

private:
    void processWindow();

    SimulationController _simController;
    TemporalControlWindow _temporalControlWindow;
    Viewport _viewport;

    State _state = State::Unintialized;
    TextureData _logo;
    std::optional<std::chrono::steady_clock::time_point> _startupTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _lastActivationTimepoint;
};
