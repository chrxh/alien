#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _StartupWindow
{
public:
    _StartupWindow(SimulationController const& simController, Viewport const& viewport);

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
    Viewport _viewport;

    State _state = State::Unintialized;
    TextureData _logo;
    std::optional<std::chrono::steady_clock::time_point> _startupTimepoint;
    std::optional<std::chrono::steady_clock::time_point> _lastActivationTimepoint;
};
