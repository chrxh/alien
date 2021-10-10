#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _StartupWindow
{
public:
    _StartupWindow(
        SimulationController const& simController,
        Viewport const& viewport,
        TemporalControlWindow const& temporalControlWindow,
        SpatialControlWindow const& spatialControlWindow,
        StatisticsWindow const& statisticsWindow);

    void process();
    enum class State
    {
        Unintialized,
        RequestLoading,
        LoadingSimulation,
        LoadingControls,
        Finished
    };
    State getState();

    void activate();

private:
    void processWindow();

    SimulationController _simController;
    Viewport _viewport;
    TemporalControlWindow _temporalControlWindow;
    SpatialControlWindow _spatialControlWindow;
    StatisticsWindow _statisticsWindow;

    State _state = State::Unintialized;
    TextureData _logo;
    boost::optional<std::chrono::steady_clock::time_point> _lastActivationTimepoint;
};
