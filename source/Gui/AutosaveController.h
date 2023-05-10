#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _AutosaveController
{
public:
    _AutosaveController(SimulationController const& simController, Viewport const& viewPort);
    ~_AutosaveController();

    void shutdown();

    bool isOn() const;
    void setOn(bool value);

    void process();

private:
    void onSave();

    SimulationController _simController;
    Viewport _viewport;

    bool _on = true;
    std::optional<std::chrono::steady_clock::time_point> _startTimePoint;
    bool _alreadySaved = false;
};