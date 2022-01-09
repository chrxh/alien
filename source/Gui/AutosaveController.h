#pragma once

#include <chrono>

#include "EngineImpl/SimulationController.h"
#include "Definitions.h"

class _AutosaveController
{
public:
    _AutosaveController(SimulationController const& simController);
    ~_AutosaveController();

    void shutdown();

    bool isOn() const;
    void setOn(bool value);

    void process();

private:
    void onSave();

    SimulationController _simController;

    bool _on = true;
    std::optional<std::chrono::steady_clock::time_point> _startTimePoint;
};