#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _AutosaveController
{
public:
    _AutosaveController(SimulationFacade const& simulationFacade);
    ~_AutosaveController();

    void shutdown();

    bool isOn() const;
    void setOn(bool value);

    void process();

private:
    void onSave();

    SimulationFacade _simulationFacade;

    bool _on = true;
    std::optional<std::chrono::steady_clock::time_point> _startTimePoint;
    bool _alreadySaved = false;
};