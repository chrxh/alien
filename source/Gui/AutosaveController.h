#pragma once

#include <chrono>

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"

class AutosaveController
{
    MAKE_SINGLETON(AutosaveController);

public:
    void init(SimulationFacade const& simulationFacade);
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