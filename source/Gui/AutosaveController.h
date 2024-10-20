#pragma once

#include <chrono>

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class AutosaveController : public MainLoopEntity
{
    MAKE_SINGLETON(AutosaveController);

public:
    void init(SimulationFacade const& simulationFacade);

    bool isOn() const;
    void setOn(bool value);

private:
    void process() override;
    void shutdown() override;

    void onSave();

    SimulationFacade _simulationFacade;

    bool _on = true;
    std::optional<std::chrono::steady_clock::time_point> _startTimePoint;
    bool _alreadySaved = false;
};