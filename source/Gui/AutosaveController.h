#pragma once

#include <chrono>

#include "Base/Singleton.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"
#include "MainLoopEntity.h"

class AutosaveController : public MainLoopEntity<SimulationFacade>
{
    MAKE_SINGLETON(AutosaveController);

public:
    bool isOn() const;
    void setOn(bool value);

private:
    void init(SimulationFacade simulationFacade) override;
    void process() override;
    void shutdown() override;

    void onSave();

    SimulationFacade _simulationFacade;

    bool _on = true;
    std::optional<std::chrono::steady_clock::time_point> _startTimePoint;
    bool _alreadySaved = false;
};