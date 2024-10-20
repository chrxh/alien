#pragma once

#include <chrono>

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "ShutdownInterface.h"

class AutosaveController : public ShutdownInterface
{
    MAKE_SINGLETON(AutosaveController);

public:
    void init(SimulationFacade const& simulationFacade);

    bool isOn() const;
    void setOn(bool value);

    void process();

private:
    void shutdown() override;

    void onSave();

    SimulationFacade _simulationFacade;

    bool _on = true;
    std::optional<std::chrono::steady_clock::time_point> _startTimePoint;
    bool _alreadySaved = false;
};