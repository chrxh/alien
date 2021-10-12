#pragma once

#include "Definitions.h"
#include "EngineImpl/Definitions.h"

class _FlowFieldWindow
{
public:
    _FlowFieldWindow(SimulationController const& simController);

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    SimulationController _simController;

    bool _on = false;
    std::vector<int> _activeTabs;
};