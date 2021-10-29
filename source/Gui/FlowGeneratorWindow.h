#pragma once

#include "EngineImpl/Definitions.h"
#include "EngineInterface/FlowFieldSettings.h"
#include "Definitions.h"

class _FlowGeneratorWindow
{
public:
    _FlowGeneratorWindow(SimulationController const& simController);
    ~_FlowGeneratorWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    FlowCenter createFlowCenter();

    SimulationController _simController;

    bool _on = false;
};