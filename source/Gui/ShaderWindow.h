#pragma once

#include "AlienWindow.h"
#include "Definitions.h"

class _ShaderWindow : public _AlienWindow
{
public:
    _ShaderWindow(SimulationView const& simulationView);
    ~_ShaderWindow();

private:
    void processIntern() override;

    SimulationView _simulationView;
};