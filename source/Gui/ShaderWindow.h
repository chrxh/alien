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
    float _brightness = 1.0f;
    float _contrast = 1.0f;
    float _motionBlur = 1.0f;
};