#pragma once

#include "AlienDialog.h"
#include "Definitions.h"

class _ResizeWorldDialog : public _AlienDialog
{
public:
    _ResizeWorldDialog(SimulationFacade const& simulationFacade, TemporalControlWindow const& temporalControlWindow);

    void open();

private:
    void processIntern() override;

    void onResizing();

    SimulationFacade _simulationFacade;
    TemporalControlWindow _temporalControlWindow;

    bool _scaleContent = false;
    int _width = 0;
    int _height = 0;
};
