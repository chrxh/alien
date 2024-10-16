#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _SpatialControlWindow : public _AlienWindow
{
public:
    _SpatialControlWindow(SimulationFacade const& simulationFacade, TemporalControlWindow const& temporalControlWindow);
    ~_SpatialControlWindow();

private:
    void processIntern() override;
    void processBackground() override;

    void processZoomInButton();
    void processZoomOutButton();
    void processCenterButton();
    void processResizeButton();

    void processCenterOnSelection();

    SimulationFacade _simulationFacade;
    ResizeWorldDialog _resizeWorldDialog;

    bool _centerSelection = false;
};