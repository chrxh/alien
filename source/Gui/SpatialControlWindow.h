#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _SpatialControlWindow : public AlienWindow
{
public:
    _SpatialControlWindow(SimulationFacade const& simulationFacade);
    ~_SpatialControlWindow() override;

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