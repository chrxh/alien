#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _SpatialControlWindow : public _AlienWindow
{
public:
    _SpatialControlWindow(SimulationController const& simController, TemporalControlWindow const& temporalControlWindow);
    ~_SpatialControlWindow();

private:
    void processIntern() override;
    void processBackground() override;

    void processZoomInButton();
    void processZoomOutButton();
    void processCenterButton();
    void processResizeButton();

    void processCenterOnSelection();

    SimulationController _simController;
    ResizeWorldDialog _resizeWorldDialog;

    bool _centerSelection = false;
};