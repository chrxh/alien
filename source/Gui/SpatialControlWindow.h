#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"
#include "AlienWindow.h"

class _SpatialControlWindow : public _AlienWindow
{
public:
    _SpatialControlWindow(SimulationController const& simController, Viewport const& viewport);

private:
    void processIntern() override;
    void processBackground() override;

    void processZoomInButton();
    void processZoomOutButton();
    void processResizeButton();

    void processCenterOnSelection();

    SimulationController _simController;
    Viewport _viewport;
    ResizeWorldDialog _resizeWorldDialog;

    bool _centerSelection = false;
};