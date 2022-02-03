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

    void processZoomInButton();
    void processZoomOutButton();
    void processResizeButton();

    void processResizeDialog();
    void processCenterOnSelection();

    void onResizing();

    SimulationController _simController;
    Viewport _viewport;

    bool _showResizeDialog = false;
    bool _scaleContent = false;
    bool _centerSelection = false;
    int _width = 0;
    int _height = 0;
};