#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"

class _SpatialControlWindow
{
public:
    _SpatialControlWindow(SimulationController const& simController, Viewport const& viewport);
    ~_SpatialControlWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    void processZoomInButton();
    void processZoomOutButton();
    void processResizeButton();
    void processZoomSensitivitySlider();

    void processResizeDialog();

    void onResizing();

    SimulationController _simController;
    Viewport _viewport;

    bool _on = false;
    bool _showResizeDialog = false;
    bool _scaleContent = false;
    int _width = 0;
    int _height = 0;
};