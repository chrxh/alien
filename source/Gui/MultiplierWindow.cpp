#include "MultiplierWindow.h"

#include "imgui.h"

_MultiplierWindow::_MultiplierWindow(SimulationController const& simController, Viewport const& viewport)
    : _AlienWindow("Multiplier", "editor.multiplier", false)
    , _simController(simController)
    , _viewport(viewport)
{}

void _MultiplierWindow::processIntern() {}

