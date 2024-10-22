#pragma once

#include "EngineInterface/Definitions.h"
#include "PersisterInterface/Definitions.h"
#include "Network/Definitions.h"

#include "Definitions.h"

class _MainWindow
{
public:
    _MainWindow(SimulationFacade const& simulationFacade, PersisterFacade const& persisterFacade, GuiLogger const& logger);
    void mainLoop();
    void shutdown();

private:
    void initGlfwAndOpenGL();
    void initGlad();
    void initFileDialogs();

    GuiLogger _logger;

    PersisterFacade _persisterFacade;
    SimulationFacade _simulationFacade;
};