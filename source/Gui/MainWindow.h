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
    char const* initGlfwAndReturnGlslVersion();
    void initFileDialogs();

    void mainLoopForLoadingScreen();
    void mainLoopForFadeoutLoadingScreen();
    void mainLoopForFadeInUI();
    void mainLoopForUI();

    void renderSimulation();

    void processMenubar();
    void processDialogs();
    void processControllers();

    void onRunSimulation();
    void onPauseSimulation();
    void onExit();

    void finishFrame();

    void pushGlobalStyle();
    void popGlobalStyle();

    GuiLogger _logger;

    PersisterFacade _persisterFacade;
    SimulationFacade _simulationFacade;

    bool _onExit = false;
    bool _simulationMenuToggled = false;
    bool _networkMenuToggled = false;
    bool _windowMenuToggled = false;
    bool _settingsMenuToggled = false;
    bool _viewMenuToggled = false;
    bool _editorMenuToggled = false;
    bool _toolsMenuToggled = false;
    bool _helpMenuToggled = false;
    bool _renderSimulation = true;
};