#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _MainWindow
{
public:
    GLFWwindow* init(SimulationController const& simController);
    void mainLoop(GLFWwindow* window);
    void shutdown(GLFWwindow* window);

private:
    void processEvents();

    void drawMenubar();
    void drawToolbar();
    void drawDialogs();

    SimulationController _simController;
    SimulationView _simulationView;
};