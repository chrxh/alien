#pragma once

#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class MainWindow
{
public:
    GLFWwindow* init(SimulationController* simController);
    void mainLoop(GLFWwindow* window);
    void shutdown(GLFWwindow* window);

private:
    void processEvents();

    void drawMenubar();
    void drawToolbar();

    SimulationController* _simController = nullptr;
    MacroView* _macroView = nullptr;
};