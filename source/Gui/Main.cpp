#include "Base/BaseServices.h"
#include "EngineImpl/SimulationController.h"
#include "EngineInterface/EngineInterfaceSettings.h"

#include "MainWindow.h"

int main(int, char**)
{
    BaseServices baseServices;

    MainWindow mainWindow;
    SimulationController simController;

    simController.initCuda();

    simController.newSimulation({ 600, 300 }, 0, SimulationParameters(), GpuConstants());

    auto glfwWindow = mainWindow.init(&simController);
    if (!glfwWindow) {
        return 1;
    }

    mainWindow.mainLoop(glfwWindow);
    mainWindow.shutdown(glfwWindow);
    simController.closeSimulation();

    return 0;
}
