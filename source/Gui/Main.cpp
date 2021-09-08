#include "Base/BaseServices.h"
#include "EngineImpl/SimulationController.h"
#include "EngineInterface/EngineInterfaceSettings.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/ChangeDescriptions.h"

#include "MainWindow.h"

int main(int, char**)
{
    BaseServices baseServices;

    MainWindow mainWindow = boost::make_shared<_MainWindow>();
    SimulationController simController = boost::make_shared<_SimulationController>();

    try {
        simController->initCuda();

        Serializer serializer = boost::make_shared<_Serializer>();
        SerializedSimulation serializedData;
        serializer->loadSimulationDataFromFile("d:\\temp\\simulations\\evolution.sim", serializedData);
        auto deserializedData = serializer->deserializeSimulation(serializedData);

        simController->newSimulation(
            deserializedData.generalSettings.worldSize,
            deserializedData.timestep,
            deserializedData.simulationParameters,
            deserializedData.generalSettings.gpuConstants);

        simController->updateData(deserializedData.content);

        simController->runSimulation();

        auto glfwWindow = mainWindow->init(simController);
        if (!glfwWindow) {
            return 1;
        }

        mainWindow->mainLoop(glfwWindow);

        mainWindow->shutdown(glfwWindow);
        simController->closeSimulation();
    } catch (std::exception const& e) {
        printf("%s\n", e.what());
    }
    return 0;
}
