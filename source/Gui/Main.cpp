#include <iostream>

#include "Base/BaseServices.h"
#include "Base/LoggingService.h"
#include "Base/ServiceLocator.h"
#include "EngineImpl/SimulationController.h"
#include "EngineInterface/EngineInterfaceSettings.h"
#include "EngineInterface/Serializer.h"
#include "EngineInterface/ChangeDescriptions.h"

#include "MainWindow.h"
#include "Resources.h"
#include "SimpleLogger.h"

int main(int, char**)
{
    BaseServices baseServices;
    SimpleLogger logger = boost::make_shared<_SimpleLogger>();

    try {
        MainWindow mainWindow = boost::make_shared<_MainWindow>();
        SimulationController simController = boost::make_shared<_SimulationController>();

        auto glfwWindow = mainWindow->init(simController, logger);
        simController->initCuda();

        mainWindow->mainLoop(glfwWindow);

        mainWindow->shutdown(glfwWindow);
        simController->closeSimulation();
    } catch (std::exception const& e) {
        auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
        loggingService->logMessage(Priority::Important, std::string("The following exception occurred: ") + e.what());
        for (auto const& message : logger->getMessages(Priority::Important)) {
            std::cerr << message << std::endl;
        }

        std::cerr << std::endl << std::endl << "See log.txt for more detailed information." << std::endl;
    }
    return 0;
}
