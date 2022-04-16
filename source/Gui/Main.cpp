#include <iostream>

#include "Base/LoggingService.h"
#include "EngineInterface/Serializer.h"
#include "EngineImpl/SimulationControllerImpl.h"
#include "Base/Resources.h"

#include "MainWindow.h"
#include "SimpleLogger.h"
#include "FileLogger.h"

int main(int, char**)
{
    SimpleLogger logger = std::make_shared<_SimpleLogger>();
    FileLogger fileLogger = std::make_shared<_FileLogger>();

    SimulationController simController;
    MainWindow mainWindow;

    try {
        simController = std::make_shared<_SimulationControllerImpl>();
        mainWindow = std::make_shared<_MainWindow>(simController, logger);

        simController->initCuda();

        mainWindow->mainLoop();

        mainWindow->shutdown();
        simController->closeSimulation();
    } catch (std::exception const& e) {
        auto message = std::string("The following exception occurred: ")
            + e.what();
        log(Priority::Important, message);
        std::cerr << message
                  << std::endl
                  << std::endl
                  << "See log.txt for more detailed information."
                  << std::endl;
    }
    return 0;
}
