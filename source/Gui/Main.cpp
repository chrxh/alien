#include <iostream>

#include "Base/LoggingService.h"
#include "EngineImpl/SimulationController.h"
#include "EngineInterface/Serializer.h"

#include "MainWindow.h"
#include "Resources.h"
#include "SimpleLogger.h"
#include "FileLogger.h"

int main(int, char**)
{
    SimpleLogger logger = boost::make_shared<_SimpleLogger>();
    FileLogger fileLogger = boost::make_shared<_FileLogger>();

    SimulationController simController;
    MainWindow mainWindow;

    try {
        simController = boost::make_shared<_SimulationController>();
        mainWindow = boost::make_shared<_MainWindow>(simController, logger);

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
