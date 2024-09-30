#include <iostream>
#include <cstring>

#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"
#include "Base/FileLogger.h"
#include "EngineInterface/SerializerService.h"
#include "EngineImpl/SimulationControllerImpl.h"
#include "PersisterImpl/PersisterControllerImpl.h"

#include "MainWindow.h"
#include "GuiLogger.h"
#include "HelpStrings.h"

namespace
{
    bool isInDebugMode(int argc, char** argv)
    {
        return argc == 2 && strcmp(argv[1], "-d") == 0;
    }
}

int main(int argc, char** argv)
{
    auto inDebugMode = isInDebugMode(argc, argv);
    GlobalSettings::getInstance().setDebugMode(inDebugMode);

    GuiLogger logger = std::make_shared<_GuiLogger>();
    FileLogger fileLogger = std::make_shared<_FileLogger>();

    if (inDebugMode) {
        log(Priority::Important, "DEBUG mode");
    }

    SimulationController simController;
    PersisterController persisterController;
    MainWindow mainWindow;

    try {
        simController = std::make_shared<_SimulationControllerImpl>();
        persisterController = std::make_shared<_PersisterControllerImpl>();
        mainWindow = std::make_shared<_MainWindow>(simController, logger);
        mainWindow->mainLoop();
        mainWindow->shutdown();

    } catch (std::exception const& e) {
        std::cerr << "An uncaught exception occurred: "
                  << e.what()
                  << std::endl
                  << std::endl
                  << Const::GeneralInformation
                  << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred."
                  << std::endl
                  << std::endl
                  << Const::GeneralInformation
                  << std::endl;
    }
    return 0;
}
