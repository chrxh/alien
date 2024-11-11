#include <iostream>
#include <cstring>

#include "Base/GlobalSettings.h"
#include "Base/LoggingService.h"
#include "Base/FileLogger.h"
#include "Base/Exceptions.h"
#include "PersisterInterface/SerializerService.h"
#include "EngineImpl/SimulationFacadeImpl.h"
#include "PersisterImpl/PersisterFacadeImpl.h"

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
    GlobalSettings::get().setDebugMode(inDebugMode);

    GuiLogger logger = std::make_shared<_GuiLogger>();
    FileLogger fileLogger = std::make_shared<_FileLogger>();

    if (inDebugMode) {
        log(Priority::Important, "DEBUG mode");
    }

    SimulationFacade simulationFacade;
    PersisterFacade persisterFacade;
    MainWindow mainWindow;

    try {
        simulationFacade = std::make_shared<_SimulationFacadeImpl>();
        persisterFacade = std::make_shared<_PersisterFacadeImpl>();
        mainWindow = std::make_shared<_MainWindow>(simulationFacade, persisterFacade, logger);
        mainWindow->mainLoop();
        mainWindow->shutdown();

    } catch (InitialCheckException const& e) {
        std::cerr << "Initial checks failed: " << std::endl << e.what() << std::endl;
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
