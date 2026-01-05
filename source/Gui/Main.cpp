#include <cstring>
#include <iostream>

#include <Base/Exceptions.h>
#include <Base/FileLogger.h>
#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>

#include <EngineImpl/SimulationFacadeImpl.h>

#include <PersisterInterface/SerializerService.h>

#include <PersisterImpl/PersisterFacadeImpl.h>

#include "HelpStrings.h"
#include "MainWindow.h"

namespace
{
    bool hasArgument(int argc, char** argv, const char* arg)
    {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], arg) == 0) {
                return true;
            }
        }
        return false;
    }
}

int main(int argc, char** argv)
{
    auto inDebugMode = hasArgument(argc, argv, "-d");
    auto useInterop = hasArgument(argc, argv, "--interop");
    GlobalSettings::get().setDebugMode(inDebugMode);
    if (useInterop) {
        GlobalSettings::get().setNoInterop(false);
    }

    FileLogger fileLogger = std::make_shared<_FileLogger>();

    if (inDebugMode) {
        log(Priority::Important, "DEBUG mode");
    }
    if (useInterop) {
        log(Priority::Important, "INTEROP mode: Using CUDA-OpenGL interop for rendering");
    }

    MainWindow mainWindow;

    try {
        log(Priority::Important, "starting ALIEN v" + Const::ProgramVersion);

        _SimulationFacadeImpl::set(std::make_shared<_SimulationFacadeImpl>());
        _PersisterFacadeImpl::set(std::make_shared<_PersisterFacadeImpl>());

        mainWindow = std::make_shared<_MainWindow>();
        mainWindow->mainLoop();
        mainWindow->shutdown();

    } catch (InitialCheckException const& e) {
        std::cerr << "Initial checks failed: " << std::endl << e.what() << std::endl;
    } catch (std::exception const& e) {
        std::cerr << "An uncaught exception occurred: " << e.what() << std::endl << std::endl << Const::GeneralInformation << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred." << std::endl << std::endl << Const::GeneralInformation << std::endl;
    }
    return 0;
}
