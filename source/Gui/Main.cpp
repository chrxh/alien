#include <cstring>
#include <iostream>

#include <Base/AlienExceptions.h>
#include <Base/FileLogger.h>
#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>
#include <Base/StringHelper.h>

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
    GlobalSettings::get().setInterop(useInterop);

    FileLogger fileLogger = std::make_shared<_FileLogger>();

    MainWindow mainWindow;

    auto error = false;
    try {
        log(Priority::Important, "starting ALIEN v" + Const::ProgramVersion);

        if (inDebugMode) {
            log(Priority::Important, "DEBUG mode");
        }
        if (useInterop) {
            log(Priority::Important, "INTEROP mode: Using CUDA-OpenGL interop for rendering");
        }

        _SimulationFacadeImpl::set(std::make_shared<_SimulationFacadeImpl>());
        _PersisterFacadeImpl::set(std::make_shared<_PersisterFacadeImpl>());

        mainWindow = std::make_shared<_MainWindow>();
        mainWindow->mainLoop();
        mainWindow->shutdown();

    } catch (InitialCheckException const& e) {
        log(Priority::Important, std::string("Initial checks failed: ") + e.what());
        log(Priority::Important, "Callstack:\n" + e.getCallstack());
        error = true;
    } catch (AlienException const& e) {
        log(Priority::Important, std::string("An uncaught exception occurred: ") + e.what());
        log(Priority::Important, "Callstack:\n" + e.getCallstack());
        error = true;
    } catch (std::exception const& e) {
        log(Priority::Important, std::string("An uncaught exception occurred: ") + e.what());
        error = true;
    } catch (...) {
        log(Priority::Important, std::string("An unknown exception occurred."));
        error = true;
    }
    if (error) {
        std::cerr << LoggingService::get().getLogString();
        std::cerr << std::endl << std::endl << Const::GeneralInformation << std::endl;
        return 1;
    }
    return 0;
}
