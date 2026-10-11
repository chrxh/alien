#include <cstring>
#include <iostream>

#include <Base/Interface/AlienExceptions.h>
#include <Base/Interface/FileLogger.h>
#include <Base/Interface/GlobalSettings.h>
#include <Base/Interface/KernelProfiler.h>
#include <Base/Interface/KernelTracer.h>
#include <Base/Interface/LoggingService.h>
#include <Base/Interface/Resources.h>

#include <Engine/Impl/SimulationFacadeImpl.h>

#include <McpTools/Impl/McpToolsFacadeImpl.h>

#include <Persister/Impl/PersisterFacadeImpl.h>

#include <Rendering/Impl/RenderingFacadeImpl.h>

#include <Gui/Interface/GuiFacade.h>

#include <Gui/Impl/GuiFacadeImpl.h>

#include "HelpStrings.h"


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
    auto useInterop = !hasArgument(argc, argv, "--no-interop");
    GlobalSettings::get().setDebugMode(inDebugMode);
    GlobalSettings::get().setInterop(useInterop);

    FileLogger fileLogger = std::make_shared<_FileLogger>();

    auto error = false;
    try {
        log(Priority::Important, "starting ALIEN v" + Const::ProgramVersion);

        if (inDebugMode) {
            log(Priority::Important, "DEBUG mode: writing " + Const::ProfileFilename.string() + " and " + Const::TraceFilename.string());
            KernelProfiler::get().init(Const::ProfileFilename);
            KernelTracer::get().init(Const::TraceFilename);
        }
        if (!useInterop) {
            log(Priority::Important, "INTEROP disabled by command line: rendering data takes the detour over host memory");
        }

        _SimulationFacadeImpl::set(std::make_shared<_SimulationFacadeImpl>());
        _PersisterFacadeImpl::set(std::make_shared<_PersisterFacadeImpl>());
        _McpToolsFacadeImpl::set(std::make_shared<_McpToolsFacadeImpl>());
        _RenderingFacadeImpl::set(std::make_shared<_RenderingFacadeImpl>());
        _GuiFacadeImpl::set(std::make_shared<_GuiFacadeImpl>());

        _GuiFacade::get()->setup();
        _GuiFacade::get()->runMainLoop();
        _GuiFacade::get()->shutdown();

    } catch (InitialCheckException const& e) {
        log(Priority::Important, std::string("Initial checks failed: ") + e.what());
        log(Priority::Important, "Callstack:\n" + e.getCallstack());
        error = true;
    } catch (AlienException const& e) {
        log(Priority::Important, std::string("An exception occurred: ") + e.what());
        log(Priority::Important, "Callstack:\n" + e.getCallstack());
        error = true;
    } catch (std::exception const& e) {
        log(Priority::Important, std::string("An exception occurred: ") + e.what());
        error = true;
    } catch (...) {
        log(Priority::Important, std::string("An unknown exception occurred."));
        error = true;
    }
    if (error) {
        std::cerr << LoggingService::get().getLogString();
        std::cerr << std::endl << std::endl << Const::getGeneralInformation() << std::endl;
        return 1;
    }
    return 0;
}
