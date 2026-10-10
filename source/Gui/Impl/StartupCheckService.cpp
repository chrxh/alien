#include "StartupCheckService.h"

#include <filesystem>

#include <Base/Interface/AlienExceptions.h>
#include <Base/Interface/LoggingService.h>
#include <Base/Interface/Resources.h>

void StartupCheckService::check()
{
    log(Priority::Important, "check if resource folder exist");
    if (!std::filesystem::exists(Const::ResourcePath)) {
        throw InitialCheckException("The resource folder has not been found. Please start ALIEN from the directory in which the folder `resource` is located.");
    }

    log(Priority::Important, "check if cuda device exist");
    _SimulationFacade::get()->getGpuName();
}
