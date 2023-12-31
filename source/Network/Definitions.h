#pragma once

#include "Base/Definitions.h"

class NetworkService;

using NetworkResourceType = int;
enum NetworkResourceType_
{
    NetworkResourceType_Simulation,
    NetworkResourceType_Genome,
    NetworkResourceType_Count
};

using WorkspaceType = int;
enum WorkspaceType_
{
    WorkspaceType_AlienProject,
    WorkspaceType_Shared,
    WorkspaceType_Count
};


struct _NetworkResourceRawTO;
using NetworkResourceRawTO = std::shared_ptr<_NetworkResourceRawTO>;

struct _NetworkResourceTreeTO;
using NetworkResourceTreeTO = std::shared_ptr<_NetworkResourceTreeTO>;
