#pragma once

#include "Base/Definitions.h"

class NetworkService;

using NetworkResourceType = int;
enum NetworkResourceType_
{
    NetworkResourceType_Simulation,
    NetworkResourceType_Genome
};


struct _NetworkResourceRawTO;
using NetworkResourceRawTO = std::shared_ptr<_NetworkResourceRawTO>;

struct _NetworkResourceTreeTO;
using NetworkResourceTreeTO = std::shared_ptr<_NetworkResourceTreeTO>;
