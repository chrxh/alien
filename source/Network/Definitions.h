#pragma once

#include "Base/Definitions.h"

class _NetworkService;
using NetworkService = std::shared_ptr<_NetworkService>;

using DataType = int;
enum DataType_
{
    DataType_Simulation,
    DataType_Genome
};

struct _BrowserDataTO;
using BrowserDataTO = std::shared_ptr<_BrowserDataTO>;

struct _NetworkDataTO;
using NetworkDataTO = std::shared_ptr<_NetworkDataTO>;
