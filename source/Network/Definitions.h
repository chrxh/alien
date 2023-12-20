#pragma once

#include "Base/Definitions.h"

class _NetworkController;
using NetworkController = std::shared_ptr<_NetworkController>;

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
