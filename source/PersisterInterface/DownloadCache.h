#pragma once

#include <string>

#include "Base/Cache.h"
#include "EngineInterface/DeserializedSimulation.h"

using _DownloadCache = Cache<std::string, DeserializedSimulation, 5>;
using DownloadCache = std::shared_ptr<_DownloadCache>;
