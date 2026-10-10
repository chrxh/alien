#pragma once

#include <string>

#include <Base/Interface/Cache.h>

#include <Data/Interface/Descs.h>

using _DownloadCache = Cache<std::string, SimulationDesc, 5>;
using DownloadCache = std::shared_ptr<_DownloadCache>;
