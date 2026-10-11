#pragma once

#include <string>

#include <Base/Cache.h>

#include <Data/Descs.h>

using _DownloadCache = Cache<std::string, SimulationDesc, 5>;
using DownloadCache = std::shared_ptr<_DownloadCache>;
