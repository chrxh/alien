#pragma once

#include <string>
#include <unordered_map>

struct GetSimulationPicturesResultData
{
    std::unordered_map<std::string, std::string> jpgBySimId;
};
