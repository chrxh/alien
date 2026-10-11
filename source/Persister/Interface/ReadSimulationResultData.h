#pragma once

#include <filesystem>

#include <Data/Interface/Descs.h>

struct ReadSimulationResultData
{
    std::filesystem::path filename;
    SimulationDesc simulationDesc;
};
