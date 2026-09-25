#pragma once

#include <filesystem>

#include <Data/Descs.h>

struct ReadSimulationResultData
{
    std::filesystem::path filename;
    SimulationDesc simulationDesc;
};
