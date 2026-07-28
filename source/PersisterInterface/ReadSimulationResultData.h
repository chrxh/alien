#pragma once

#include <filesystem>

#include <EngineInterface/Descs.h>

struct ReadSimulationResultData
{
    std::filesystem::path filename;
    SimulationDesc simulationDesc;
};
