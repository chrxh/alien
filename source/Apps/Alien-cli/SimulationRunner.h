#pragma once

#include <cstdint>
#include <optional>

#include <Engine/Interface/Definitions.h>

class ConsoleLiveOutput;
class PeriodicUploader;

class SimulationRunner
{
public:
    static uint64_t calcTimestepsWithLiveOutput(
        SimulationFacade const& simulationFacade,
        std::optional<uint64_t> timesteps,
        ConsoleLiveOutput& liveOutput,
        PeriodicUploader* periodicUploader);
};
