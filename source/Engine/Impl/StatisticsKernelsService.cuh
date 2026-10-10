#pragma once

#include <Base/Interface/Singleton.h>

#include <Engine/Interface/KernelLaunchSettings.h>

#include <Engine/Kernels/Base.cuh>
#include <Engine/Kernels/Definitions.cuh>
#include <Engine/Kernels/Macros.cuh>

class StatisticsKernelsService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(StatisticsKernelsService);

public:
    void init();
    void shutdown();

    void updateStatistics(KernelLaunchSettings const& launchSettings, SimulationData const& data, SimulationStatistics const& simulationStatistics);

private:
    StatisticsKernelsService() = default;
};
