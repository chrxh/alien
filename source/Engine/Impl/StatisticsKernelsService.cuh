#pragma once

#include <Base/Singleton.h>

#include <Engine/Interface/KernelLaunchSettings.h>

#include <Engine/Impl/Kernels/Base.cuh>
#include <Engine/Impl/Kernels/Definitions.cuh>
#include <Engine/Impl/Kernels/Macros.cuh>

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
