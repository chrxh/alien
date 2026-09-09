#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <EngineInterface/KernelLaunchSettings.h>

class KernelLaunchSettingsService
{
    MAKE_SINGLETON(KernelLaunchSettingsService);

public:
    KernelLaunchSettings deriveFromDevice(int deviceNumber) const;

    // Resident warps per multiprocessor for 1, 2, ... MAX_FLUID_WARPS_PER_BLOCK warps per block
    std::vector<int> calcFluidResidentWarps() const;

private:
    int calcNumBlocks(int multiProcessorCount) const;
    int calcFluidWarpsPerBlock() const;
};
