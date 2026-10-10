#pragma once

struct KernelLaunchSettings
{
    int numBlocks = 16384;

    int fluidWarpsPerBlock = 1;

    bool operator==(KernelLaunchSettings const& other) const = default;
};
