#pragma once

#include <algorithm>

// Replaces the former user setting for the number of CUDA blocks. That value could not be chosen sensibly from outside
// -- it depends on the GPU, and on how a kernel splits its work -- so it is derived from the device at startup.
//
// Two kernel families need different things from it:
//
//  - Kernels iterating with a grid stride cover their entities with blockDim.x * gridDim.x threads, so what matters is
//    that the grid holds the GPU busy.
//  - Kernels dividing their entities by the grid size hand each block a fixed share, so a larger grid means smaller
//    shares and better load balance, bounded by the launch overhead of the extra blocks.
//
// Both are served by scaling with the number of multiprocessors. The reference point is a GPU with 128 of them, which
// is where the former fixed value of 16384 was tuned.
struct KernelLaunchSettings
{
    int numBlocks = 16384;

    // How many warps a block of the warp-cooperative kernels holds. One warp per block leaves architectures idle that
    // keep only a single block resident per multiprocessor, and costs registers on those that keep many; see
    // calcWarpsPerBlock.
    int fluidWarpsPerBlock = 1;

    bool operator==(KernelLaunchSettings const& other) const = default;

    static int calcNumBlocks(int multiProcessorCount)
    {
        auto constexpr BlocksPerMultiProcessor = 128;
        return std::clamp(multiProcessorCount * BlocksPerMultiProcessor, 1024, 65536);
    }

    // The fluid kernels work on one object per warp. Where those warps come from depends on how many blocks the
    // hardware keeps resident for the kernel, and the two regimes want opposite things:
    //
    //  - Many resident blocks: one warp per block. The object index stays block-uniform, which needs about 25 fewer
    //    registers and buys back the occupancy those registers would cost.
    //  - Few resident blocks: the warps have to come from inside the block, or the multiprocessor works on a single
    //    object at a time. Blackwell consumer GPUs report one resident block per multiprocessor here, which left them
    //    at 32 of 1536 threads and an order of magnitude behind.
    //
    // Only these two are instantiated; a value in between would trade one effect against the other without winning
    // either. The threshold sits well below what every architecture that likes small blocks reports.
    static int calcWarpsPerBlock(int blocksPerMultiProcessor)
    {
        auto constexpr WarpsPerBlockWhenBlocksAreScarce = 16;
        auto constexpr EnoughResidentBlocks = 8;
        return blocksPerMultiProcessor >= EnoughResidentBlocks ? 1 : WarpsPerBlockWhenBlocksAreScarce;
    }
};
