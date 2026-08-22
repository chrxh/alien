#pragma once

#include "Base.cuh"

// Counters that break the runtime of cudaNextTimestep_physics_calcFluidForces down into its phases. The kernel is
// orders of magnitude slower on some GPUs than its instruction count and memory traffic can explain, and a wall-clock
// figure per kernel does not say where the time goes. Filled only while enabled (debug mode), so the regular path pays
// for one predicated branch per object.
//
// Every block accumulates in registers and adds its totals once at the end of the kernel, which keeps the counters out
// of the inner loops.
struct FluidKernelProfile
{
    // Phases of one object iteration, in SM clock cycles. "sync" covers the wait for the slowest thread of the block,
    // since the scan loop has no barrier of its own: a large share there means the threads diverge heavily.
    unsigned long long initCycles;
    unsigned long long scanCycles;
    unsigned long long syncAndReduceCycles;
    unsigned long long tailCycles;

    // Whole block, from its first to its last instruction. Cycles count only what the SM executed for this block,
    // nanoseconds count what passed on the wall clock. If the two disagree, the block was descheduled rather than busy.
    unsigned long long blockCycles;
    unsigned long long blockNanoseconds;

    // Work actually done, to tell a slow GPU apart from a simulation state that simply asks for more.
    unsigned long long numBlocks;
    unsigned long long numObjects;
    unsigned long long numScanCells;  // Map cells visited, summed over all threads
    unsigned long long numRecords;    // Object-map records walked, summed over all threads
    unsigned long long numLaunches;
};

extern __device__ FluidKernelProfile cudaFluidKernelProfile;
extern __device__ int cudaFluidKernelProfilingEnabled;

__device__ __inline__ unsigned long long readSmClock()
{
    return static_cast<unsigned long long>(clock64());
}

// Device-wide nanosecond timer. clock64() runs per SM and its origins differ between SMs, so spans across blocks can
// only be compared through this one.
__device__ __inline__ unsigned long long readGlobalTimer()
{
#if defined(USE_HIP)
    return static_cast<unsigned long long>(wall_clock64());
#else
    unsigned long long result;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(result));
    return result;
#endif
}
