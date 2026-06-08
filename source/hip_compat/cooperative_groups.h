#pragma once
// CUDA <cooperative_groups.h> maps to HIP's header. Basics used here
// (this_thread_block, tiled_partition<32>, thread_rank, shfl_xor, size) work on
// both wave32 and wave64: tiled_partition<32>'s shfl uses the tile size as the
// shuffle width and stays within the tile's lane span on a 64-lane wavefront.
#include <hip/hip_cooperative_groups.h>
