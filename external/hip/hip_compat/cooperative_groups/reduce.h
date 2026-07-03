#pragma once

// HIP build shim for <cooperative_groups/reduce.h>.
//
// ROCm 7.13+ provides cg::plus/greater/less and cg::reduce() natively in
// amd_hip_cooperative_groups.h / amd_hip_cooperative_groups_reduce.h.
// When those headers are present, forward directly to the real headers.
//
// ROCm 7.2.x (gfx90a/gfx1100) provides neither.
// The shim then supplies its own implementations.
//
// Active-lane handling (shim path only). The two fluid kernels launch at
// blockDim 25 and 81 (smoothingLength is passed raw, never rounded to a wave
// multiple), so the last tile is PARTIAL: 25-of-32 for the inner kernel and
// 32/32/17 for the boundary kernel. The resident threads are the contiguous
// prefix [0, A) of the tile; lanes A..31 never executed and their value
// registers are undefined at the source level. A plain shfl_xor butterfly is an
// all-reduce in which lane 0 sums every lane, so it would fold those undefined
// registers into the result unless a shuffle of a non-resident lane returns 0 --
// which is unspecified at the HIP source level. We remove that dependency by
// substituting the reduction identity for any partner read from a non-resident
// lane (rank ^ offset >= A). With identity-substitution the lane-0 result is the
// exact same XOR-butterfly summation tree (in the same order) as a butterfly in
// which the non-resident lanes hold the identity, so lane 0 -- the only lane the
// call sites consume -- yields precisely the resident-lane reduction, bit-for-bit
// and identically on wave32 and wave64, with no reliance on inactive-lane
// shuffle behaviour. (Lanes A..31 still hold partial values and must not be read;
// the call sites only ever use lane 0.)

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#if __has_include(<hip/amd_detail/amd_hip_cooperative_groups_reduce.h>)
// ROCm 7.13+: real cg::plus/greater/less and cg::reduce() are already in scope
// via hip_cooperative_groups.h / amd_hip_cooperative_groups.h. Pull in the
// reduce() overload set from the SDK.
#include <hip/cooperative_groups/hip_reduce.h>

#else
// ROCm 7.2.x: no native cg::reduce. Provide shim functors and reduce().

#include <limits>

namespace cooperative_groups {

// Each functor exposes identity() -- the value that leaves the accumulator
// unchanged -- so reduce() can substitute it for a non-resident partner lane.
template <typename T>
struct plus
{
    __device__ __forceinline__ T operator()(T const& a, T const& b) const { return a + b; }
    __device__ __forceinline__ T identity() const { return T(0); }
};

template <typename T>
struct greater
{
    __device__ __forceinline__ T operator()(T const& a, T const& b) const { return a > b ? a : b; }
    __device__ __forceinline__ T identity() const { return std::numeric_limits<T>::lowest(); }
};

template <typename T>
struct less
{
    __device__ __forceinline__ T operator()(T const& a, T const& b) const { return a < b ? a : b; }
    __device__ __forceinline__ T identity() const { return std::numeric_limits<T>::max(); }
};

template <unsigned int Size, typename ParentT, typename T, typename Op>
__device__ __forceinline__ T reduce(thread_block_tile<Size, ParentT> const& tile, T value, Op op)
{
    unsigned int const tileSize = tile.size();
    unsigned int const blockSize = blockDim.x * blockDim.y * blockDim.z;
    unsigned int const flatRank = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    unsigned int const tileBase = (flatRank / tileSize) * tileSize;
    unsigned int const activeLanes = blockSize > tileBase ? min(tileSize, blockSize - tileBase) : 0u;

    unsigned int const rank = tile.thread_rank();
    for (unsigned int offset = tileSize / 2; offset > 0; offset >>= 1) {
        T const partner = tile.shfl_xor(value, offset);
        value = op(value, (rank ^ offset) < activeLanes ? partner : op.identity());
    }
    return value;
}

}  // namespace cooperative_groups

#endif  // __has_include amd_hip_cooperative_groups_reduce.h
