#pragma once
// CUDA's sm_60 atomics (atomicAdd on double, etc.) are part of the HIP runtime.
#include <EngineKernels/cuda_to_hip.h>
