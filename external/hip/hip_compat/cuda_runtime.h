#pragma once
// HIP build shim: the project includes <cuda_runtime.h>; ROCm ships no such
// header. The CUDA->HIP aliases live in the force-included cuda_to_hip.h.
#include <cuda_to_hip.h>
