#pragma once

// CUDA-to-HIP compatibility shim for the ROCm/HIP build (USE_HIP).
//
// This is the single file that knows about HIP. It is force-included on every
// HIP and host translation unit of the AMD build (via
// add_compile_options($<$<COMPILE_LANGUAGE:HIP|CXX>:-include ...>) in the
// top-level CMakeLists.txt) and aliases the
// CUDA runtime, graph, cooperative-groups and OpenGL-interop symbols the
// project uses to their HIP equivalents. On the NVIDIA path this header is not
// compiled at all, so the CUDA build is byte-for-byte unchanged.
//
// Symbols that look like CUDA names but are project identifiers
// (cudaSimulationParameters, cudaSettings, cudaTO*, the cudaNextTimestep_*
// kernels, etc.) are deliberately NOT aliased here.

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

// The HIP language toolchain defines __HIP_PLATFORM_AMD__ automatically, but
// this header is also force-included into host C++ translation units (a host
// .cpp pulls <cuda_fp16.h>); define it there so the HIP runtime headers parse.
#if !defined(__HIP_PLATFORM_AMD__)
#define __HIP_PLATFORM_AMD__
#endif

#include <cstring>
#include <cstdlib>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// --- error handling ---
using cudaError = hipError_t;
using cudaError_t = hipError_t;
#define cudaSuccess                       hipSuccess
#define cudaGetErrorString                hipGetErrorString
#define cudaGetLastError                  hipGetLastError
#define cudaErrorInsufficientDriver       hipErrorInsufficientDriver
#define cudaErrorOperatingSystem          hipErrorOperatingSystem
#define cudaErrorInitializationError      hipErrorNotInitialized
#define cudaErrorUnsupportedPtxVersion    hipErrorInvalidImage
#define cudaErrorMemoryAllocation         hipErrorOutOfMemory
#define cudaErrorIllegalAddress           hipErrorIllegalAddress
#define cudaErrorLaunchFailure            hipErrorLaunchFailure

// --- device management ---
#define cudaDeviceProp                    hipDeviceProp_t
#define cudaGetDeviceCount                hipGetDeviceCount
#define cudaGetDeviceProperties           hipGetDeviceProperties
#define cudaMemGetInfo                    hipMemGetInfo
#define cudaSetDevice                     hipSetDevice
#define cudaDeviceSynchronize             hipDeviceSynchronize
#define cudaDeviceReset                   hipDeviceReset

// --- memory ---
#define cudaMalloc                        hipMalloc
#define cudaFree                          hipFree
#define cudaMemcpy                        hipMemcpy
#define cudaMemset                        hipMemset
#define cudaMemset2D                      hipMemset2D
#define cudaMemcpyToSymbol                hipMemcpyToSymbol
#define cudaMemcpyHostToDevice            hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost            hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice          hipMemcpyDeviceToDevice

// --- streams ---
#define cudaStream_t                      hipStream_t
#define cudaStreamCreate                  hipStreamCreate
#define cudaStreamDestroy                 hipStreamDestroy
#define cudaStreamSynchronize             hipStreamSynchronize

// --- CUDA graphs ---
#define cudaGraph_t                       hipGraph_t
#define cudaGraphExec_t                   hipGraphExec_t
#define cudaStreamBeginCapture            hipStreamBeginCapture
#define cudaStreamEndCapture              hipStreamEndCapture
#define cudaStreamCaptureModeGlobal       hipStreamCaptureModeGlobal
#define cudaGraphInstantiate              hipGraphInstantiate
#define cudaGraphLaunch                   hipGraphLaunch
#define cudaGraphDestroy                  hipGraphDestroy
#define cudaGraphExecDestroy              hipGraphExecDestroy

// --- OpenGL interop (render path only) ---
#define cudaGraphicsResource              hipGraphicsResource
#define cudaGraphicsGLRegisterBuffer      hipGraphicsGLRegisterBuffer
#define cudaGraphicsMapFlagsWriteDiscard  hipGraphicsRegisterFlagsWriteDiscard
#define cudaGraphicsMapResources          hipGraphicsMapResources
#define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
#define cudaGraphicsUnmapResources        hipGraphicsUnmapResources
#define cudaGraphicsUnregisterResource    hipGraphicsUnregisterResource

// --- device intrinsics ---
// CUDA's block-scoped atomics; HIP's plain atomics are correct on shared and
// global memory (the block scope is a relaxation hint, not a correctness
// requirement).
#define atomicAdd_block                   atomicAdd
#define atomicExch_block                  atomicExch

#else
#include <cuda_runtime.h>
#endif
