#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <mma.h>

#include "sm_60_atomic_functions.h"

#include "SignalProcessor.cuh"
#include "SimulationData.cuh"

namespace cg = cooperative_groups;
using namespace nvcuda;

// WMMA configuration for tensor core operations (requires compute capability 7.0+)
// With MAX_CHANNELS = 16, we use 16x16x16 matrix dimensions directly - optimal for tensor cores
// Each cell is processed with a single WMMA operation: C = weights × diag(inputs)
// Row sums of C give the dot product results (weights × inputs)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

class NeuronProcessor
{
public:
    // needs to be called with 32 threads (one warp for WMMA operations)
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    // Process a single cell using tensor cores (WMMA) - optimal for 16x16 matrix
    __inline__ __device__ static void processCellWMMA(SimulationData& data, SimulationStatistics& statistics, Object* cell);

    __inline__ __device__ static float applyActivationFunction(ActivationFunction activationFunction, float x);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuronProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto& objects = data.entities.objects;
    auto partition = calcBlockPartition(objects.getNumEntries());

    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto& object = objects.at(i);
        if (object->type == ObjectType_Cell && object->typeData.cell.signalState == SignalState_Active) {
            processCellWMMA(data, statistics, object);
        }
    }
}

__inline__ __device__ void NeuronProcessor::processCellWMMA(SimulationData& data, SimulationStatistics& statistics, Object* cell)
{
    auto block = cg::this_thread_block();
    const int laneId = block.thread_rank();

    auto* nn = cell->typeData.cell.neuralNetwork;
    auto& signal = cell->typeData.cell.signal;

    // Shared memory for half-precision matrices (16x16 layout)
    // Matrix A: 16x16 weight matrix (direct fit - no padding needed!)
    // Matrix B: 16x16 diagonal matrix with inputs on diagonal
    // Matrix C: 16x16 output matrix
    __shared__ __align__(32) half sharedA[WMMA_M * WMMA_K];  // Weights (16x16)
    __shared__ __align__(32) half sharedB[WMMA_K * WMMA_N];  // Diagonal inputs (16x16)
    __shared__ __align__(32) float sharedC[WMMA_M * WMMA_N]; // Outputs (16x16)

    // Initialize B matrix to zero (only diagonal elements will be set to inputs)
    // Note: A will be fully overwritten below, C is initialized by fill_fragment
    for (int i = laneId; i < WMMA_K * WMMA_N; i += 32) {
        sharedB[i] = __float2half(0.0f);
    }
    block.sync();

    // Load 16x16 weight matrix directly - perfect fit for tensor cores!
    // Each thread loads 8 elements (256 total / 32 threads)
    for (int elem = laneId; elem < MAX_CHANNELS * MAX_CHANNELS; elem += 32) {
        sharedA[elem] = __float2half(nn->weights[elem]);
    }

    // Load input signals onto the diagonal of B matrix
    // B[k][k] = input[k], all other elements are 0
    // This makes C = A × B equivalent to each column k of C being A[:,k] × input[k]
    // Summing row i gives: sum_k(A[i][k] × input[k]) = dot(weights[i], input)
    for (int ch = laneId; ch < MAX_CHANNELS; ch += 32) {
        sharedB[ch * WMMA_N + ch] = __float2half(signal.channels[ch]);
    }
    block.sync();

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;

    // Initialize accumulator to zero
    wmma::fill_fragment(fragC, 0.0f);

    // Load matrices into fragments
    wmma::load_matrix_sync(fragA, sharedA, WMMA_K);
    wmma::load_matrix_sync(fragB, sharedB, WMMA_N);

    // Perform tensor core matrix multiply-accumulate: C = A × B
    wmma::mma_sync(fragC, fragA, fragB, fragC);

    // Store result back to shared memory
    wmma::store_matrix_sync(sharedC, fragC, WMMA_N, wmma::mem_row_major);
    block.sync();

    // Extract results: sum each row to get the dot product result
    // Then apply biases and activation functions
    // Each thread processes some output channels (16 channels / 32 threads = some threads do 1, some do 0)
    for (int ch = laneId; ch < MAX_CHANNELS; ch += 32) {
        // Sum across all 16 columns for this row's dot product
        float sum = 0.0f;
        for (int k = 0; k < MAX_CHANNELS; ++k) {
            sum += sharedC[ch * WMMA_N + k];
        }

        // Add bias
        sum += nn->biases[ch];

        // Apply activation function and clamp
        float output = applyActivationFunction(nn->activationFunctions[ch], sum);
        output = max(-2.0f, min(2.0f, output));

        signal.channels[ch] = output;
    }
    block.sync();

    // Update statistics (only thread 0)
    if (laneId == 0) {
        statistics.incNumNeuronActivities(cell->color);
    }
}

__inline__ __device__ float NeuronProcessor::applyActivationFunction(ActivationFunction activationFunction, float x)
{
    switch (activationFunction) {
    case ActivationFunction_Sigmoid:
        return 2.0f / (1.0f + __expf(-x)) - 1.0f;
    case ActivationFunction_BinaryStep:
        return x >= NEAR_ZERO ? 1.0f : 0.0f;
    case ActivationFunction_Identity:
        return x;
    case ActivationFunction_Abs:
        return abs(x);
    case ActivationFunction_Gaussian:
        return __expf(-2 * x * x);
    }
    return 0;
}
