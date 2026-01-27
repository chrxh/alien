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
// Using 16x16x16 matrix dimensions with half precision input and float accumulator
// We batch 2 cells together: each 8x8 weight matrix is placed in block-diagonal pattern
// within a 16x16 matrix A. Input signals are placed on the diagonal of matrix B.
// After C = A × B, summing each row gives the dot product results.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int CELLS_PER_WMMA_BATCH = 2;  // 2 cells processed in parallel

class NeuronProcessor
{
public:
    // needs to be called with 32 threads (one warp for WMMA operations)
    __inline__ __device__ static void process(SimulationData& data, SimulationStatistics& statistics);

private:
    // Process a batch of up to 2 cells using tensor cores (WMMA)
    __inline__ __device__ static void processCellBatchWMMA(
        SimulationData& data,
        SimulationStatistics& statistics,
        Object* cells[CELLS_PER_WMMA_BATCH],
        int numCells);

    __inline__ __device__ static float applyActivationFunction(ActivationFunction activationFunction, float x);
};

/************************************************************************/
/* Implementation                                                       */
/************************************************************************/

__device__ __inline__ void NeuronProcessor::process(SimulationData& data, SimulationStatistics& statistics)
{
    auto block = cg::this_thread_block();
    auto& objects = data.entities.objects;
    auto partition = calcBlockPartition(objects.getNumEntries());

    // Shared memory for cell batch coordination (only thread 0 manages batching)
    __shared__ Object* cellBatch[CELLS_PER_WMMA_BATCH];
    __shared__ int batchCount;

    if (block.thread_rank() == 0) {
        batchCount = 0;
    }
    block.sync();

    for (int i = partition.startIndex; i <= partition.endIndex; ++i) {
        auto& object = objects.at(i);
        if (object->type == ObjectType_Cell && object->typeData.cell.signalState == SignalState_Active) {
            // Thread 0 manages the batch collection
            if (block.thread_rank() == 0) {
                cellBatch[batchCount++] = object;
            }
            block.sync();

            // All threads check if batch is full and process together
            if (batchCount == CELLS_PER_WMMA_BATCH) {
                processCellBatchWMMA(data, statistics, cellBatch, batchCount);
                if (block.thread_rank() == 0) {
                    batchCount = 0;
                }
                block.sync();
            }
        }
    }

    // Process remaining cells in partial batch
    block.sync();
    if (batchCount > 0) {
        processCellBatchWMMA(data, statistics, cellBatch, batchCount);
    }
}

__inline__ __device__ void NeuronProcessor::processCellBatchWMMA(
    SimulationData& data,
    SimulationStatistics& statistics,
    Object* cells[CELLS_PER_WMMA_BATCH],
    int numCells)
{
    auto block = cg::this_thread_block();
    const int laneId = block.thread_rank();

    // Shared memory for half-precision matrices (16x16 layout)
    // Matrix A: 16x16 block diagonal weights
    //   - Cell 0 weights (8x8) in top-left quadrant (rows 0-7, cols 0-7)
    //   - Cell 1 weights (8x8) in bottom-right quadrant (rows 8-15, cols 8-15)
    // Matrix B: 16x16 diagonal matrix with inputs on diagonal
    //   - Cell 0 inputs on diagonal positions 0-7
    //   - Cell 1 inputs on diagonal positions 8-15
    __shared__ __align__(32) half sharedA[WMMA_M * WMMA_K];  // Weights (16x16)
    __shared__ __align__(32) half sharedB[WMMA_K * WMMA_N];  // Diagonal inputs (16x16)
    __shared__ __align__(32) float sharedC[WMMA_M * WMMA_N]; // Outputs (16x16)

    // Initialize shared memory to zero (parallel across 32 threads)
    for (int i = laneId; i < WMMA_M * WMMA_K; i += 32) {
        sharedA[i] = __float2half(0.0f);
    }
    for (int i = laneId; i < WMMA_K * WMMA_N; i += 32) {
        sharedB[i] = __float2half(0.0f);
    }
    block.sync();

    // Load weight matrices from cells into block-diagonal pattern
    for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
        auto* nn = cells[cellIdx]->typeData.cell.neuralNetwork;
        auto& signal = cells[cellIdx]->typeData.cell.signal;

        // Cell 0: rows 0-7, cols 0-7 | Cell 1: rows 8-15, cols 8-15
        int blockOffset = cellIdx * MAX_CHANNELS;

        // Load 8x8 weight matrix into diagonal block
        for (int elem = laneId; elem < MAX_CHANNELS * MAX_CHANNELS; elem += 32) {
            int r = elem / MAX_CHANNELS;
            int c = elem % MAX_CHANNELS;
            int globalRow = blockOffset + r;
            int globalCol = blockOffset + c;
            sharedA[globalRow * WMMA_K + globalCol] = __float2half(nn->weights[elem]);
        }

        // Load input signals onto the diagonal of B matrix
        // B[k][k] = input[k] for the appropriate cell's channels
        for (int ch = laneId; ch < MAX_CHANNELS; ch += 32) {
            int diagIdx = blockOffset + ch;
            sharedB[diagIdx * WMMA_N + diagIdx] = __float2half(signal.channels[ch]);
        }
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

    // Perform tensor core matrix multiply-accumulate: C = A * B + C
    wmma::mma_sync(fragC, fragA, fragB, fragC);

    // Store result back to shared memory
    wmma::store_matrix_sync(sharedC, fragC, WMMA_N, wmma::mem_row_major);
    block.sync();

    // Extract results: sum each row within the cell's block to get the dot product result
    // Then apply biases and activation functions
    for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
        auto* nn = cells[cellIdx]->typeData.cell.neuralNetwork;
        auto& signal = cells[cellIdx]->typeData.cell.signal;

        int blockOffset = cellIdx * MAX_CHANNELS;

        // Each thread processes some output channels
        for (int ch = laneId; ch < MAX_CHANNELS; ch += 32) {
            // Sum across the 8 relevant columns for this cell's dot product
            float sum = 0.0f;
            int outputRow = blockOffset + ch;
            for (int k = 0; k < MAX_CHANNELS; ++k) {
                sum += sharedC[outputRow * WMMA_N + blockOffset + k];
            }

            // Add bias
            sum += nn->biases[ch];

            // Apply activation function and clamp
            float output = applyActivationFunction(nn->activationFunctions[ch], sum);
            output = max(-2.0f, min(2.0f, output));

            signal.channels[ch] = output;
        }
    }
    block.sync();

    // Update statistics (only thread 0)
    if (laneId == 0) {
        for (int cellIdx = 0; cellIdx < numCells; ++cellIdx) {
            statistics.incNumNeuronActivities(cells[cellIdx]->color);
        }
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
