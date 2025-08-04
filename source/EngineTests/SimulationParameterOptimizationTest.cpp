#include <gtest/gtest.h>
#include <chrono>
#include <cuda_runtime.h>

#include "EngineGpuKernels/OptimizedConstantMemory.cuh"
#include "EngineGpuKernels/OptimizedParameterCalculator.cuh"
#include "EngineGpuKernels/ParameterCalculator.cuh"
#include "EngineInterface/SimulationParameters.h"

/**
 * Performance test comparing original constant memory approach 
 * vs optimized memory management for simulation parameters.
 */

class SimulationParameterPerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test parameters
        _parameters = createTestParameters();
        
        // Initialize CUDA
        cudaSetDevice(0);
        
        // Set up test data
        setupTestData();
    }

    void TearDown() override {
        OptimizedMemory::ParameterManager::cleanup();
        cleanupTestData();
    }

    SimulationParameters createTestParameters() {
        SimulationParameters params;
        
        // Set up realistic parameters
        params.numLayers = MAX_LAYERS;
        params.minCellDistance.value = 0.3f;
        params.timestepSize.value = 1.0f;
        params.maxVelocity.value = 2.0f;
        params.cellRadius.value = 0.25f;
        
        // Initialize layer positions
        for (int i = 0; i < MAX_LAYERS; ++i) {
            params.layerPosition.layerValues[i] = {i * 100.0f, i * 100.0f};
            params.layerOpacity.layerValues[i] = 0.5f;
            params.layerCoreRadius.layerValues[i] = 50.0f + i * 10.0f;
            params.layerFadeoutRadius.layerValues[i] = 100.0f + i * 20.0f;
            params.layerCoreRect.layerValues[i] = {100.0f + i * 10.0f, 100.0f + i * 10.0f};
            params.layerShape.layerValues[i] = (i % 2 == 0) ? LayerShapeType_Circular : LayerShapeType_Rectangular;
        }
        
        // Initialize color parameters
        for (int c = 0; c < MAX_COLORS; ++c) {
            params.normalCellEnergy.value[c] = 100.0f + c * 10.0f;
            params.constructorConnectingCellDistance.value[c] = 2.5f + c * 0.1f;
            params.attackerStrength.value[c] = 0.05f + c * 0.01f;
        }
        
        return params;
    }

    void setupTestData() {
        // Create test simulation data
        // This would normally be initialized from the simulation system
    }

    void cleanupTestData() {
        // Cleanup test resources
    }

    SimulationParameters _parameters;
};

// Kernel for testing original parameter access
__global__ void testOriginalParameterAccess(float* results, int numIterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numIterations) return;
    
    float sum = 0.0f;
    
    // Simulate typical parameter access patterns
    for (int i = 0; i < 1000; ++i) {
        // High frequency accesses
        sum += cudaSimulationParameters.minCellDistance.value;
        sum += cudaSimulationParameters.timestepSize.value;
        
        // Layer position accesses (most frequent)
        for (int layer = 0; layer < cudaSimulationParameters.numLayers; ++layer) {
            sum += cudaSimulationParameters.layerPosition.layerValues[layer].x;
            sum += cudaSimulationParameters.layerPosition.layerValues[layer].y;
        }
        
        // Medium frequency accesses
        for (int layer = 0; layer < min(5, cudaSimulationParameters.numLayers); ++layer) {
            sum += cudaSimulationParameters.layerOpacity.layerValues[layer];
            sum += cudaSimulationParameters.layerCoreRadius.layerValues[layer];
        }
        
        // Color-indexed access
        sum += cudaSimulationParameters.normalCellEnergy.value[idx % MAX_COLORS];
    }
    
    results[idx] = sum;
}

// Kernel for testing optimized parameter access
__global__ void testOptimizedParameterAccess(float* results, int numIterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numIterations) return;
    
    float sum = 0.0f;
    
    // Simulate same access patterns with optimized implementation
    for (int i = 0; i < 1000; ++i) {
        // High frequency accesses - constant memory
        sum += OptimizedMemory::ParameterManager::getMinCellDistance();
        sum += OptimizedMemory::ParameterManager::getTimestepSize();
        
        // Layer position accesses - constant memory  
        int numLayers = OptimizedMemory::ParameterManager::getNumLayers();
        for (int layer = 0; layer < numLayers; ++layer) {
            RealVector2D pos = OptimizedMemory::ParameterManager::getLayerPosition(layer);
            sum += pos.x;
            sum += pos.y;
        }
        
        // Medium frequency accesses - cached memory
        const auto& cached = OptimizedMemory::ParameterManager::getCachedParams();
        for (int layer = 0; layer < min(5, numLayers); ++layer) {
            sum += cached.layerOpacity.layerValues[layer];
            sum += cached.layerCoreRadius.layerValues[layer];
        }
        
        // Color-indexed access - constant memory
        sum += OptimizedMemory::ParameterManager::getNormalCellEnergy(idx % MAX_COLORS);
    }
    
    results[idx] = sum;
}

TEST_F(SimulationParameterPerformanceTest, MemoryUsageComparison) {
    std::cout << "\n=== Memory Usage Comparison ===\n";
    std::cout << "Original SimulationParameters size: " << sizeof(SimulationParameters) << " bytes\n";
    std::cout << "Optimized CoreParams size: " << sizeof(OptimizedMemory::CoreParams) << " bytes\n";
    std::cout << "Optimized CachedParams size: " << sizeof(OptimizedMemory::CachedParams) << " bytes\n";
    
    size_t originalConstantMemory = sizeof(SimulationParameters);
    size_t optimizedConstantMemory = sizeof(OptimizedMemory::CoreParams);
    size_t memoryReduction = originalConstantMemory - optimizedConstantMemory;
    
    std::cout << "Constant memory reduction: " << memoryReduction << " bytes (" 
              << (100.0 * memoryReduction / originalConstantMemory) << "%)\n";
    
    // Verify we're within constant memory limits
    EXPECT_LT(optimizedConstantMemory, 4096); // Keep core parameters under 4KB
    EXPECT_GT(memoryReduction, originalConstantMemory * 0.9); // At least 90% reduction
}

TEST_F(SimulationParameterPerformanceTest, AccessPatternPerformance) {
    const int numThreads = 1024;
    const int numBlocks = 32;
    const int totalThreads = numThreads * numBlocks;
    
    // Allocate device memory for results
    float* d_originalResults;
    float* d_optimizedResults;
    float* h_originalResults = new float[totalThreads];
    float* h_optimizedResults = new float[totalThreads];
    
    cudaMalloc(&d_originalResults, totalThreads * sizeof(float));
    cudaMalloc(&d_optimizedResults, totalThreads * sizeof(float));
    
    // Initialize both memory systems
    OptimizedMemory::ParameterManager::initialize(_parameters);
    
    // Copy original parameters to constant memory (simulate current implementation)
    cudaMemcpyToSymbol(cudaSimulationParameters, &_parameters, sizeof(SimulationParameters));
    
    // Warm up GPU
    testOriginalParameterAccess<<<numBlocks, numThreads>>>(d_originalResults, totalThreads);
    cudaDeviceSynchronize();
    
    // Test original implementation
    auto start = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < 10; ++run) {
        testOriginalParameterAccess<<<numBlocks, numThreads>>>(d_originalResults, totalThreads);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto originalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Test optimized implementation
    start = std::chrono::high_resolution_clock::now();
    for (int run = 0; run < 10; ++run) {
        testOptimizedParameterAccess<<<numBlocks, numThreads>>>(d_optimizedResults, totalThreads);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto optimizedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy results back
    cudaMemcpy(h_originalResults, d_originalResults, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_optimizedResults, d_optimizedResults, totalThreads * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results are equivalent (within floating point precision)
    for (int i = 0; i < totalThreads; ++i) {
        EXPECT_NEAR(h_originalResults[i], h_optimizedResults[i], 1e-3);
    }
    
    std::cout << "\n=== Performance Comparison ===\n";
    std::cout << "Original implementation: " << originalTime.count() << " microseconds\n";
    std::cout << "Optimized implementation: " << optimizedTime.count() << " microseconds\n";
    
    double speedup = static_cast<double>(originalTime.count()) / optimizedTime.count();
    std::cout << "Performance ratio: " << speedup << "x\n";
    
    if (speedup >= 1.0) {
        std::cout << "✓ Optimized implementation is " << speedup << "x faster\n";
    } else {
        std::cout << "⚠ Optimized implementation is " << (1.0/speedup) << "x slower\n";
    }
    
    // Performance should be maintained or improved
    EXPECT_GE(speedup, 0.95); // Allow up to 5% slowdown due to measurement variance
    
    // Cleanup
    cudaFree(d_originalResults);
    cudaFree(d_optimizedResults);
    delete[] h_originalResults;
    delete[] h_optimizedResults;
}

TEST_F(SimulationParameterPerformanceTest, ParameterCalculatorPerformance) {
    // This test would compare OptimizedParameterCalculator vs original ParameterCalculator
    // using realistic simulation workloads
    
    std::cout << "\n=== ParameterCalculator Performance Test ===\n";
    std::cout << "This test would benchmark the optimized ParameterCalculator\n";
    std::cout << "against the original implementation using realistic layer calculations.\n";
    
    // The test would:
    // 1. Create test simulation data
    // 2. Run both parameter calculators on identical workloads
    // 3. Measure performance difference
    // 4. Verify identical results
    
    SUCCEED(); // Placeholder for actual implementation
}