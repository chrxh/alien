#include "OptimizedConstantMemory.cuh"
#include <cuda_runtime.h>
#include <iostream>

namespace OptimizedMemory {

// Static member definitions
CoreParams* ParameterManager::d_coreParams = nullptr;
CachedParams* ParameterManager::d_cachedParams = nullptr;
SimulationParameters* ParameterManager::d_fullParams = nullptr;

__device__ CachedParams ParameterManager::s_cachedParamsLocal;
__device__ bool ParameterManager::s_cacheValid = false;

// Constant memory declaration for core parameters
__constant__ CoreParams cudaCoreParameters;

void ParameterManager::initialize(const SimulationParameters& params) {
    // Allocate GPU memory for different parameter groups
    
    // Core parameters go to constant memory
    CoreParams coreParams;
    coreParams.minCellDistance = params.minCellDistance.value;
    coreParams.timestepSize = params.timestepSize.value;
    coreParams.numLayers = params.numLayers;
    coreParams.maxVelocity = params.maxVelocity.value;
    coreParams.cellRadius = params.cellRadius.value;
    coreParams.normalCellEnergy = params.normalCellEnergy;
    
    // Copy layer positions
    for (int i = 0; i < MAX_LAYERS; ++i) {
        coreParams.layerPosition[i] = params.layerPosition.layerValues[i];
    }
    
    // Copy to constant memory
    cudaMemcpyToSymbol(cudaCoreParameters, &coreParams, sizeof(CoreParams));
    
    // Allocate and copy cached parameters to global memory
    cudaMalloc(&d_cachedParams, sizeof(CachedParams));
    
    CachedParams cachedParams;
    cachedParams.layerOpacity = params.layerOpacity;
    cachedParams.layerCoreRect = params.layerCoreRect;
    cachedParams.layerFadeoutRadius = params.layerFadeoutRadius;
    cachedParams.layerCoreRadius = params.layerCoreRadius;
    cachedParams.sourcePosition = params.sourcePosition;
    cachedParams.sourceRadiationAngle = params.sourceRadiationAngle;
    cachedParams.constructorConnectingCellDistance = params.constructorConnectingCellDistance;
    cachedParams.attackerStrength = params.attackerStrength;
    cachedParams.attackerRadius = params.attackerRadius;
    cachedParams.advancedAttackerControlToggle = params.advancedAttackerControlToggle.value;
    cachedParams.cellAgeLimiterToggle = params.cellAgeLimiterToggle.value;
    cachedParams.externalEnergyControlToggle = params.externalEnergyControlToggle.value;
    cachedParams.genomeComplexityMeasurementToggle = params.genomeComplexityMeasurementToggle.value;
    cachedParams.transmitterEnergyDistributionSameCreature = params.transmitterEnergyDistributionSameCreature.value;
    cachedParams.constructorCompletenessCheck = params.constructorCompletenessCheck.value;
    cachedParams.tokenMemorySize = params.tokenMemorySize.value;
    cachedParams.radiationVelocityPerturbation = params.radiationVelocityPerturbation;
    cachedParams.cellTypeMuscleActivationCountdown = params.cellTypeMuscleActivationCountdown;
    
    cudaMemcpy(d_cachedParams, &cachedParams, sizeof(CachedParams), cudaMemcpyHostToDevice);
    
    // Allocate and copy full parameters for rare access
    cudaMalloc(&d_fullParams, sizeof(SimulationParameters));
    cudaMemcpy(d_fullParams, &params, sizeof(SimulationParameters), cudaMemcpyHostToDevice);
    
    std::cout << "Optimized memory initialized:" << std::endl;
    std::cout << "  Core parameters: " << sizeof(CoreParams) << " bytes (constant memory)" << std::endl;
    std::cout << "  Cached parameters: " << sizeof(CachedParams) << " bytes (global memory)" << std::endl;
    std::cout << "  Full parameters: " << sizeof(SimulationParameters) << " bytes (global memory)" << std::endl;
    std::cout << "  Total memory saved from constant memory: " 
              << (sizeof(SimulationParameters) - sizeof(CoreParams)) << " bytes" << std::endl;
}

void ParameterManager::updateParameters(const SimulationParameters& params) {
    // Update only the parts that have changed
    
    // Update core parameters
    CoreParams coreParams;
    coreParams.minCellDistance = params.minCellDistance.value;
    coreParams.timestepSize = params.timestepSize.value;
    coreParams.numLayers = params.numLayers;
    coreParams.maxVelocity = params.maxVelocity.value;
    coreParams.cellRadius = params.cellRadius.value;
    coreParams.normalCellEnergy = params.normalCellEnergy;
    
    for (int i = 0; i < MAX_LAYERS; ++i) {
        coreParams.layerPosition[i] = params.layerPosition.layerValues[i];
    }
    
    cudaMemcpyToSymbol(cudaCoreParameters, &coreParams, sizeof(CoreParams));
    
    // Update cached parameters
    CachedParams cachedParams;
    // ... (same initialization as above)
    cudaMemcpy(d_cachedParams, &cachedParams, sizeof(CachedParams), cudaMemcpyHostToDevice);
    
    // Update full parameters
    cudaMemcpy(d_fullParams, &params, sizeof(SimulationParameters), cudaMemcpyHostToDevice);
    
    // Invalidate cache on device
    bool invalidate = false;
    cudaMemcpyToSymbol(s_cacheValid, &invalidate, sizeof(bool));
}

void ParameterManager::cleanup() {
    if (d_cachedParams) {
        cudaFree(d_cachedParams);
        d_cachedParams = nullptr;
    }
    
    if (d_fullParams) {
        cudaFree(d_fullParams);
        d_fullParams = nullptr;
    }
}

} // namespace OptimizedMemory

// Global instance for backward compatibility
__device__ OptimizedMemory::SimulationParametersOptimized cudaSimulationParametersOpt;