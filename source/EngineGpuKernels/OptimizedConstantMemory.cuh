#pragma once

#include "EngineInterface/SimulationParameters.h"

/**
 * Alternative implementation that splits SimulationParameters into different
 * memory types based on access frequency and patterns.
 * 
 * This implementation maintains API compatibility while optimizing memory usage.
 */

namespace OptimizedMemory {

// High-frequency parameters kept in constant memory (< 4KB total)
struct CoreParams {
    // Most critical parameters accessed in every kernel call
    float minCellDistance;
    float timestepSize;
    int numLayers;
    
    // Layer positions are heavily accessed by ParameterCalculator
    RealVector2D layerPosition[MAX_LAYERS];  // 20 * 8 bytes = 160 bytes
    
    // Essential cell parameters
    ColorVector<float> normalCellEnergy;  // 7 * 4 bytes = 28 bytes
    
    // Physics constants
    float maxVelocity;
    float cellRadius;
    
    // Total size estimate: ~250 bytes (well within constant memory limits)
};

// Medium-frequency parameters for texture/cached global memory
struct CachedParams {
    // Layer parameters accessed moderately
    LayerParameter<float> layerOpacity;                    // 20 * 4 = 80 bytes
    LayerParameter<RealVector2D> layerCoreRect;           // 20 * 8 = 160 bytes  
    LayerParameter<float> layerFadeoutRadius;             // 20 * 4 = 80 bytes
    LayerParameter<float> layerCoreRadius;                // 20 * 4 = 80 bytes
    
    // Source parameters
    SourceParameter<RealVector2D> sourcePosition;         // 40 * 8 = 320 bytes
    SourceParameter<float> sourceRadiationAngle;          // 40 * 4 = 160 bytes
    
    // Color-indexed parameters  
    ColorVector<float> constructorConnectingCellDistance; // 7 * 4 = 28 bytes
    ColorVector<float> attackerStrength;                 // 7 * 4 = 28 bytes
    ColorVector<float> attackerRadius;                   // 7 * 4 = 28 bytes
    
    // Toggle flags (1 byte each)
    bool advancedAttackerControlToggle;
    bool cellAgeLimiterToggle;
    bool externalEnergyControlToggle;
    bool genomeComplexityMeasurementToggle;
    bool transmitterEnergyDistributionSameCreature;
    bool constructorCompletenessCheck;
    
    // Additional parameters
    int tokenMemorySize;
    float radiationVelocityPerturbation;
    int cellTypeMuscleActivationCountdown;
    
    // Total size estimate: ~1KB (reasonable for cached access)
};

// Memory management interface
class ParameterManager {
private:
    static CoreParams* d_coreParams;           // Constant memory
    static CachedParams* d_cachedParams;       // Global memory with caching
    static SimulationParameters* d_fullParams; // Complete backup in global memory
    
    // Cached access optimization
    __device__ static CachedParams s_cachedParamsLocal;
    __device__ static bool s_cacheValid;
    
public:
    // Initialize memory and copy parameters
    static void initialize(const SimulationParameters& params);
    static void updateParameters(const SimulationParameters& params);
    static void cleanup();
    
    // High-performance parameter access
    __device__ __forceinline__ static float getMinCellDistance() {
        return d_coreParams->minCellDistance;
    }
    
    __device__ __forceinline__ static float getTimestepSize() {
        return d_coreParams->timestepSize;
    }
    
    __device__ __forceinline__ static int getNumLayers() {
        return d_coreParams->numLayers;
    }
    
    __device__ __forceinline__ static RealVector2D getLayerPosition(int layer) {
        return d_coreParams->layerPosition[layer];
    }
    
    __device__ __forceinline__ static float getNormalCellEnergy(int color) {
        return d_coreParams->normalCellEnergy.value[color];
    }
    
    // Cached parameter access (loads once per thread block)
    __device__ static const CachedParams& getCachedParams() {
        if (!s_cacheValid) {
            s_cachedParamsLocal = *d_cachedParams;
            s_cacheValid = true;
        }
        return s_cachedParamsLocal;
    }
    
    // Full parameter access (for rare cases)
    __device__ __forceinline__ static const SimulationParameters& getFullParams() {
        return *d_fullParams;
    }
};

// Optimized access macros for migration
#define GET_CORE_PARAM(param) ParameterManager::get##param()
#define GET_CACHED_PARAM(param) ParameterManager::getCachedParams().param
#define GET_FULL_PARAM(param) ParameterManager::getFullParams().param

// Backward compatibility layer
struct SimulationParametersOptimized {
    // Provide the same interface as SimulationParameters but with optimized access
    
    // High-frequency parameters - direct access to constant memory
    __device__ float getMinCellDistanceValue() const {
        return ParameterManager::getMinCellDistance();
    }
    
    __device__ float getTimestepSizeValue() const {
        return ParameterManager::getTimestepSize();
    }
    
    __device__ int getNumLayers() const {
        return ParameterManager::getNumLayers();
    }
    
    __device__ RealVector2D getLayerPosition(int layer) const {
        return ParameterManager::getLayerPosition(layer);
    }
    
    // Medium-frequency parameters - cached access
    __device__ float getLayerOpacity(int layer) const {
        return ParameterManager::getCachedParams().layerOpacity.layerValues[layer];
    }
    
    __device__ bool getAdvancedAttackerControlToggle() const {
        return ParameterManager::getCachedParams().advancedAttackerControlToggle;
    }
    
    // Low-frequency parameters - direct global memory access
    __device__ float getCopyMutationNeuronData(int color) const {
        return ParameterManager::getFullParams().copyMutationNeuronData.baseValue.value[color];
    }
    
    // For complete backward compatibility during transition
    __device__ SimulationParameters getFullCopy() const {
        // This is expensive and should be avoided - for transition only
        return ParameterManager::getFullParams();
    }
};

} // namespace OptimizedMemory

// Global optimized instance for backward compatibility
extern __device__ OptimizedMemory::SimulationParametersOptimized cudaSimulationParametersOpt;