#include "RenderingKernelsLauncher.cuh"

#include "RenderingData.cuh"
#include "RenderingKernels.cuh"

void _RenderingKernelsLauncher::drawImage(
    GpuSettings const& gpuSettings,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    int2 imageSize,
    float zoom,
    SimulationData data,
    RenderingData renderingData)
{
    uint64_t* targetImage = renderingData.imageData;

    KERNEL_CALL(cudaDrawBackground, targetImage, imageSize, data.size, zoom, rectUpperLeft, rectLowerRight);
    KERNEL_CALL(cudaDrawCells, data.size, rectUpperLeft, rectLowerRight, data.entities.cellPointers, targetImage, imageSize, zoom);
    KERNEL_CALL(cudaDrawTokens, data.size, rectUpperLeft, rectLowerRight, data.entities.tokenPointers, targetImage, imageSize, zoom);
    KERNEL_CALL(cudaDrawParticles, data.size, rectUpperLeft, rectLowerRight, data.entities.particlePointers, targetImage, imageSize, zoom);
    KERNEL_CALL(cudaDrawFlowCenters, targetImage, rectUpperLeft, imageSize, zoom);
}
