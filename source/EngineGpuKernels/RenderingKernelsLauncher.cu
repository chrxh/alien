#include "RenderingKernelsLauncher.cuh"

#include "RenderingData.cuh"
#include "RenderingKernels.cuh"

void _RenderingKernelsLauncher::drawImage(
    Settings const& settings,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    int2 imageSize,
    float zoom,
    SimulationData data,
    RenderingData renderingData)
{
    uint64_t* targetImage = renderingData.imageData;
    auto const& gpuSettings = settings.gpuSettings;

    KERNEL_CALL(cudaDrawBackground, targetImage, imageSize, data.worldSize, zoom, rectUpperLeft, rectLowerRight);
    KERNEL_CALL_1_1(cudaPrepareFilteringForRendering, data.tempObjects.cellPointers, data.tempObjects.particlePointers);
    KERNEL_CALL(cudaFilterCellsForRendering, data.worldSize, rectUpperLeft, data.objects.cellPointers, data.tempObjects.cellPointers, imageSize, zoom);
    KERNEL_CALL(cudaFilterParticlesForRendering, data.worldSize, rectUpperLeft, data.objects.particlePointers, data.tempObjects.particlePointers, imageSize, zoom);
    KERNEL_CALL(cudaDrawCells, data.timestep, data.worldSize, rectUpperLeft, rectLowerRight, data.tempObjects.cellPointers, targetImage, imageSize, zoom);
    KERNEL_CALL(cudaDrawParticles, data.worldSize, rectUpperLeft, rectLowerRight, data.tempObjects.particlePointers, targetImage, imageSize, zoom);
    KERNEL_CALL_1_1(cudaDrawRadiationSources, targetImage, rectUpperLeft, data.worldSize, imageSize, zoom);

    if (settings.simulationParameters.features.cellGlow) {
        cudaDrawCellGlow<<<512, 128>>>(data.worldSize, rectUpperLeft, data.tempObjects.cellPointers, targetImage, imageSize, zoom);
    }

    if (settings.simulationParameters.borderlessRendering) {
        KERNEL_CALL(cudaDrawRepetition, data.worldSize, imageSize, rectUpperLeft, rectLowerRight, targetImage, zoom);
    }
}
