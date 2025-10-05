#include "RenderingKernelsService.cuh"

#include "RenderingData.cuh"
#include "RenderingKernels.cuh"

void _RenderingKernelsService::drawImage(
    SettingsForSimulation const& settings,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    int2 imageSize,
    float zoom,
    SimulationData data,
    RenderingData renderingData)
{
    uint64_t* targetImage = renderingData.imageData;
    auto const& gpuSettings = settings.cudaSettings;

    KERNEL_CALL(cudaBackground, targetImage, imageSize, data.worldSize, zoom, rectUpperLeft, rectLowerRight);
    KERNEL_CALL_1_1(cudaPrepareFilteringForRendering, data.tempObjects.cells, data.tempObjects.particles);
    KERNEL_CALL(cudaFilterCellsForRendering, data.worldSize, rectUpperLeft, data.objects.cells, data.tempObjects.cells, imageSize, zoom);
    KERNEL_CALL(cudaFilterParticlesForRendering, data.worldSize, rectUpperLeft, data.objects.particles, data.tempObjects.particles, imageSize, zoom);
    KERNEL_CALL(cudaDrawCells, data.timestep, data.worldSize, rectUpperLeft, rectLowerRight, data.tempObjects.cells, targetImage, imageSize, zoom);
    KERNEL_CALL(cudaDrawParticles, data.worldSize, rectUpperLeft, rectLowerRight, data.tempObjects.particles, targetImage, imageSize, zoom);
    KERNEL_CALL_1_1(cudaDrawRadiationSources, targetImage, rectUpperLeft, data.worldSize, imageSize, zoom);

    if (settings.simulationParameters.cellGlowToggle.value) {
        int blocks;
        int threadsPerBlock;
        if (zoom < 4) {
            blocks = 2048;
            threadsPerBlock = 32;
        } else if (zoom < 30) {
            blocks = 512;
            threadsPerBlock = 16;
        } else {
            blocks = 32;
            threadsPerBlock = 1024;
        }
        cudaDrawCellGlow<<<blocks, threadsPerBlock>>>(data.worldSize, rectUpperLeft, data.tempObjects.cells, targetImage, imageSize, zoom);
    }

    if (settings.simulationParameters.borderlessRendering.value) {
        KERNEL_CALL(cudaDrawRepetition, data.worldSize, imageSize, rectUpperLeft, rectLowerRight, targetImage, zoom);
    }
}

void _RenderingKernelsService::extractObjectData(SettingsForSimulation const& settings, SimulationData data, RenderingData& renderingData)
{
    auto const& gpuSettings = settings.cudaSettings;
    
    // Resize buffer if necessary
    uint64_t maxObjects = data.objects.cells.getNumEntries_host() + data.objects.particles.getNumEntries_host();
    renderingData.resizeObjectBufferIfNecessary(maxObjects);

    // Extract object data
    CHECK_FOR_CUDA_ERROR(cudaMemset(renderingData.numVertices, 0, sizeof(uint64_t)));
    KERNEL_CALL(cudaExtractObjectData, data.worldSize, data.objects.cells, data.objects.particles, renderingData.vertices, renderingData.numVertices);
}
