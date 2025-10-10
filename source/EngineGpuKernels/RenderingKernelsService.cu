#include "RenderingKernelsService.cuh"

#include "BufferData.cuh"
#include "RenderingKernels.cuh"

void _RenderingKernelsService::drawImage(
    SettingsForSimulation const& settings,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    int2 imageSize,
    float zoom,
    SimulationData data,
    BufferData renderingData)
{
    //auto const& gpuSettings = settings.cudaSettings;

    //KERNEL_CALL(cudaBackground, targetImage, imageSize, data.worldSize, zoom, rectUpperLeft, rectLowerRight);
    //KERNEL_CALL_1_1(cudaPrepareFilteringForRendering, data.tempObjects.cells, data.tempObjects.particles);
    //KERNEL_CALL(cudaFilterCellsForRendering, data.worldSize, rectUpperLeft, data.objects.cells, data.tempObjects.cells, imageSize, zoom);
    //KERNEL_CALL(cudaFilterParticlesForRendering, data.worldSize, rectUpperLeft, data.objects.particles, data.tempObjects.particles, imageSize, zoom);
    //KERNEL_CALL(cudaDrawCells, data.timestep, data.worldSize, rectUpperLeft, rectLowerRight, data.tempObjects.cells, targetImage, imageSize, zoom);
    //KERNEL_CALL(cudaDrawParticles, data.worldSize, rectUpperLeft, rectLowerRight, data.tempObjects.particles, targetImage, imageSize, zoom);
    //KERNEL_CALL_1_1(cudaDrawRadiationSources, targetImage, rectUpperLeft, data.worldSize, imageSize, zoom);

    //if (settings.simulationParameters.cellGlowToggle.value) {
    //    int blocks;
    //    int threadsPerBlock;
    //    if (zoom < 4) {
    //        blocks = 2048;
    //        threadsPerBlock = 32;
    //    } else if (zoom < 30) {
    //        blocks = 512;
    //        threadsPerBlock = 16;
    //    } else {
    //        blocks = 32;
    //        threadsPerBlock = 1024;
    //    }
    //    cudaDrawCellGlow<<<blocks, threadsPerBlock>>>(data.worldSize, rectUpperLeft, data.tempObjects.cells, targetImage, imageSize, zoom);
    //}

    //if (settings.simulationParameters.borderlessRendering.value) {
    //    KERNEL_CALL(cudaDrawRepetition, data.worldSize, imageSize, rectUpperLeft, rectLowerRight, targetImage, zoom);
    //}
}

NumRenderObjects _RenderingKernelsService::getNumRenderObjects(SimulationData data)
{
    NumRenderObjects result;
    result.vertices = data.objects.cells.getNumEntries_host() + data.objects.particles.getNumEntries_host();
    
    // Estimate maximum number of line indices (each cell can have up to MAX_CELL_BONDS connections)
    // We count each connection once, so it's approximately numCells * avgConnections
    result.lineIndices = data.objects.cells.getNumEntries_host() * 4;  // Conservative estimate
    return result;
}

void _RenderingKernelsService::extractObjectData(SettingsForSimulation const& settings, SimulationData data, BufferData& renderingData)
{
    auto const& gpuSettings = settings.cudaSettings;
    
    // Extract object data
    CHECK_FOR_CUDA_ERROR(cudaMemset(renderingData.numVertices, 0, sizeof(uint64_t)));
    CHECK_FOR_CUDA_ERROR(cudaMemset(renderingData.numLineIndices, 0, sizeof(uint64_t)));

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.vertexBuffer));
    VertexData* mappedBuffer;
    size_t bufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedBuffer), &bufferSize, renderingData.vertexBuffer));

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.lineIndexBuffer));
    unsigned int* mappedLineIndexBuffer;
    size_t lineIndexBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedLineIndexBuffer), &lineIndexBufferSize, renderingData.lineIndexBuffer));

    // First kernel: Extract vertex data and store cell indices
    KERNEL_CALL(cudaExtractObjectData, data.worldSize, data.objects.cells, data.objects.particles, mappedBuffer, renderingData.numVertices);
    
    // Ensure first kernel completes before second kernel starts
    CHECK_FOR_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Second kernel: Extract line indices from cell connections
    KERNEL_CALL(cudaExtractLineIndices, data.objects.cells, mappedLineIndexBuffer, renderingData.numLineIndices);

    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.lineIndexBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.vertexBuffer));
}
