#include "GeometryKernelsService.cuh"

#include "EngineInterface/SettingsForSimulation.h"

#include "CudaGeometryBuffers.cuh"
#include "GeometryKernels.cuh"

_GeometryKernelsService::_GeometryKernelsService()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.acquireMemory(1, _numLineIndices);
    memoryManager.acquireMemory(1, _numTriangleIndices);
}

_GeometryKernelsService::~_GeometryKernelsService()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.freeMemory(_numLineIndices);
    memoryManager.freeMemory(_numTriangleIndices);
}

void _GeometryKernelsService::drawImage(
    SettingsForSimulation const& settings,
    float2 rectUpperLeft,
    float2 rectLowerRight,
    int2 imageSize,
    float zoom,
    SimulationData data,
    CudaGeometryBuffers renderingData)
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

NumRenderObjects _GeometryKernelsService::getNumRenderObjects(SettingsForSimulation const& settings, SimulationData data)
{
    auto const& gpuSettings = settings.cudaSettings;

    NumRenderObjects result;
    result.vertices = data.objects.cells.getNumEntries_host()/* + data.objects.particles.getNumEntries_host()*/;
    result.energyParticles = data.objects.particles.getNumEntries_host();
    
    setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractLineIndices, data, nullptr, _numLineIndices);
    cudaDeviceSynchronize();
    result.lineIndices = copyToHost(_numLineIndices);

    setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractTriangleIndices, data, nullptr, _numTriangleIndices);
    cudaDeviceSynchronize();
    result.triangleIndices = copyToHost(_numTriangleIndices);

    return result;
}

void _GeometryKernelsService::extractObjectData(SettingsForSimulation const& settings, SimulationData data, CudaGeometryBuffers& renderingData)
{
    auto const& gpuSettings = settings.cudaSettings;
    
    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.vertexBuffer));
    CellVertexData* mappedBuffer;
    size_t bufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedBuffer), &bufferSize, renderingData.vertexBuffer));

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.energyParticleBuffer));
    EnergyParticleVertexData* mappedEnergyParticleBuffer;
    size_t energyParticleBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedEnergyParticleBuffer), &energyParticleBufferSize, renderingData.energyParticleBuffer));

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.lineIndexBuffer));
    unsigned int* mappedLineIndexBuffer;
    size_t lineIndexBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedLineIndexBuffer), &lineIndexBufferSize, renderingData.lineIndexBuffer));

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.triangleIndexBuffer));
    unsigned int* mappedTriangleIndexBuffer;
    size_t triangleIndexBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedTriangleIndexBuffer), &triangleIndexBufferSize, renderingData.triangleIndexBuffer));

    // First kernel: Extract vertex data (cells at indices 0..numCells-1)
    KERNEL_CALL(cudaExtractObjectData, data, mappedBuffer);
    
    // Extract energy particle data
    KERNEL_CALL(cudaExtractEnergyParticleData, data, mappedEnergyParticleBuffer);
    
    // Second kernel: Extract line indices from cell connections
    setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractLineIndices, data, mappedLineIndexBuffer, _numLineIndices);

    // Third kernel: Extract triangle indices from cell connections
    setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractTriangleIndices, data, mappedTriangleIndexBuffer, _numTriangleIndices);

    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.vertexBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.energyParticleBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.lineIndexBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.triangleIndexBuffer));
}
