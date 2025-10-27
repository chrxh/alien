#include "GeometryKernelsService.cuh"

#include "EngineInterface/SettingsForSimulation.h"

#include "CudaGeometryBuffers.cuh"
#include "GeometryKernels.cuh"

_GeometryKernelsService::_GeometryKernelsService()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.acquireMemory(1, _numLineIndices);
    memoryManager.acquireMemory(1, _numTriangleIndices);
    memoryManager.acquireMemory(1, _numSelectedConnectionVertices);
    memoryManager.acquireMemory(1, _numSelectedObjects);
    memoryManager.acquireMemory(1, _numAttackEventVertices);
    memoryManager.acquireMemory(1, _numLocations);
}

_GeometryKernelsService::~_GeometryKernelsService()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.freeMemory(_numLineIndices);
    memoryManager.freeMemory(_numTriangleIndices);
    memoryManager.freeMemory(_numSelectedConnectionVertices);
    memoryManager.freeMemory(_numSelectedObjects);
    memoryManager.freeMemory(_numAttackEventVertices);
    memoryManager.freeMemory(_numLocations);
}

NumRenderObjects _GeometryKernelsService::getNumRenderObjects(SettingsForSimulation const& settings, SimulationData data)
{
    auto const& gpuSettings = settings.cudaSettings;

    NumRenderObjects result;
    result.cells = data.objects.cells.getNumEntries_host()/* + data.objects.particles.getNumEntries_host()*/;
    result.energyParticles = data.objects.particles.getNumEntries_host();
    
    // Count selected objects
    setValueToDevice(_numSelectedObjects, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractSelectedObjectData, data, nullptr, _numSelectedObjects);
    cudaDeviceSynchronize();
    result.selectedObjects = copyToHost(_numSelectedObjects);
    
    setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractLineIndices, data, nullptr, _numLineIndices);
    cudaDeviceSynchronize();
    result.lineIndices = copyToHost(_numLineIndices);

    setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractTriangleIndices, data, nullptr, _numTriangleIndices);
    cudaDeviceSynchronize();
    result.triangleIndices = copyToHost(_numTriangleIndices);

    setValueToDevice(_numSelectedConnectionVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractSelectedConnectionData, data, nullptr, _numSelectedConnectionVertices);
    cudaDeviceSynchronize();
    result.connectionArrowVertices = copyToHost(_numSelectedConnectionVertices);

    setValueToDevice(_numAttackEventVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractAttackEventData, data, nullptr, _numAttackEventVertices);
    cudaDeviceSynchronize();
    result.attackEventVertices = copyToHost(_numAttackEventVertices);

    setValueToDevice(_numLocations, static_cast<uint64_t>(0));
    KERNEL_CALL_1_1(cudaExtractLocationData, data, nullptr, _numLocations);
    cudaDeviceSynchronize();
    result.locations = copyToHost(_numLocations);

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

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.locationBuffer));
    LocationVertexData* mappedLocationBuffer;
    size_t locationBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedLocationBuffer), &locationBufferSize, renderingData.locationBuffer));

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.selectedObjectBuffer));
    SelectedObjectVertexData* mappedSelectedObjectBuffer;
    size_t selectedObjectBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedSelectedObjectBuffer), &selectedObjectBufferSize, renderingData.selectedObjectBuffer));

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.lineIndexBuffer));
    unsigned int* mappedLineIndexBuffer;
    size_t lineIndexBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedLineIndexBuffer), &lineIndexBufferSize, renderingData.lineIndexBuffer));

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.triangleIndexBuffer));
    unsigned int* mappedTriangleIndexBuffer;
    size_t triangleIndexBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedTriangleIndexBuffer), &triangleIndexBufferSize, renderingData.triangleIndexBuffer));

    // First kernel: Extract vertex data (cells at indices 0..numCells-1)
    KERNEL_CALL(cudaExtractCellData, data, mappedBuffer);
    
    // Extract energy particle data
    KERNEL_CALL(cudaExtractEnergyParticleData, data, mappedEnergyParticleBuffer);
    
    // Extract location data
    setValueToDevice(_numLocations, static_cast<uint64_t>(0));
    KERNEL_CALL_1_1(cudaExtractLocationData, data, mappedLocationBuffer, _numLocations);
    
    // Extract selected object data
    setValueToDevice(_numSelectedObjects, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractSelectedObjectData, data, mappedSelectedObjectBuffer, _numSelectedObjects);
    
    // Second kernel: Extract line indices from cell connections
    setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractLineIndices, data, mappedLineIndexBuffer, _numLineIndices);

    // Third kernel: Extract triangle indices from cell connections
    setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractTriangleIndices, data, mappedTriangleIndexBuffer, _numTriangleIndices);

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.selectedConnectionBuffer));
    ConnectionArrowVertexData* mappedSelectedConnectionBuffer;
    size_t selectedConnectionBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedSelectedConnectionBuffer), &selectedConnectionBufferSize, renderingData.selectedConnectionBuffer));

    // Fourth kernel: Extract connection arrow data
    setValueToDevice(_numSelectedConnectionVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractSelectedConnectionData, data, mappedSelectedConnectionBuffer, _numSelectedConnectionVertices);

    CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.attackEventBuffer));
    AttackEventVertexData* mappedAttackEventBuffer;
    size_t attackEventBufferSize;
    CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedAttackEventBuffer), &attackEventBufferSize, renderingData.attackEventBuffer));

    // Fifth kernel: Extract attack event data
    setValueToDevice(_numAttackEventVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractAttackEventData, data, mappedAttackEventBuffer, _numAttackEventVertices);

    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.vertexBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.energyParticleBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.locationBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.selectedObjectBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.lineIndexBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.triangleIndexBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.selectedConnectionBuffer));
    CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.attackEventBuffer));
}
