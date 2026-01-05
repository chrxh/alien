#include <Base/GlobalSettings.h>
#include <EngineInterface/SettingsForSimulation.h>

#include "CudaGeometryBuffers.cuh"
#include "GeometryKernels.cuh"
#include "GeometryKernelsService.cuh"

void GeometryKernelsService::init()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.acquireMemory(1, _numLineIndices);
    memoryManager.acquireMemory(1, _numTriangleIndices);
    memoryManager.acquireMemory(1, _numSelectedConnectionVertices);
    memoryManager.acquireMemory(1, _numSelectedObjects);
    memoryManager.acquireMemory(1, _numAttackEventVertices);
    memoryManager.acquireMemory(1, _numDetonationEventVertices);
    memoryManager.acquireMemory(1, _numLocations);
}

void GeometryKernelsService::shutdown()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.freeMemory(_numLineIndices);
    memoryManager.freeMemory(_numTriangleIndices);
    memoryManager.freeMemory(_numSelectedConnectionVertices);
    memoryManager.freeMemory(_numSelectedObjects);
    memoryManager.freeMemory(_numAttackEventVertices);
    memoryManager.freeMemory(_numDetonationEventVertices);
    memoryManager.freeMemory(_numLocations);
}

void GeometryKernelsService::correctPositionsForRendering(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect)
{
    auto const& gpuSettings = settings.cudaSettings;
    float2 const visibleTopLeft{visibleWorldRect.topLeft.x, visibleWorldRect.topLeft.y};

    KERNEL_CALL(cudaCorrectPositionsForRendering, data, visibleTopLeft);
}

void GeometryKernelsService::restorePositions(SettingsForSimulation const& settings, SimulationData data)
{
    auto const& gpuSettings = settings.cudaSettings;

    KERNEL_CALL(cudaCorrectPositionsForRendering, data, float2{0, 0});
}

NumRenderObjects GeometryKernelsService::getNumRenderObjects(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect)
{
    auto const& gpuSettings = settings.cudaSettings;
    float2 const visibleTopLeft{visibleWorldRect.topLeft.x, visibleWorldRect.topLeft.y};

    NumRenderObjects result;
    result.cells = data.objects.cells.getNumEntries_host();
    result.energyParticles = data.objects.particles.getNumEntries_host();

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

    setValueToDevice(_numDetonationEventVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractDetonationEventData, data, nullptr, _numDetonationEventVertices);
    cudaDeviceSynchronize();
    result.detonationEventVertices = copyToHost(_numDetonationEventVertices);

    setValueToDevice(_numLocations, static_cast<uint64_t>(0));
    KERNEL_CALL_1_1(cudaExtractLocationData, data, nullptr, _numLocations, visibleTopLeft);
    cudaDeviceSynchronize();
    result.locations = copyToHost(_numLocations);

    return result;
}

void GeometryKernelsService::extractObjectData(
    SettingsForSimulation const& settings,
    SimulationData data,
    CudaGeometryBuffers& renderingData,
    RealRect const& visibleWorldRect,
    bool useInterop)
{
    auto const& gpuSettings = settings.cudaSettings;
    float2 const visibleTopLeft{visibleWorldRect.topLeft.x, visibleWorldRect.topLeft.y};

    if (useInterop) {
        // Interop mode: use CUDA-OpenGL interoperability
        CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.vertexBuffer));
        CellVertexData* mappedCellBuffer;
        size_t bufferSize;
        CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedCellBuffer), &bufferSize, renderingData.vertexBuffer));
        KERNEL_CALL(cudaExtractCellData, data, mappedCellBuffer);
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.vertexBuffer));

        CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.energyParticleBuffer));
        EnergyParticleVertexData* mappedEnergyParticleBuffer;
        size_t energyParticleBufferSize;
        CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedEnergyParticleBuffer), &energyParticleBufferSize, renderingData.energyParticleBuffer));
        KERNEL_CALL(cudaExtractEnergyParticleData, data, mappedEnergyParticleBuffer);
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.energyParticleBuffer));

        CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.locationBuffer));
        LocationVertexData* mappedLocationBuffer;
        size_t locationBufferSize;
        CHECK_FOR_CUDA_ERROR(
            cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedLocationBuffer), &locationBufferSize, renderingData.locationBuffer));
        setValueToDevice(_numLocations, static_cast<uint64_t>(0));
        KERNEL_CALL_1_1(cudaExtractLocationData, data, mappedLocationBuffer, _numLocations, visibleTopLeft);
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.locationBuffer));

        CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.selectedObjectBuffer));
        SelectedObjectVertexData* mappedSelectedObjectBuffer;
        size_t selectedObjectBufferSize;
        CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedSelectedObjectBuffer), &selectedObjectBufferSize, renderingData.selectedObjectBuffer));
        setValueToDevice(_numSelectedObjects, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedObjectData, data, mappedSelectedObjectBuffer, _numSelectedObjects);
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.selectedObjectBuffer));

        CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.lineIndexBuffer));
        unsigned int* mappedLineIndexBuffer;
        size_t lineIndexBufferSize;
        CHECK_FOR_CUDA_ERROR(
            cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedLineIndexBuffer), &lineIndexBufferSize, renderingData.lineIndexBuffer));
        setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractLineIndices, data, mappedLineIndexBuffer, _numLineIndices);
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.lineIndexBuffer));

        CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.triangleIndexBuffer));
        unsigned int* mappedTriangleIndexBuffer;
        size_t triangleIndexBufferSize;
        CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedTriangleIndexBuffer), &triangleIndexBufferSize, renderingData.triangleIndexBuffer));
        setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractTriangleIndices, data, mappedTriangleIndexBuffer, _numTriangleIndices);
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.triangleIndexBuffer));

        CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.selectedConnectionBuffer));
        ConnectionArrowVertexData* mappedSelectedConnectionBuffer;
        size_t selectedConnectionBufferSize;
        CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedSelectedConnectionBuffer), &selectedConnectionBufferSize, renderingData.selectedConnectionBuffer));
        setValueToDevice(_numSelectedConnectionVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedConnectionData, data, mappedSelectedConnectionBuffer, _numSelectedConnectionVertices);
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.selectedConnectionBuffer));

        CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.attackEventBuffer));
        AttackEventVertexData* mappedAttackEventBuffer;
        size_t attackEventBufferSize;
        CHECK_FOR_CUDA_ERROR(
            cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedAttackEventBuffer), &attackEventBufferSize, renderingData.attackEventBuffer));
        setValueToDevice(_numAttackEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractAttackEventData, data, mappedAttackEventBuffer, _numAttackEventVertices);
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.attackEventBuffer));

        CHECK_FOR_CUDA_ERROR(cudaGraphicsMapResources(1, &renderingData.detonationEventBuffer));
        DetonationEventVertexData* mappedDetonationEventBuffer;
        size_t detonationEventBufferSize;
        CHECK_FOR_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedDetonationEventBuffer), &detonationEventBufferSize, renderingData.detonationEventBuffer));
        setValueToDevice(_numDetonationEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractDetonationEventData, data, mappedDetonationEventBuffer, _numDetonationEventVertices);
        CHECK_FOR_CUDA_ERROR(cudaGraphicsUnmapResources(1, &renderingData.detonationEventBuffer));
    } else {
        // No-interop mode: extract to device buffers
        KERNEL_CALL(cudaExtractCellData, data, renderingData.deviceCellBuffer);

        KERNEL_CALL(cudaExtractEnergyParticleData, data, renderingData.deviceEnergyParticleBuffer);

        setValueToDevice(_numLocations, static_cast<uint64_t>(0));
        KERNEL_CALL_1_1(cudaExtractLocationData, data, renderingData.deviceLocationBuffer, _numLocations, visibleTopLeft);

        setValueToDevice(_numSelectedObjects, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedObjectData, data, renderingData.deviceSelectedObjectBuffer, _numSelectedObjects);

        setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractLineIndices, data, renderingData.deviceLineIndexBuffer, _numLineIndices);

        setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractTriangleIndices, data, renderingData.deviceTriangleIndexBuffer, _numTriangleIndices);

        setValueToDevice(_numSelectedConnectionVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedConnectionData, data, renderingData.deviceSelectedConnectionBuffer, _numSelectedConnectionVertices);

        setValueToDevice(_numAttackEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractAttackEventData, data, renderingData.deviceAttackEventBuffer, _numAttackEventVertices);

        setValueToDevice(_numDetonationEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractDetonationEventData, data, renderingData.deviceDetonationEventBuffer, _numDetonationEventVertices);
    }
}

void GeometryKernelsService::extractObjectDataToCpuBuffers(
    SettingsForSimulation const& settings,
    SimulationData data,
    CpuGeometryBuffers& cpuBuffers,
    RealRect const& visibleWorldRect)
{
    auto const& gpuSettings = settings.cudaSettings;
    float2 const visibleTopLeft{visibleWorldRect.topLeft.x, visibleWorldRect.topLeft.y};

    // Get counts first
    auto numCells = data.objects.cells.getNumEntries_host();
    auto numParticles = data.objects.particles.getNumEntries_host();

    setValueToDevice(_numLocations, static_cast<uint64_t>(0));
    KERNEL_CALL_1_1(cudaExtractLocationData, data, nullptr, _numLocations, visibleTopLeft);
    cudaDeviceSynchronize();
    auto numLocations = copyToHost(_numLocations);

    setValueToDevice(_numSelectedObjects, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractSelectedObjectData, data, nullptr, _numSelectedObjects);
    cudaDeviceSynchronize();
    auto numSelectedObjects = copyToHost(_numSelectedObjects);

    setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractLineIndices, data, nullptr, _numLineIndices);
    cudaDeviceSynchronize();
    auto numLineIndices = copyToHost(_numLineIndices);

    setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractTriangleIndices, data, nullptr, _numTriangleIndices);
    cudaDeviceSynchronize();
    auto numTriangleIndices = copyToHost(_numTriangleIndices);

    setValueToDevice(_numSelectedConnectionVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractSelectedConnectionData, data, nullptr, _numSelectedConnectionVertices);
    cudaDeviceSynchronize();
    auto numConnectionArrows = copyToHost(_numSelectedConnectionVertices);

    setValueToDevice(_numAttackEventVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractAttackEventData, data, nullptr, _numAttackEventVertices);
    cudaDeviceSynchronize();
    auto numAttackEvents = copyToHost(_numAttackEventVertices);

    setValueToDevice(_numDetonationEventVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractDetonationEventData, data, nullptr, _numDetonationEventVertices);
    cudaDeviceSynchronize();
    auto numDetonationEvents = copyToHost(_numDetonationEventVertices);

    // Allocate device buffers
    CellVertexData* deviceCellBuffer = nullptr;
    EnergyParticleVertexData* deviceEnergyParticleBuffer = nullptr;
    LocationVertexData* deviceLocationBuffer = nullptr;
    SelectedObjectVertexData* deviceSelectedObjectBuffer = nullptr;
    unsigned int* deviceLineIndexBuffer = nullptr;
    unsigned int* deviceTriangleIndexBuffer = nullptr;
    ConnectionArrowVertexData* deviceConnectionArrowBuffer = nullptr;
    AttackEventVertexData* deviceAttackEventBuffer = nullptr;
    DetonationEventVertexData* deviceDetonationEventBuffer = nullptr;

    if (numCells > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&deviceCellBuffer, numCells * sizeof(CellVertexData)));
    }
    if (numParticles > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&deviceEnergyParticleBuffer, numParticles * sizeof(EnergyParticleVertexData)));
    }
    if (numLocations > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&deviceLocationBuffer, numLocations * sizeof(LocationVertexData)));
    }
    if (numSelectedObjects > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&deviceSelectedObjectBuffer, numSelectedObjects * sizeof(SelectedObjectVertexData)));
    }
    if (numLineIndices > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&deviceLineIndexBuffer, numLineIndices * sizeof(unsigned int)));
    }
    if (numTriangleIndices > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&deviceTriangleIndexBuffer, numTriangleIndices * sizeof(unsigned int)));
    }
    if (numConnectionArrows > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&deviceConnectionArrowBuffer, numConnectionArrows * sizeof(ConnectionArrowVertexData)));
    }
    if (numAttackEvents > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&deviceAttackEventBuffer, numAttackEvents * sizeof(AttackEventVertexData)));
    }
    if (numDetonationEvents > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMalloc(&deviceDetonationEventBuffer, numDetonationEvents * sizeof(DetonationEventVertexData)));
    }

    // Extract data to device buffers
    if (numCells > 0) {
        KERNEL_CALL(cudaExtractCellData, data, deviceCellBuffer);
    }
    if (numParticles > 0) {
        KERNEL_CALL(cudaExtractEnergyParticleData, data, deviceEnergyParticleBuffer);
    }
    if (numLocations > 0) {
        setValueToDevice(_numLocations, static_cast<uint64_t>(0));
        KERNEL_CALL_1_1(cudaExtractLocationData, data, deviceLocationBuffer, _numLocations, visibleTopLeft);
    }
    if (numSelectedObjects > 0) {
        setValueToDevice(_numSelectedObjects, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedObjectData, data, deviceSelectedObjectBuffer, _numSelectedObjects);
    }
    if (numLineIndices > 0) {
        setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractLineIndices, data, deviceLineIndexBuffer, _numLineIndices);
    }
    if (numTriangleIndices > 0) {
        setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractTriangleIndices, data, deviceTriangleIndexBuffer, _numTriangleIndices);
    }
    if (numConnectionArrows > 0) {
        setValueToDevice(_numSelectedConnectionVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedConnectionData, data, deviceConnectionArrowBuffer, _numSelectedConnectionVertices);
    }
    if (numAttackEvents > 0) {
        setValueToDevice(_numAttackEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractAttackEventData, data, deviceAttackEventBuffer, _numAttackEventVertices);
    }
    if (numDetonationEvents > 0) {
        setValueToDevice(_numDetonationEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractDetonationEventData, data, deviceDetonationEventBuffer, _numDetonationEventVertices);
    }
    cudaDeviceSynchronize();

    // Resize CPU buffers
    cpuBuffers.cells.resize(numCells);
    cpuBuffers.energyParticles.resize(numParticles);
    cpuBuffers.locations.resize(numLocations);
    cpuBuffers.selectedObjects.resize(numSelectedObjects);
    cpuBuffers.lineIndices.resize(numLineIndices);
    cpuBuffers.triangleIndices.resize(numTriangleIndices);
    cpuBuffers.connectionArrows.resize(numConnectionArrows);
    cpuBuffers.attackEvents.resize(numAttackEvents);
    cpuBuffers.detonationEvents.resize(numDetonationEvents);

    // Copy from device to host
    if (numCells > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(cpuBuffers.cells.data(), deviceCellBuffer, numCells * sizeof(CellVertexData), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaFree(deviceCellBuffer));
    }
    if (numParticles > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(cpuBuffers.energyParticles.data(), deviceEnergyParticleBuffer, numParticles * sizeof(EnergyParticleVertexData), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaFree(deviceEnergyParticleBuffer));
    }
    if (numLocations > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(cpuBuffers.locations.data(), deviceLocationBuffer, numLocations * sizeof(LocationVertexData), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaFree(deviceLocationBuffer));
    }
    if (numSelectedObjects > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(cpuBuffers.selectedObjects.data(), deviceSelectedObjectBuffer, numSelectedObjects * sizeof(SelectedObjectVertexData), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaFree(deviceSelectedObjectBuffer));
    }
    if (numLineIndices > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(cpuBuffers.lineIndices.data(), deviceLineIndexBuffer, numLineIndices * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaFree(deviceLineIndexBuffer));
    }
    if (numTriangleIndices > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(cpuBuffers.triangleIndices.data(), deviceTriangleIndexBuffer, numTriangleIndices * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaFree(deviceTriangleIndexBuffer));
    }
    if (numConnectionArrows > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(cpuBuffers.connectionArrows.data(), deviceConnectionArrowBuffer, numConnectionArrows * sizeof(ConnectionArrowVertexData), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaFree(deviceConnectionArrowBuffer));
    }
    if (numAttackEvents > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(cpuBuffers.attackEvents.data(), deviceAttackEventBuffer, numAttackEvents * sizeof(AttackEventVertexData), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaFree(deviceAttackEventBuffer));
    }
    if (numDetonationEvents > 0) {
        CHECK_FOR_CUDA_ERROR(cudaMemcpy(cpuBuffers.detonationEvents.data(), deviceDetonationEventBuffer, numDetonationEvents * sizeof(DetonationEventVertexData), cudaMemcpyDeviceToHost));
        CHECK_FOR_CUDA_ERROR(cudaFree(deviceDetonationEventBuffer));
    }
}
