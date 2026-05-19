#include "GeometryKernelsService.cuh"

#include <Base/GlobalSettings.h>

#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/SettingsForSimulation.h>

#include <EngineKernels/CudaGeometryBuffers.cuh>
#include <EngineKernels/GeometryKernels.cuh>

namespace
{
    float computeCullingMargin(SettingsForSimulation const& settings)
    {
        float result = 10.0f;
        for (int i = 0; i < MAX_COLORS; ++i) {
            result = std::max(result, settings.simulationParameters.maxBindingDistance.value[i]);
        }
        return result;
    }
}

void GeometryKernelsService::init()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.acquireMemory(1, _numObjects);
    memoryManager.acquireMemory(1, _numLineIndices);
    memoryManager.acquireMemory(1, _numTriangleIndices);
    memoryManager.acquireMemory(1, _numSelectedConnectionVertices);
    memoryManager.acquireMemory(1, _numSelectedObjects);
    memoryManager.acquireMemory(1, _numAttackEventVertices);
    memoryManager.acquireMemory(1, _numDetonationEventVertices);
    memoryManager.acquireMemory(1, _numLocations);
    memoryManager.acquireMemory(1, _numFluidParticles);
}

void GeometryKernelsService::shutdown()
{
    auto& memoryManager = CudaMemoryManager::getInstance();
    memoryManager.freeMemory(_numObjects);
    memoryManager.freeMemory(_numLineIndices);
    memoryManager.freeMemory(_numTriangleIndices);
    memoryManager.freeMemory(_numSelectedConnectionVertices);
    memoryManager.freeMemory(_numSelectedObjects);
    memoryManager.freeMemory(_numAttackEventVertices);
    memoryManager.freeMemory(_numDetonationEventVertices);
    memoryManager.freeMemory(_numLocations);
    memoryManager.freeMemory(_numFluidParticles);
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
    float2 const visibleBottomRight{visibleWorldRect.bottomRight.x, visibleWorldRect.bottomRight.y};
    GeometryExtractionContext const context{visibleTopLeft, visibleBottomRight, computeCullingMargin(settings)};

    NumRenderObjects result;
    setValueToDevice(_numObjects, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractObjectData, data, nullptr, _numObjects, context);
    cudaDeviceSynchronize();
    result.objects = copyToHost(_numObjects);

    setValueToDevice(_numFluidParticles, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractFluidParticleData, data, nullptr, _numFluidParticles, context);
    cudaDeviceSynchronize();
    result.fluidParticles = copyToHost(_numFluidParticles);

    setValueToDevice(_numSelectedObjects, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractSelectedObjectData, data, nullptr, _numSelectedObjects, context);
    cudaDeviceSynchronize();
    result.selectedObjects = copyToHost(_numSelectedObjects);

    setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractLineIndices, data, nullptr, _numLineIndices, context);
    cudaDeviceSynchronize();
    result.lineIndices = copyToHost(_numLineIndices);

    setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractTriangleIndices, data, nullptr, _numTriangleIndices, context);
    cudaDeviceSynchronize();
    result.triangleIndices = copyToHost(_numTriangleIndices);

    setValueToDevice(_numSelectedConnectionVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractSelectedConnectionData, data, nullptr, _numSelectedConnectionVertices, context);
    cudaDeviceSynchronize();
    result.connectionArrowVertices = copyToHost(_numSelectedConnectionVertices);

    setValueToDevice(_numAttackEventVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractAttackEventData, data, nullptr, _numAttackEventVertices, context);
    cudaDeviceSynchronize();
    result.attackEventVertices = copyToHost(_numAttackEventVertices);

    setValueToDevice(_numDetonationEventVertices, static_cast<uint64_t>(0));
    KERNEL_CALL(cudaExtractDetonationEventData, data, nullptr, _numDetonationEventVertices, context);
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
    float2 const visibleBottomRight{visibleWorldRect.bottomRight.x, visibleWorldRect.bottomRight.y};
    GeometryExtractionContext const context{visibleTopLeft, visibleBottomRight, computeCullingMargin(settings)};

    if (useInterop) {
        // Interop mode: use CUDA-OpenGL interoperability
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(1, &renderingData.vertexBuffer));
        ObjectVertexData* mappedCellBuffer;
        size_t bufferSize;
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedCellBuffer), &bufferSize, renderingData.vertexBuffer));
        setValueToDevice(_numObjects, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractObjectData, data, mappedCellBuffer, _numObjects, context);
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(1, &renderingData.vertexBuffer));

        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(1, &renderingData.fluidParticleBuffer));
        FluidParticleVertexData* mappedFluidParticleBuffer;
        size_t fluidParticleBufferSize;
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedFluidParticleBuffer), &fluidParticleBufferSize, renderingData.fluidParticleBuffer));
        setValueToDevice(_numFluidParticles, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractFluidParticleData, data, mappedFluidParticleBuffer, _numFluidParticles, context);
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(1, &renderingData.fluidParticleBuffer));

        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(1, &renderingData.locationBuffer));
        LocationVertexData* mappedLocationBuffer;
        size_t locationBufferSize;
        CHECK_FOR_DEVICE_ERRORS(
            cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedLocationBuffer), &locationBufferSize, renderingData.locationBuffer));
        setValueToDevice(_numLocations, static_cast<uint64_t>(0));
        KERNEL_CALL_1_1(cudaExtractLocationData, data, mappedLocationBuffer, _numLocations, visibleTopLeft);
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(1, &renderingData.locationBuffer));

        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(1, &renderingData.selectedObjectBuffer));
        SelectedObjectVertexData* mappedSelectedObjectBuffer;
        size_t selectedObjectBufferSize;
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedSelectedObjectBuffer), &selectedObjectBufferSize, renderingData.selectedObjectBuffer));
        setValueToDevice(_numSelectedObjects, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedObjectData, data, mappedSelectedObjectBuffer, _numSelectedObjects, context);
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(1, &renderingData.selectedObjectBuffer));

        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(1, &renderingData.lineIndexBuffer));
        unsigned int* mappedLineIndexBuffer;
        size_t lineIndexBufferSize;
        CHECK_FOR_DEVICE_ERRORS(
            cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedLineIndexBuffer), &lineIndexBufferSize, renderingData.lineIndexBuffer));
        setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractLineIndices, data, mappedLineIndexBuffer, _numLineIndices, context);
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(1, &renderingData.lineIndexBuffer));

        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(1, &renderingData.triangleIndexBuffer));
        unsigned int* mappedTriangleIndexBuffer;
        size_t triangleIndexBufferSize;
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedTriangleIndexBuffer), &triangleIndexBufferSize, renderingData.triangleIndexBuffer));
        setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractTriangleIndices, data, mappedTriangleIndexBuffer, _numTriangleIndices, context);
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(1, &renderingData.triangleIndexBuffer));

        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(1, &renderingData.selectedConnectionBuffer));
        ConnectionArrowVertexData* mappedSelectedConnectionBuffer;
        size_t selectedConnectionBufferSize;
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedSelectedConnectionBuffer), &selectedConnectionBufferSize, renderingData.selectedConnectionBuffer));
        setValueToDevice(_numSelectedConnectionVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedConnectionData, data, mappedSelectedConnectionBuffer, _numSelectedConnectionVertices, context);
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(1, &renderingData.selectedConnectionBuffer));

        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(1, &renderingData.attackEventBuffer));
        AttackEventVertexData* mappedAttackEventBuffer;
        size_t attackEventBufferSize;
        CHECK_FOR_DEVICE_ERRORS(
            cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedAttackEventBuffer), &attackEventBufferSize, renderingData.attackEventBuffer));
        setValueToDevice(_numAttackEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractAttackEventData, data, mappedAttackEventBuffer, _numAttackEventVertices, context);
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(1, &renderingData.attackEventBuffer));

        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(1, &renderingData.detonationEventBuffer));
        DetonationEventVertexData* mappedDetonationEventBuffer;
        size_t detonationEventBufferSize;
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsResourceGetMappedPointer(
            reinterpret_cast<void**>(&mappedDetonationEventBuffer), &detonationEventBufferSize, renderingData.detonationEventBuffer));
        setValueToDevice(_numDetonationEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractDetonationEventData, data, mappedDetonationEventBuffer, _numDetonationEventVertices, context);
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(1, &renderingData.detonationEventBuffer));
    } else {
        // No-interop mode: extract to device buffers
        setValueToDevice(_numObjects, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractObjectData, data, renderingData.deviceObjectBuffer, _numObjects, context);

        setValueToDevice(_numFluidParticles, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractFluidParticleData, data, renderingData.deviceFluidParticleBuffer, _numFluidParticles, context);

        setValueToDevice(_numLocations, static_cast<uint64_t>(0));
        KERNEL_CALL_1_1(cudaExtractLocationData, data, renderingData.deviceLocationBuffer, _numLocations, visibleTopLeft);

        setValueToDevice(_numSelectedObjects, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedObjectData, data, renderingData.deviceSelectedObjectBuffer, _numSelectedObjects, context);

        setValueToDevice(_numLineIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractLineIndices, data, renderingData.deviceLineIndexBuffer, _numLineIndices, context);

        setValueToDevice(_numTriangleIndices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractTriangleIndices, data, renderingData.deviceTriangleIndexBuffer, _numTriangleIndices, context);

        setValueToDevice(_numSelectedConnectionVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractSelectedConnectionData, data, renderingData.deviceSelectedConnectionBuffer, _numSelectedConnectionVertices, context);

        setValueToDevice(_numAttackEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractAttackEventData, data, renderingData.deviceAttackEventBuffer, _numAttackEventVertices, context);

        setValueToDevice(_numDetonationEventVertices, static_cast<uint64_t>(0));
        KERNEL_CALL(cudaExtractDetonationEventData, data, renderingData.deviceDetonationEventBuffer, _numDetonationEventVertices, context);
    }
}
