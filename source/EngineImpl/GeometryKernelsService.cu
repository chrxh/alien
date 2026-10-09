#include "GeometryKernelsService.cuh"

#include <ranges>
#include <vector>

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>

#include <Data/EngineConstants.h>

#include <EngineInterface/SettingsForSimulation.h>

#include <EngineKernels/CudaGeometryBuffers.cuh>
#include <EngineKernels/GeometryKernels.cuh>
#include <EngineKernels/KernelLauncher.cuh>

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
    CudaMemoryManager::getInstance().acquireMemory(1, _counters);
}

void GeometryKernelsService::shutdown()
{
    CudaMemoryManager::getInstance().freeMemory(_counters);
}

namespace
{
    // Writes a known pattern into the shared memory through CUDA and reads it back through the geometry buffers. Only if
    // that value survives do both APIs really address the same allocation, which is not a given when the graphics device
    // and the CUDA device are different GPUs. Leaves no error state behind.
    bool isSharedMemoryWorking(GeometryBuffers const& geometryBuffers, CudaGeometryBuffers const& renderingData)
    {
        auto constexpr NumValues = 64;
        auto constexpr PatternByte = 0xA5;
        auto constexpr ExpectedValue = 0xA5A5A5A5u;
        auto constexpr SizeInBytes = NumValues * sizeof(uint32_t);

        auto succeeded = cudaMemset(renderingData.getBuffer<void>(GeometryBufferType_Objects), PatternByte, SizeInBytes) == cudaSuccess
            && cudaDeviceSynchronize() == cudaSuccess;
        cudaGetLastError();
        if (!succeeded) {
            return false;
        }
        std::vector<uint32_t> readBack(NumValues, 0);
        try {
            geometryBuffers->download(GeometryBufferType_Objects, readBack.data(), SizeInBytes);
        } catch (std::exception const&) {
            return false;
        }
        return std::ranges::all_of(readBack, [](uint32_t value) { return value == ExpectedValue; });
    }
}

bool GeometryKernelsService::prepareInterop(GeometryBuffers const& geometryBuffers, CudaGeometryBuffers& renderingData)
{
    if (!geometryBuffers->isMemoryShareable() || _interopUsable == false) {
        return false;
    }
    auto importResult = renderingData.importSharedMemory(geometryBuffers);
    if (_interopUsable.has_value()) {
        CHECK_FOR_DEVICE_ERRORS(importResult);
        return true;
    }

    cudaGetLastError();  // A failed probe must not leave the error state behind for the rest of the program
    _interopUsable = importResult == cudaSuccess && isSharedMemoryWorking(geometryBuffers, renderingData);
    if (*_interopUsable) {
        log(Priority::Important, "CUDA-Vulkan interop is working");
    } else {
        renderingData.release();
        GlobalSettings::get().setInterop(false);
        log(Priority::Important, "CUDA-Vulkan interop is not working on this system, falling back to the transfer over host memory");
    }
    return *_interopUsable;
}

void GeometryKernelsService::correctPositionsForRendering(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect)
{
    auto const& launchSettings = settings.kernelLaunchSettings;
    float2 const visibleTopLeft{visibleWorldRect.topLeft.x, visibleWorldRect.topLeft.y};

    launchKernelOnDefaultStream(KERNEL(cudaCorrectPositionsForRendering), LaunchConfig{launchSettings.numBlocks, 8}, data, visibleTopLeft);
}

void GeometryKernelsService::restorePositions(SettingsForSimulation const& settings, SimulationData data)
{
    auto const& launchSettings = settings.kernelLaunchSettings;

    launchKernelOnDefaultStream(KERNEL(cudaCorrectPositionsForRendering), LaunchConfig{launchSettings.numBlocks, 8}, data, float2{0, 0});
}

NumRenderObjects GeometryKernelsService::getNumRenderObjects(SettingsForSimulation const& settings, SimulationData data, RealRect const& visibleWorldRect)
{
    auto const& launchSettings = settings.kernelLaunchSettings;
    float2 const visibleTopLeft{visibleWorldRect.topLeft.x, visibleWorldRect.topLeft.y};
    float2 const visibleBottomRight{visibleWorldRect.bottomRight.x, visibleWorldRect.bottomRight.y};
    GeometryExtractionContext const context{visibleTopLeft, visibleBottomRight, computeCullingMargin(settings)};

    CHECK_FOR_DEVICE_ERRORS(cudaMemset(_counters, 0, sizeof(NumRenderObjects)));

    launchKernelOnDefaultStream(KERNEL(cudaExtractObjectData), LaunchConfig{launchSettings.numBlocks, 8}, data, nullptr, &_counters->objects, context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractFluidParticleData), LaunchConfig{launchSettings.numBlocks, 8}, data, nullptr, &_counters->fluidParticles, context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractSelectedObjectData), LaunchConfig{launchSettings.numBlocks, 8}, data, nullptr, &_counters->selectedObjects, context);
    launchKernelOnDefaultStream(KERNEL(cudaExtractLineIndices), LaunchConfig{launchSettings.numBlocks, 8}, data, nullptr, &_counters->lineIndices, context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractTriangleIndices), LaunchConfig{launchSettings.numBlocks, 8}, data, nullptr, &_counters->triangleIndices, context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractSelectedConnectionData), LaunchConfig{launchSettings.numBlocks, 8}, data, nullptr, &_counters->connectionArrowVertices, context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractAttackEventData), LaunchConfig{launchSettings.numBlocks, 8}, data, nullptr, &_counters->attackEventVertices, context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractDetonationEventData), LaunchConfig{launchSettings.numBlocks, 8}, data, nullptr, &_counters->detonationEventVertices, context);
    launchKernelOnDefaultStream(KERNEL(cudaExtractLocationData), LaunchConfig{1, 1}, data, nullptr, &_counters->locations, visibleTopLeft);

    NumRenderObjects result;
    copyToHost(&result, _counters);
    return result;
}

void GeometryKernelsService::extractObjectData(
    SettingsForSimulation const& settings,
    SimulationData data,
    CudaGeometryBuffers const& renderingData,
    RealRect const& visibleWorldRect)
{
    auto const& launchSettings = settings.kernelLaunchSettings;
    float2 const visibleTopLeft{visibleWorldRect.topLeft.x, visibleWorldRect.topLeft.y};
    float2 const visibleBottomRight{visibleWorldRect.bottomRight.x, visibleWorldRect.bottomRight.y};
    GeometryExtractionContext const context{visibleTopLeft, visibleBottomRight, computeCullingMargin(settings)};

    CHECK_FOR_DEVICE_ERRORS(cudaMemset(_counters, 0, sizeof(NumRenderObjects)));

    launchKernelOnDefaultStream(
        KERNEL(cudaExtractObjectData),
        LaunchConfig{launchSettings.numBlocks, 8},
        data,
        renderingData.getBuffer<ObjectVertexData>(GeometryBufferType_Objects),
        &_counters->objects,
        context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractFluidParticleData),
        LaunchConfig{launchSettings.numBlocks, 8},
        data,
        renderingData.getBuffer<FluidParticleVertexData>(GeometryBufferType_FluidParticles),
        &_counters->fluidParticles,
        context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractLocationData),
        LaunchConfig{1, 1},
        data,
        renderingData.getBuffer<LocationVertexData>(GeometryBufferType_Locations),
        &_counters->locations,
        visibleTopLeft);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractSelectedObjectData),
        LaunchConfig{launchSettings.numBlocks, 8},
        data,
        renderingData.getBuffer<SelectedObjectVertexData>(GeometryBufferType_SelectedObjects),
        &_counters->selectedObjects,
        context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractLineIndices),
        LaunchConfig{launchSettings.numBlocks, 8},
        data,
        renderingData.getBuffer<unsigned int>(GeometryBufferType_LineIndices),
        &_counters->lineIndices,
        context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractTriangleIndices),
        LaunchConfig{launchSettings.numBlocks, 8},
        data,
        renderingData.getBuffer<unsigned int>(GeometryBufferType_TriangleIndices),
        &_counters->triangleIndices,
        context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractSelectedConnectionData),
        LaunchConfig{launchSettings.numBlocks, 8},
        data,
        renderingData.getBuffer<ConnectionArrowVertexData>(GeometryBufferType_SelectedConnections),
        &_counters->connectionArrowVertices,
        context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractAttackEventData),
        LaunchConfig{launchSettings.numBlocks, 8},
        data,
        renderingData.getBuffer<AttackEventVertexData>(GeometryBufferType_AttackEvents),
        &_counters->attackEventVertices,
        context);
    launchKernelOnDefaultStream(
        KERNEL(cudaExtractDetonationEventData),
        LaunchConfig{launchSettings.numBlocks, 8},
        data,
        renderingData.getBuffer<DetonationEventVertexData>(GeometryBufferType_DetonationEvents),
        &_counters->detonationEventVertices,
        context);
}
