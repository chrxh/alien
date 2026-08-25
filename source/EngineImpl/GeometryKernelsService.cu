#include "GeometryKernelsService.cuh"

#include <ranges>
#include <vector>

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>

#include <EngineInterface/EngineConstants.h>
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

    // Writes a known pattern into an OpenGL buffer through CUDA and reads it back with OpenGL. Only if that value
    // survives do both APIs really address the same allocation, which is not a given when the OpenGL context and the
    // CUDA device belong to different GPUs. Leaves no error state behind.
    bool isCudaOpenGLInteropWorking()
    {
        auto constexpr NumValues = 64;
        auto constexpr PatternByte = 0xA5;
        auto constexpr ExpectedValue = 0xA5A5A5A5u;
        auto constexpr SizeInBytes = NumValues * sizeof(uint32_t);

        GLuint buffer = 0;
        glGenBuffers(1, &buffer);
        if (buffer == 0) {
            return false;
        }
        glBindBuffer(GL_ARRAY_BUFFER, buffer);
        glBufferData(GL_ARRAY_BUFFER, SizeInBytes, nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        if (glGetError() != GL_NO_ERROR) {
            glDeleteBuffers(1, &buffer);
            return false;
        }

        auto succeeded = false;
        cudaGraphicsResource* resource = nullptr;
        if (cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsWriteDiscard) == cudaSuccess) {
            if (cudaGraphicsMapResources(1, &resource) == cudaSuccess) {
                void* mapped = nullptr;
                size_t mappedSize = 0;
                succeeded = cudaGraphicsResourceGetMappedPointer(&mapped, &mappedSize, resource) == cudaSuccess && mappedSize >= SizeInBytes
                    && cudaMemset(mapped, PatternByte, SizeInBytes) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
                if (cudaGraphicsUnmapResources(1, &resource) != cudaSuccess) {
                    succeeded = false;
                }
            }
            cudaGraphicsUnregisterResource(resource);
        }
        cudaGetLastError();  // A failed probe must not leave the error state behind for the rest of the program

        if (succeeded) {
            std::vector<uint32_t> readBack(NumValues, 0);
            glBindBuffer(GL_ARRAY_BUFFER, buffer);
            glGetBufferSubData(GL_ARRAY_BUFFER, 0, SizeInBytes, readBack.data());
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            succeeded = glGetError() == GL_NO_ERROR && std::ranges::all_of(readBack, [](uint32_t value) { return value == ExpectedValue; });
        }

        glDeleteBuffers(1, &buffer);
        return succeeded;
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

bool GeometryKernelsService::checkForInterop()
{
    if (!_interopUsable.has_value()) {
        _interopUsable = isCudaOpenGLInteropWorking();
        if (*_interopUsable) {
            log(Priority::Important, "CUDA-OpenGL interop is working");
        } else {
            GlobalSettings::get().setInterop(false);
            log(Priority::Important, "CUDA-OpenGL interop is not working on this system, falling back to the transfer over host memory");
        }
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
    CudaGeometryBuffers& renderingData,
    RealRect const& visibleWorldRect,
    bool useInterop)
{
    auto const& launchSettings = settings.kernelLaunchSettings;
    float2 const visibleTopLeft{visibleWorldRect.topLeft.x, visibleWorldRect.topLeft.y};
    float2 const visibleBottomRight{visibleWorldRect.bottomRight.x, visibleWorldRect.bottomRight.y};
    GeometryExtractionContext const context{visibleTopLeft, visibleBottomRight, computeCullingMargin(settings)};

    if (useInterop) {
        // Interop mode: Mapping is a synchronization point between the OpenGL and the CUDA context
        cudaGraphicsResource* resources[] = {
            renderingData.vertexBuffer,
            renderingData.fluidParticleBuffer,
            renderingData.locationBuffer,
            renderingData.selectedObjectBuffer,
            renderingData.lineIndexBuffer,
            renderingData.triangleIndexBuffer,
            renderingData.selectedConnectionBuffer,
            renderingData.attackEventBuffer,
            renderingData.detonationEventBuffer,
        };
        auto constexpr NumResources = static_cast<int>(sizeof(resources) / sizeof(resources[0]));
        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsMapResources(NumResources, resources));

        auto const mappedPointer = [&resources](int index) {
            void* result = nullptr;
            size_t size = 0;
            CHECK_FOR_DEVICE_ERRORS(cudaGraphicsResourceGetMappedPointer(&result, &size, resources[index]));
            return result;
        };
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_counters, 0, sizeof(NumRenderObjects)));

        launchKernelOnDefaultStream(
            KERNEL(cudaExtractObjectData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            static_cast<ObjectVertexData*>(mappedPointer(0)),
            &_counters->objects,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractFluidParticleData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            static_cast<FluidParticleVertexData*>(mappedPointer(1)),
            &_counters->fluidParticles,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractLocationData),
            LaunchConfig{1, 1},
            data,
            static_cast<LocationVertexData*>(mappedPointer(2)),
            &_counters->locations,
            visibleTopLeft);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractSelectedObjectData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            static_cast<SelectedObjectVertexData*>(mappedPointer(3)),
            &_counters->selectedObjects,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractLineIndices),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            static_cast<unsigned int*>(mappedPointer(4)),
            &_counters->lineIndices,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractTriangleIndices),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            static_cast<unsigned int*>(mappedPointer(5)),
            &_counters->triangleIndices,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractSelectedConnectionData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            static_cast<ConnectionArrowVertexData*>(mappedPointer(6)),
            &_counters->connectionArrowVertices,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractAttackEventData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            static_cast<AttackEventVertexData*>(mappedPointer(7)),
            &_counters->attackEventVertices,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractDetonationEventData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            static_cast<DetonationEventVertexData*>(mappedPointer(8)),
            &_counters->detonationEventVertices,
            context);

        CHECK_FOR_DEVICE_ERRORS(cudaGraphicsUnmapResources(NumResources, resources));
    } else {
        // No-interop mode: extract to device buffers
        CHECK_FOR_DEVICE_ERRORS(cudaMemset(_counters, 0, sizeof(NumRenderObjects)));

        launchKernelOnDefaultStream(
            KERNEL(cudaExtractObjectData), LaunchConfig{launchSettings.numBlocks, 8}, data, renderingData.deviceObjectBuffer, &_counters->objects, context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractFluidParticleData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            renderingData.deviceFluidParticleBuffer,
            &_counters->fluidParticles,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractLocationData), LaunchConfig{1, 1}, data, renderingData.deviceLocationBuffer, &_counters->locations, visibleTopLeft);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractSelectedObjectData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            renderingData.deviceSelectedObjectBuffer,
            &_counters->selectedObjects,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractLineIndices),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            renderingData.deviceLineIndexBuffer,
            &_counters->lineIndices,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractTriangleIndices),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            renderingData.deviceTriangleIndexBuffer,
            &_counters->triangleIndices,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractSelectedConnectionData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            renderingData.deviceSelectedConnectionBuffer,
            &_counters->connectionArrowVertices,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractAttackEventData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            renderingData.deviceAttackEventBuffer,
            &_counters->attackEventVertices,
            context);
        launchKernelOnDefaultStream(
            KERNEL(cudaExtractDetonationEventData),
            LaunchConfig{launchSettings.numBlocks, 8},
            data,
            renderingData.deviceDetonationEventBuffer,
            &_counters->detonationEventVertices,
            context);
    }
}
