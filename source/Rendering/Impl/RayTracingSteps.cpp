#include "RayTracingSteps.h"

#include <algorithm>

#include <Rendering/Impl/Shaders/ShaderSources.h>

#include "VulkanContext.h"

namespace
{
    // Steps that compute into the shared resources. Steps that work on the scene of the objects are skipped while it is not needed.
    class _RayTracingStep : public _RenderStep
    {
    public:
        bool hasTarget() const override { return false; }

    protected:
        _RayTracingStep(RayTracingResources const& resources, bool requiresScene)
            : _RenderStep(StepParameters())
            , _resources(resources)
            , _requiresScene(requiresScene)
        {}

        bool isSkipped(ExecutionParameters const& parameters) const
        {
            return _requiresScene && !_RayTracingResources::isSceneNeeded(parameters._view, *parameters._simulationParameters);
        }

        RayTracingResources _resources;
        bool _requiresScene = false;
    };

    class _RayTracingFrameStep : public _RayTracingStep
    {
    public:
        static RenderStep create(RayTracingResources const& resources) { return RenderStep(new _RayTracingFrameStep(resources)); }

        void execute(ExecutionParameters parameters) override { _resources->beginFrame(parameters); }

    private:
        _RayTracingFrameStep(RayTracingResources const& resources)
            : _RayTracingStep(resources, false)
        {}
    };

    using DispatchSizeFunc = std::function<IntVector2D(_RayTracingResources const&, NumRenderObjects const&)>;

    struct ComputeStepParameters
    {
        // Appended to Shaders/RayTracingCommonCS.h
        MEMBER(ComputeStepParameters, std::string, shader, std::string());

        MEMBER(ComputeStepParameters, IntVector2D, workGroupSize, IntVector2D());

        // At least one invocation per element
        MEMBER(ComputeStepParameters, DispatchSizeFunc, dispatchSize, DispatchSizeFunc());

        // The barriers in between the steps follow from these accesses
        MEMBER(ComputeStepParameters, RayTracingAccesses, accesses, RayTracingAccesses());

        MEMBER(ComputeStepParameters, bool, requiresScene, false);
    };

    class _RayTracingComputeStep : public _RayTracingStep
    {
    public:
        static RenderStep create(RayTracingResources const& resources, ComputeStepParameters const& parameters)
        {
            return RenderStep(new _RayTracingComputeStep(resources, parameters));
        }

        void execute(ExecutionParameters parameters) override
        {
            if (isSkipped(parameters)) {
                return;
            }
            auto commandBuffer = parameters._renderInfo.commandBuffer;
            _resources->access(commandBuffer, _accesses);
            _resources->bindDescriptorSet(commandBuffer);
            _pipeline->dispatch(commandBuffer, _dispatchSize(*_resources, parameters._geometryBuffers->getNumObjects()));
        }

    private:
        _RayTracingComputeStep(RayTracingResources const& resources, ComputeStepParameters const& parameters)
            : _RayTracingStep(resources, parameters._requiresScene)
            , _pipeline(resources->createPipeline(parameters._shader, parameters._workGroupSize))
            , _dispatchSize(parameters._dispatchSize)
            , _accesses(parameters._accesses)
        {}

        RayTracingComputePipeline _pipeline;
        DispatchSizeFunc _dispatchSize;
        RayTracingAccesses _accesses;
    };

    // Builds the acceleration structures from the boxes of the objects
    class _RayTracingSceneStep : public _RayTracingStep
    {
    public:
        static RenderStep create(RayTracingResources const& resources) { return RenderStep(new _RayTracingSceneStep(resources)); }

        void execute(ExecutionParameters parameters) override
        {
            if (isSkipped(parameters)) {
                return;
            }
            _resources->buildScene(parameters._renderInfo.commandBuffer, parameters._geometryBuffers->getNumObjects());
        }

    private:
        _RayTracingSceneStep(RayTracingResources const& resources)
            : _RayTracingStep(resources, true)
        {}
    };

    class _RayTracingPresentStep : public _RenderStep
    {
    public:
        static RenderStep create(RayTracingResources const& resources) { return RenderStep(new _RayTracingPresentStep(resources)); }

        void execute(ExecutionParameters parameters) override
        {
            auto commandBuffer = parameters._renderInfo.commandBuffer;
            auto& image = _resources->getOutput();
            VulkanBarriers().add(image, ImageUsage::ShaderRead).record(commandBuffer);

            prepareExecution(parameters);
            _shader->setTexture("inputTexture1", image);
            _shader->bind(commandBuffer, PipelineState{.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, .colorFormat = parameters._target->color.format});
            vkCmdDraw(commandBuffer, 3, 1, 0, 0);
            finishExecution(parameters);

            _resources->finishFrame();
        }

    private:
        _RayTracingPresentStep(RayTracingResources const& resources)
            : _RenderStep(StepParameters().shader(ShaderSources::RayTracingPresent))
            , _resources(resources)
        {}

        RayTracingResources _resources;
    };

    IntVector2D perObject(_RayTracingResources const&, NumRenderObjects const& numObjects)
    {
        return {toInt(numObjects.objects), 1};
    }

    IntVector2D perGasNode(_RayTracingResources const& resources, NumRenderObjects const&)
    {
        return resources.getGasGrid().size;
    }

    IntVector2D perPixel(_RayTracingResources const& resources, NumRenderObjects const&)
    {
        return resources.getViewSize();
    }
}

std::vector<RenderStep> createRayTracingSteps(RayTracingResources const& resources)
{
    using enum RayTracingResource;
    using enum RayTracingAccess;

    // The lights are sorted into the light grid while the acceleration structures are built
    return {
        _RayTracingFrameStep::create(resources),
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader(std::string(Shaders::RayTracingPrepareCS))
                .workGroupSize({64, 1})
                .dispatchSize([](auto const&, NumRenderObjects const& numObjects) {
                    return IntVector2D{toInt(std::max({numObjects.objects, numObjects.lineIndices / 2, numObjects.triangleIndices / 3})), 1};
                })
                .accesses({{Spheres, ReadWrite}, {SphereAabbs, ReadWrite}, {CapsuleAabbs, ReadWrite}, {TriangleAabbs, ReadWrite}, {CellCounts, Clear}})
                .requiresScene(true)),
        _RayTracingSceneStep::create(resources),
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader(std::string(Shaders::RayTracingAllocateCS))
                .workGroupSize({256, 1})
                .dispatchSize([](_RayTracingResources const& resources, auto const&) { return IntVector2D{toInt(resources.getLightGrid().getNumCells()), 1}; })
                .accesses({{CellCounts, Read}, {CellCursors, ReadWrite}, {Allocation, Clear}})
                .requiresScene(true)),
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader(std::string(Shaders::RayTracingScatterCS))
                .workGroupSize({64, 1})
                .dispatchSize(perObject)
                .accesses({{Spheres, Read}, {CellCursors, ReadWrite}, {CellEntries, ReadWrite}})
                .requiresScene(true)),
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader(std::string(Shaders::RayTracingGasSplatCS))
                .workGroupSize({64, 1})
                .dispatchSize([](auto const&, NumRenderObjects const& numObjects) { return IntVector2D{toInt(numObjects.fluidParticles), 1}; })
                .accesses({{GasAccumulation, Clear}})),
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader(std::string(Shaders::RayTracingGasLightsCS))
                .workGroupSize({64, 1})
                .dispatchSize(perObject)
                .accesses({{GasAccumulation, ReadWrite}, {GasLightClusters, Clear}})),
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader(std::string(Shaders::RayTracingGasBlurCS))
                .workGroupSize({8, 8})
                .dispatchSize(perGasNode)
                .accesses({{GasAccumulation, Read}, {GasIntermediate, ReadWrite}})),
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader("#define VERTICAL\n" + std::string(Shaders::RayTracingGasBlurCS))
                .workGroupSize({8, 8})
                .dispatchSize(perGasNode)
                .accesses({{GasIntermediate, Read}, {GasImage, ReadWrite}})),

        // The light of the cells and of the radiation sources reaches the gas through the gas and the objects
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader(std::string(Shaders::RayTracingGasIrradianceCS))
                .workGroupSize({8, 8})
                .dispatchSize([](_RayTracingResources const& resources, auto const&) { return resources.getGasGrid().irradianceSize; })
                .accesses({{Scene, Read}, {Spheres, Read}, {GasLightClusters, Read}, {GasImage, Read}, {GasIrradiance, ReadWrite}})),
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader(std::string(Shaders::RayTracingGasShadingCS) + std::string(Shaders::RayTracingTissueCS))
                .workGroupSize({8, 8})
                .dispatchSize(perPixel)
                .accesses(
                    {{Scene, Read},
                     {Spheres, Read},
                     {CellCounts, Read},
                     {CellCursors, Read},
                     {CellEntries, Read},
                     {GasImage, Read},
                     {GasIrradiance, Read},
                     {SceneColor, ReadWrite}})),
        _RayTracingComputeStep::create(
            resources,
            ComputeStepParameters()
                .shader(std::string(Shaders::RayTracingAntiAliasingCS))
                .workGroupSize({8, 8})
                .dispatchSize(perPixel)
                .accesses({{SceneColor, Read}, {PreviousAntiAliased, Read}, {AntiAliased, ReadWrite}})),
        _RayTracingPresentStep::create(resources),
    };
}
