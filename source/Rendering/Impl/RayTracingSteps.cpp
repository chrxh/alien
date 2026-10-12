#include "RayTracingSteps.h"

#include <algorithm>

#include <Rendering/Impl/Shaders/ShaderSources.h>

#include "VulkanContext.h"

namespace
{
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
    auto isSceneNeeded = [](SimulationParameters const& parameters, RenderView const& view) { return _RayTracingResources::isSceneNeeded(view, parameters); };
    auto isEnergyLightActive = [](SimulationParameters const& parameters, RenderView const&) { return _RayTracingResources::isEnergyLightActive(parameters); };

    // The lights are sorted into the light grid while the acceleration structures are built
    return {
        _RayTracingFrameStep::create(resources),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader(std::string(Shaders::RayTracingPrepareCS))
                .workGroupSize({64, 1})
                .dispatchSize([](auto const&, NumRenderObjects const& numObjects) {
                    return IntVector2D{toInt(std::max({numObjects.objects, numObjects.lineIndices / 2, numObjects.triangleIndices / 3})), 1};
                })
                .accesses({{Spheres, ReadWrite}, {SphereAabbs, ReadWrite}, {CapsuleAabbs, ReadWrite}, {TriangleAabbs, ReadWrite}, {CellCounts, Clear}})
                .condition(isSceneNeeded)),
        _RayTracingSceneStep::create(resources, isSceneNeeded),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader(std::string(Shaders::RayTracingAllocateCS))
                .workGroupSize({256, 1})
                .dispatchSize([](_RayTracingResources const& resources, auto const&) { return IntVector2D{toInt(resources.getLightGrid().getNumCells()), 1}; })
                .accesses({{CellCounts, Read}, {CellOffsets, ReadWrite}, {CellCursors, ReadWrite}, {Allocation, Clear}})
                .condition(isSceneNeeded)),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader(std::string(Shaders::RayTracingScatterCS))
                .workGroupSize({64, 1})
                .dispatchSize(perObject)
                .accesses({{Spheres, Read}, {CellCursors, ReadWrite}, {CellEntries, ReadWrite}})
                .condition(isSceneNeeded)),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader(std::string(Shaders::RayTracingGasSplatCS))
                .workGroupSize({64, 1})
                .dispatchSize([](auto const&, NumRenderObjects const& numObjects) { return IntVector2D{toInt(numObjects.fluidParticles), 1}; })
                .accesses({{GasAccumulation, Clear}})),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader(std::string(Shaders::RayTracingEnergyGridCS))
                .workGroupSize({64, 1})
                .dispatchSize(perObject)
                .accesses({{EnergyCells, Clear}})
                .condition(isEnergyLightActive)),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader(std::string(Shaders::RayTracingGasLightsCS))
                .workGroupSize({64, 1})
                .dispatchSize(perObject)
                .accesses({{GasAccumulation, Read}, {EnergyCells, Read}, {GasLightClusters, Clear}})),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader("const ivec2 BlurDirection = ivec2(1, 0);\n" + std::string(Shaders::RayTracingGasBlurCS))
                .workGroupSize({8, 8})
                .dispatchSize(perGasNode)
                .accesses({{GasAccumulation, Read}, {GasIntermediate, ReadWrite}})),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader("#define VERTICAL\nconst ivec2 BlurDirection = ivec2(0, 1);\n" + std::string(Shaders::RayTracingGasBlurCS))
                .workGroupSize({8, 8})
                .dispatchSize(perGasNode)
                .accesses({{GasIntermediate, Read}, {GasImage, ReadWrite}})),

        // The light reaches the gas through the gas and the cells
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader(std::string(Shaders::RayTracingGasIrradianceCS))
                .workGroupSize({8, 8})
                .dispatchSize([](_RayTracingResources const& resources, auto const&) { return resources.getGasGrid().irradianceSize; })
                .accesses({{Scene, Read}, {Spheres, Read}, {GasLightClusters, Read}, {EnergyCells, Read}, {GasImage, Read}, {GasIrradiance, ReadWrite}})),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader(std::string(Shaders::RayTracingGasShadingCS) + std::string(Shaders::RayTracingTissueCS))
                .workGroupSize({8, 8})
                .dispatchSize(perPixel)
                .accesses(
                    {{Scene, Read},
                     {Spheres, Read},
                     {CellCounts, Read},
                     {CellOffsets, Read},
                     {CellEntries, Read},
                     {GasImage, Read},
                     {GasIrradiance, Read},
                     {SceneColor, ReadWrite}})),
        _RayTracingComputeStep::create(
            resources,
            RayTracingComputeStepParameters()
                .shader(std::string(Shaders::RayTracingAntiAliasingCS))
                .workGroupSize({8, 8})
                .dispatchSize(perPixel)
                .accesses({{SceneColor, Read}, {PreviousAntiAliased, Read}, {AntiAliased, ReadWrite}})),
        _RayTracingPresentStep::create(resources),
    };
}

bool _RayTracingStep::hasTarget() const
{
    return false;
}

bool _RayTracingStep::isActive(SimulationParameters const& parameters, RenderView const& view) const
{
    return !_condition || _condition(parameters, view);
}

_RayTracingStep::_RayTracingStep(RayTracingResources const& resources, RayTracingConditionFunc const& condition)
    : _RenderStep(StepParameters())
    , _resources(resources)
    , _condition(condition)
{}

RenderStep _RayTracingFrameStep::create(RayTracingResources const& resources)
{
    return RenderStep(new _RayTracingFrameStep(resources));
}

void _RayTracingFrameStep::resetImageStates()
{
    _resources->resetImageStates();
}

void _RayTracingFrameStep::execute(ExecutionParameters parameters)
{
    _resources->beginFrame(parameters);
}

_RayTracingFrameStep::_RayTracingFrameStep(RayTracingResources const& resources)
    : _RayTracingStep(resources)
{}

RenderStep _RayTracingComputeStep::create(RayTracingResources const& resources, RayTracingComputeStepParameters const& parameters)
{
    return RenderStep(new _RayTracingComputeStep(resources, parameters));
}

void _RayTracingComputeStep::execute(ExecutionParameters parameters)
{
    auto commandBuffer = parameters._renderInfo.commandBuffer;
    _resources->access(commandBuffer, _accesses);
    _resources->bindDescriptorSet(commandBuffer);
    _pipeline->dispatch(commandBuffer, _dispatchSize(*_resources, parameters._geometryBuffers->getNumObjects()));
}

_RayTracingComputeStep::_RayTracingComputeStep(RayTracingResources const& resources, RayTracingComputeStepParameters const& parameters)
    : _RayTracingStep(resources, parameters._condition)
    , _pipeline(resources->createPipeline(parameters._shader, parameters._workGroupSize))
    , _dispatchSize(parameters._dispatchSize)
    , _accesses(parameters._accesses)
{}

RenderStep _RayTracingSceneStep::create(RayTracingResources const& resources, RayTracingConditionFunc const& condition)
{
    return RenderStep(new _RayTracingSceneStep(resources, condition));
}

void _RayTracingSceneStep::execute(ExecutionParameters parameters)
{
    _resources->buildScene(parameters._renderInfo.commandBuffer, parameters._geometryBuffers->getNumObjects());
}

_RayTracingSceneStep::_RayTracingSceneStep(RayTracingResources const& resources, RayTracingConditionFunc const& condition)
    : _RayTracingStep(resources, condition)
{}

RenderStep _RayTracingPresentStep::create(RayTracingResources const& resources)
{
    return RenderStep(new _RayTracingPresentStep(resources));
}

void _RayTracingPresentStep::execute(ExecutionParameters parameters)
{
    auto commandBuffer = parameters._renderInfo.commandBuffer;
    auto& image = _resources->getOutput();
    VulkanBarriers().add(image, ImageUsage::ShaderRead).record(commandBuffer);

    prepareExecution(parameters);
    _shader->setTexture("inputTexture1", image);
    _shader->bind(commandBuffer, PipelineState{.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, .colorFormat = parameters._target->color.format});
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    finishExecution(parameters);

    _resources->finishFrame(parameters._view);
}

_RayTracingPresentStep::_RayTracingPresentStep(RayTracingResources const& resources)
    : _RenderStep(StepParameters().shader(ShaderSources::RayTracingPresent))
    , _resources(resources)
{}
