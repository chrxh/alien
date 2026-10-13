#include "RayTracingSteps.h"

#include <Rendering/Impl/Shaders/ShaderSources.h>

#include "VulkanContext.h"

_RayTracingStep::_RayTracingStep(RayTracingResources const& resources, bool requiresScene)
    : _RenderStep(StepParameters())
    , _resources(resources)
    , _requiresScene(requiresScene)
{}

bool _RayTracingStep::isSkipped(ExecutionParameters const& parameters) const
{
    return _requiresScene && !_RayTracingResources::isSceneNeeded(parameters._view, *parameters._simulationParameters);
}

RayTracingFrameStep _RayTracingFrameStep::create(RayTracingResources const& resources)
{
    return RayTracingFrameStep(new _RayTracingFrameStep(resources));
}

void _RayTracingFrameStep::execute(ExecutionParameters parameters)
{
    _resources->beginFrame(parameters);
}

_RayTracingFrameStep::_RayTracingFrameStep(RayTracingResources const& resources)
    : _RayTracingStep(resources, false)
{}

RayTracingComputeStep _RayTracingComputeStep::create(RayTracingResources const& resources, ComputeStepParameters const& parameters)
{
    return RayTracingComputeStep(new _RayTracingComputeStep(resources, parameters));
}

void _RayTracingComputeStep::execute(ExecutionParameters parameters)
{
    if (isSkipped(parameters)) {
        return;
    }
    auto commandBuffer = parameters._renderInfo.commandBuffer;
    _resources->access(commandBuffer, _accesses);
    _resources->bindDescriptorSet(commandBuffer);
    _pipeline->dispatch(commandBuffer, _dispatchSize(*_resources, parameters._geometryBuffers->getNumObjects()));
}

_RayTracingComputeStep::_RayTracingComputeStep(RayTracingResources const& resources, ComputeStepParameters const& parameters)
    : _RayTracingStep(resources, parameters._requiresScene)
    , _pipeline(resources->createPipeline(parameters._shader, parameters._workGroupSize))
    , _dispatchSize(parameters._dispatchSize)
    , _accesses(parameters._accesses)
{}

RayTracingSceneStep _RayTracingSceneStep::create(RayTracingResources const& resources)
{
    return RayTracingSceneStep(new _RayTracingSceneStep(resources));
}

void _RayTracingSceneStep::execute(ExecutionParameters parameters)
{
    if (isSkipped(parameters)) {
        return;
    }
    _resources->buildScene(parameters._renderInfo.commandBuffer, parameters._geometryBuffers->getNumObjects());
}

_RayTracingSceneStep::_RayTracingSceneStep(RayTracingResources const& resources)
    : _RayTracingStep(resources, true)
{}

RayTracingPresentStep _RayTracingPresentStep::create(RayTracingResources const& resources)
{
    return RayTracingPresentStep(new _RayTracingPresentStep(resources));
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

    _resources->finishFrame();
}

_RayTracingPresentStep::_RayTracingPresentStep(RayTracingResources const& resources)
    : _RenderStep(StepParameters().shader(ShaderSources::RayTracingPresent))
    , _resources(resources)
{}
