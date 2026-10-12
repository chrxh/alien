#pragma once

#include "RayTracingResources.h"
#include "RenderStep.h"

// The ray tracing renders the creatures with hardware ray tracing as living tissue under a microscope, see Shaders/RayTracingTissueCS.h, and the
// fluids as gas around it, see Shaders/RayTracingGasShadingCS.h. Regions with a high energy density radiate light, which forms columns of light
// and shadow behind the objects on its way, see Shaders/RayTracingGasIrradianceCS.h. The inputs are the background and the energy particles, the
// output is the light in front of the background in high dynamic range, to which the background is added after the tone mapping as in the
// rasterized view, and in the alpha channel the strength of the energy light for the rasterized objects. Each frame is computed completely, only
// still pictures average several frames for the anti-aliasing.
//
// All steps but the last compute into the shared resources and have no target. The first step prepares the frame, the last step renders the
// result into its target and finishes the frame.
std::vector<RenderStep> createRayTracingSteps(RayTracingResources const& resources);

using RayTracingConditionFunc = std::function<bool(SimulationParameters const&, RenderView const&)>;

// Steps of the ray tracing that compute into the shared resources and can depend on a condition
class _RayTracingStep : public _RenderStep
{
public:
    bool hasTarget() const override;
    bool isActive(SimulationParameters const& parameters, RenderView const& view) const override;

protected:
    _RayTracingStep(RayTracingResources const& resources, RayTracingConditionFunc const& condition = {});

    RayTracingResources _resources;
    RayTracingConditionFunc _condition;
};

class _RayTracingFrameStep : public _RayTracingStep
{
public:
    static RenderStep create(RayTracingResources const& resources);

    void resetImageStates() override;

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _RayTracingFrameStep(RayTracingResources const& resources);
};

using RayTracingDispatchSizeFunc = std::function<IntVector2D(_RayTracingResources const&, NumRenderObjects const&)>;

struct RayTracingComputeStepParameters
{
    // Appended to Shaders/RayTracingCommonCS.h
    MEMBER(RayTracingComputeStepParameters, std::string, shader, std::string());

    // Has to match the local size declared in the shader
    MEMBER(RayTracingComputeStepParameters, IntVector2D, workGroupSize, IntVector2D());

    // At least one invocation per element
    MEMBER(RayTracingComputeStepParameters, RayTracingDispatchSizeFunc, dispatchSize, RayTracingDispatchSizeFunc());

    // The barriers in between the steps follow from these accesses
    MEMBER(RayTracingComputeStepParameters, RayTracingAccesses, accesses, RayTracingAccesses());

    MEMBER(RayTracingComputeStepParameters, RayTracingConditionFunc, condition, RayTracingConditionFunc());
};

class _RayTracingComputeStep : public _RayTracingStep
{
public:
    static RenderStep create(RayTracingResources const& resources, RayTracingComputeStepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _RayTracingComputeStep(RayTracingResources const& resources, RayTracingComputeStepParameters const& parameters);

    RayTracingComputePipeline _pipeline;
    RayTracingDispatchSizeFunc _dispatchSize;
    RayTracingAccesses _accesses;
};

// Builds the acceleration structures from the boxes of the objects
class _RayTracingSceneStep : public _RayTracingStep
{
public:
    static RenderStep create(RayTracingResources const& resources, RayTracingConditionFunc const& condition);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _RayTracingSceneStep(RayTracingResources const& resources, RayTracingConditionFunc const& condition);
};

class _RayTracingPresentStep : public _RenderStep
{
public:
    static RenderStep create(RayTracingResources const& resources);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _RayTracingPresentStep(RayTracingResources const& resources);

    RayTracingResources _resources;
};
