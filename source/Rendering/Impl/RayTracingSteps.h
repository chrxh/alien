#pragma once

#include <functional>

#include "RayTracingResources.h"
#include "RenderSteps.h"

// The ray tracing renders the creatures with hardware ray tracing as living tissue under a microscope, see Shaders/RayTracingTissueCS.h, and the
// fluids as gas around it, see Shaders/RayTracingGasShadingCS.h. The radiation sources shine, and their light forms columns of light and
// shadow behind the objects on its way, see Shaders/RayTracingGasIrradianceCS.h. The inputs are the background and the energy particles, the
// output is the light in front of the background in high dynamic range, to which the background is added after the tone mapping as in the
// rasterized view, and in the alpha channel the strength of the light of the radiation sources for the rasterized objects. Each frame is
// computed completely, only still pictures average several frames for the anti-aliasing.
//
// All steps but the last compute into the shared resources and have no target. The first step prepares the frame, the last step renders the
// result into its target and finishes the frame.

// Steps that compute into the shared resources. Steps that work on the scene of the objects are skipped while it is not needed.
class _RayTracingStep : public _RenderStep
{
public:
    bool hasTarget() const override { return false; }

protected:
    _RayTracingStep(RayTracingResources const& resources, bool requiresScene);

    bool isSkipped(ExecutionParameters const& parameters) const;

    RayTracingResources _resources;
    bool _requiresScene = false;
};

class _RayTracingFrameStep : public _RayTracingStep
{
public:
    static RayTracingFrameStep create(RayTracingResources const& resources);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _RayTracingFrameStep(RayTracingResources const& resources);
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
    static RayTracingComputeStep create(RayTracingResources const& resources, ComputeStepParameters const& parameters);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _RayTracingComputeStep(RayTracingResources const& resources, ComputeStepParameters const& parameters);

    RayTracingComputePipeline _pipeline;
    DispatchSizeFunc _dispatchSize;
    RayTracingAccesses _accesses;
};

// Builds the acceleration structures from the boxes of the objects
class _RayTracingSceneStep : public _RayTracingStep
{
public:
    static RayTracingSceneStep create(RayTracingResources const& resources);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _RayTracingSceneStep(RayTracingResources const& resources);
};

class _RayTracingPresentStep : public _RenderStep
{
public:
    static RayTracingPresentStep create(RayTracingResources const& resources);

protected:
    void execute(ExecutionParameters parameters) override;

private:
    _RayTracingPresentStep(RayTracingResources const& resources);

    RayTracingResources _resources;
};
