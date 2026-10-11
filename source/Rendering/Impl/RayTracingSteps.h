#pragma once

#include "RayTracingResources.h"
#include "RenderStep.h"

// The ray tracing renders the creatures with hardware ray tracing as living tissue under a microscope, see Shaders/RayTracingTissueCS.h, and the
// fluids as gas around it, see Shaders/RayTracingGasShadingCS.h. The radiation sources shine, and their light forms columns of light and
// shadow behind the objects on its way, see Shaders/RayTracingGasIrradianceCS.h. The inputs are the background and the energy particles, the
// output is the light in front of the background in high dynamic range, to which the background is added after the tone mapping as in the
// rasterized view, and in the alpha channel the strength of the light of the radiation sources for the rasterized objects. Each frame is
// computed completely, only still pictures average several frames for the anti-aliasing.
//
// All steps but the last compute into the shared resources and have no target. The first step prepares the frame, the last step renders the
// result into its target and finishes the frame.
std::vector<RenderStep> createRayTracingSteps(RayTracingResources const& resources);
