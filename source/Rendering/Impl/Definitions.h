#pragma once

#include <Base/Interface/Definitions.h>

#include <Rendering/Interface/Definitions.h>

class _Shader;
using Shader = std::shared_ptr<_Shader>;

class _RenderGraph;
using RenderGraph = std::shared_ptr<_RenderGraph>;

class _RenderStep;
using RenderStep = std::shared_ptr<_RenderStep>;

class _NonFluidObjectRenderStep;
using CellRenderStep = std::shared_ptr<_NonFluidObjectRenderStep>;

class _LineRenderStep;
using LineRenderStep = std::shared_ptr<_LineRenderStep>;

class _TriangleRenderStep;
using TriangleRenderStep = std::shared_ptr<_TriangleRenderStep>;

class _PostProcessingRenderStep;
using PostProcessingRenderStep = std::shared_ptr<_PostProcessingRenderStep>;

class _ForwardRenderStep;
using ForwardRenderStep = std::shared_ptr<_ForwardRenderStep>;

class _FluidParticleRenderStep;
using FluidParticleRenderStep = std::shared_ptr<_FluidParticleRenderStep>;

class _LocationRenderStep;
using LocationRenderStep = std::shared_ptr<_LocationRenderStep>;

class _SelectedObjectRenderStep;
using SelectedObjectRenderStep = std::shared_ptr<_SelectedObjectRenderStep>;

class _CellTypeOverlayRenderStep;
using CellTypeOverlayRenderStep = std::shared_ptr<_CellTypeOverlayRenderStep>;

class _SelectedConnectionRenderStep;
using SelectedConnectionRenderStep = std::shared_ptr<_SelectedConnectionRenderStep>;

class _AttackEventRenderStep;
using AttackEventRenderStep = std::shared_ptr<_AttackEventRenderStep>;

class _DetonationEventRenderStep;
using DetonationEventRenderStep = std::shared_ptr<_DetonationEventRenderStep>;

class _TextureTarget;
using TextureTarget = std::shared_ptr<_TextureTarget>;

struct GeneralRenderInfo;
