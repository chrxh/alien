#include "SimulationRenderer.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <ranges>
#include <format>
#include <span>

#include <Base/ExitScopeGuard.h>
#include <Base/LoggingService.h>

#include <Engine/Interface/SimulationFacade.h>

#include "RayTracingResources.h"
#include "RayTracingSteps.h"
#include "RenderGraph.h"
#include "RenderSteps.h"
#include "VulkanContext.h"
#include "VulkanFrameRenderer.h"

namespace
{
    // Ray traced pictures average this many frames with jittered rays for the anti-aliasing
    auto constexpr NumRayTracedPictureFrames = 16;

    // The wide glow carries the light of bright regions into dark ones, where it appears brighter after the tone mapping. It is weakened, so
    // that the overall brightness does not increase.
    auto constexpr GlowWeight = 0.7f;

    // Threads that share the work of a node of the irradiance grid, so that its rays to the radiation sources run in parallel, see
    // Shaders/RayTracingGasIrradianceCS.h. They lie in one subgroup, see VulkanContext.cpp.
    auto constexpr ThreadsPerIrradianceNode = 16;
}

void SimulationRenderer::setup(ImFont* labelFont)
{
    _geometryBuffers = _VulkanGeometryBuffers::create();
    createRasterGraph(labelFont);
    if (VulkanContext::get().isRayTracingSupported()) {
        createRayTracingGraph(labelFont);
    }
}

void SimulationRenderer::shutdown()
{
    _rayTracingResources.reset();
    _rayTracingGraph.reset();
    _rasterGraph.reset();
    _geometryBuffers.reset();
}

void SimulationRenderer::draw(RenderView const& view)
{
    VulkanFrameRenderer::get().drawScene(
        [this, view] { updateGeometry(view.visibleWorldRect); },
        [this, view](VkCommandBuffer commandBuffer) -> VulkanImage& {
            auto parameters = _SimulationFacade::get()->getSimulationParameters();
            return selectGraph(view, parameters, false)->execute(commandBuffer, view, parameters);
        });
}

PictureData SimulationRenderer::renderPicture(RenderView const& view)
{
    auto maxTextureSize = toInt(VulkanContext::get().getProperties().limits.maxImageDimension2D);
    if (view.viewSize.x > maxTextureSize || view.viewSize.y > maxTextureSize) {
        throw AlienException("The resolution must not exceed " + std::to_string(maxTextureSize) + " pixels per dimension on this GPU.");
    }

    try {
        return renderPictureInternal(view);
    } catch (AlienException const&) {
        throw;
    } catch (std::exception const& exception) {
        throw AlienException(
            std::string("The picture could not be rendered, possibly because the GPU memory does not suffice for this resolution. ") + exception.what());
    }
}

float SimulationRenderer::getBloomSceneWeight(SimulationParameters const& parameters)
{
    return 1.5f - parameters.glow.value;
}

PictureData SimulationRenderer::renderPictureInternal(RenderView const& view)
{
    auto& context = VulkanContext::get();

    // The rendering resources are shared with the frame that might still be in flight
    context.waitIdle();

    auto const& resolution = view.viewSize;
    auto target = _TextureTarget::create();
    auto readbackBuffer =
        context.createBuffer(static_cast<VkDeviceSize>(resolution.x) * resolution.y * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VulkanMemory::HostVisible);
    ExitScopeGuard destroyReadbackBuffer([&] { context.destroyBuffer(readbackBuffer); });

    target->resize(resolution, VK_FORMAT_R8G8B8A8_UNORM);

    updateGeometry(view.visibleWorldRect);
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto const& graph = selectGraph(view, parameters, true);
    auto rayTraced = graph == _rayTracingGraph;
    if (rayTraced) {
        _rayTracingResources->setStillPicture(true);
    }
    ExitScopeGuard resetStillPicture([&] {
        if (rayTraced) {
            _rayTracingResources->setStillPicture(false);
        }
    });
    try {
        if (rayTraced) {
            for ([[maybe_unused]] auto frame : std::views::iota(1, NumRayTracedPictureFrames)) {
                context.submitAndWait([&](VkCommandBuffer commandBuffer) { graph->execute(commandBuffer, view, parameters, target); });
            }
        }
        context.submitAndWait([&](VkCommandBuffer commandBuffer) {
            auto& image = graph->execute(commandBuffer, view, parameters, target);
            VulkanContext::useImage(commandBuffer, image, ImageUsage::TransferSource);
            VkBufferImageCopy region{
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {static_cast<uint32_t>(resolution.x), static_cast<uint32_t>(resolution.y), 1},
            };
            vkCmdCopyImageToBuffer(commandBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer.buffer, 1, &region);
        });
    } catch (...) {
        graph->resetImageStates();
        if (rayTraced) {
            _rayTracingResources->resetImageStates();
        }
        throw;
    }

    PictureData result{.resolution = resolution, .pixels = std::vector<uint8_t>(static_cast<size_t>(resolution.x) * resolution.y * PictureData::NumChannels)};
    auto rgbaPixels = std::span(static_cast<uint8_t const*>(readbackBuffer.mapped), static_cast<size_t>(resolution.x) * resolution.y * 4);
    for (auto const& [rgba, rgb] : std::views::zip(rgbaPixels | std::views::chunk(4), result.pixels | std::views::chunk(PictureData::NumChannels))) {
        std::ranges::copy(rgba | std::views::take(PictureData::NumChannels), rgb.begin());
    }

    // The render graph provides the rows bottom-up as OpenGL did
    auto bytesPerRow = static_cast<size_t>(resolution.x) * PictureData::NumChannels;
    for (auto row : std::views::iota(0, resolution.y / 2)) {
        auto upperRow = result.pixels.begin() + row * bytesPerRow;
        auto lowerRow = result.pixels.begin() + (resolution.y - 1 - row) * bytesPerRow;
        std::swap_ranges(upperRow, upperRow + bytesPerRow, lowerRow);
    }
    return result;
}

void SimulationRenderer::updateGeometry(RealRect const& visibleWorldRect)
{
    _SimulationFacade::get()->tryCopyBuffersFromCudaToRenderer(_geometryBuffers, visibleWorldRect);
}

RenderGraph const& SimulationRenderer::selectGraph(RenderView const& view, SimulationParameters const& parameters, bool forPicture)
{
    if (!_rayTracingGraph) {
        return _rasterGraph;
    }
    auto numObjects = _geometryBuffers->getNumObjects();
    RayTracingDemand demand{
        .viewSize = view.viewSize,
        .numSceneObjects = _RayTracingResources::isSceneNeeded(view, parameters) ? numObjects.objects : 0,
    };
    if (!forPicture && _failedRayTracingDemand && _failedRayTracingDemand->viewSize == demand.viewSize
        && demand.numSceneObjects * 3 >= _failedRayTracingDemand->numSceneObjects * 2) {
        return _rasterGraph;
    }
    try {
        _rayTracingResources->prepare(view, numObjects, parameters);
        if (!forPicture) {
            _failedRayTracingDemand.reset();
        }
        return _rayTracingGraph;
    } catch (std::exception const& exception) {
        log(Priority::Important, std::string("ray tracing is not possible, the simulation is rasterized instead: ") + exception.what());
        if (!forPicture) {
            _failedRayTracingDemand = demand;
        }
        return _rasterGraph;
    }
}

namespace
{
    // Number of blur repetitions and blur strengths is based on zoom level to balance performance and quality.
    // The screen zoom factor is used so that a picture rendered at a higher resolution keeps the same appearance.
    UniformValueMap getBlurStrength(SimulationParameters const&, RenderView const& view)
    {
        auto zoom = view.getScreenZoomFactor();
        float strength = 0.035f;
        if (zoom < 100.0f) {
            strength *= 3.5f;
        }
        if (zoom < 50.0f) {
            strength *= 2.5f;
        }
        if (zoom < 25.0f) {
            strength *= 2.5f;
        }
        if (zoom < 12.0f) {
            strength *= 2.5f;
        }
        if (zoom < 6.0f) {
            strength *= 2.5f;
        }
        return UniformValueMap{{"strength", strength}};
    }

    int getBlurRepetitions(RenderView const& view)
    {
        auto zoom = view.getScreenZoomFactor();
        auto result = 9;
        if (zoom < 100.0f) {
            --result;
        }
        if (zoom < 50.0f) {
            --result;
        }
        if (zoom < 25.0f) {
            --result;
        }
        if (zoom < 12.0f) {
            --result;
        }
        if (zoom < 6.0f) {
            --result;
        }
        return result;
    }

    UniformValueMap getBackgroundUniforms(SimulationParameters const& parameters, RenderView const&)
    {
        return UniformValueMap{
            {"background", parameters.backgroundColor.baseValue},
            {"borderlessRendering", parameters.borderlessRendering.value},
        };
    }

    UniformValueMap getGridLinesUniforms(SimulationParameters const& parameters, RenderView const&)
    {
        return UniformValueMap{
            {"gridLines", parameters.gridLines.value},
        };
    }

    UniformValueMap getModuloUniforms(SimulationParameters const& parameters, RenderView const&)
    {
        return UniformValueMap{{"borderlessRendering", parameters.borderlessRendering.value}};
    }

    UniformValueMap getBloomMergeUniforms(SimulationParameters const& parameters, RenderView const&)
    {
        float bloom = parameters.glow.value;
        return UniformValueMap{
            {"colorFactor1", bloom * GlowWeight},
            {"colorFactor2", SimulationRenderer::getBloomSceneWeight(parameters)},
        };
    }

    // Zoomed in, the rasterized objects gain an organic surface
    float getOrganicFadeIn(RenderView const& view)
    {
        return std::clamp((view.getScreenZoomFactor() - 4.0f) / 12.0f, 0.0f, 1.0f);
    }

    // Zoomed out, the objects are added brighter to the layer below
    float getObjectGain(RenderView const& view)
    {
        return std::lerp(2.0f, 1.3f, getOrganicFadeIn(view));
    }

    UniformValueMap getObjectMergeUniforms(SimulationParameters const&, RenderView const& view)
    {
        return UniformValueMap{{"colorFactor2", getObjectGain(view)}};
    }

    // Zoomed out, the rasterized objects replace the ray traced tissue and its attack events
    UniformValueMap getRayTracingMergeUniforms(SimulationParameters const& parameters, RenderView const& view)
    {
        return UniformValueMap{
            {"objectGain", getObjectGain(view)},
            {"tissueVisibility", _RayTracingResources::getTissueVisibility(view)},
            {"backgroundScale", _RayTracingResources::getBackgroundScale(parameters)},
        };
    }

    UniformValueMap getOrganicSurfaceUniforms(SimulationParameters const&, RenderView const& view)
    {
        return UniformValueMap{{"effectStrength", getOrganicFadeIn(view)}};
    }

    UniformValueMap getCellAppearanceUniforms(SimulationParameters const&, RenderView const& view)
    {
        auto fadeIn = getOrganicFadeIn(view);
        auto dimmed = 0.54f * std::min(1.0f, view.getScreenZoomFactor() * 0.5f);
        return UniformValueMap{
            {"brightness", std::lerp(dimmed, 1.0f, fadeIn)},
            {"sizeScale", std::lerp(0.87f, 1.0f, fadeIn)},
        };
    }

    // Connections, polygons, cells and attack events of the rasterized view on a black layer
    RenderSequence createObjectSequence()
    {
        return RenderSequence().steps({
            _LineRenderStep::create(StepParameters().shader(ShaderSources::Line)),
            _TriangleRenderStep::create(StepParameters().shader(ShaderSources::Triangle).previousTargetSelection(0)),
            _NonFluidObjectRenderStep::create(
                StepParameters().shader(ShaderSources::NonFluidObject).uniformFunc(getCellAppearanceUniforms).previousTargetSelection(0)),
            _AttackEventRenderStep::create(StepParameters().shader(ShaderSources::AttackEvent).previousTargetSelection(0)),
            _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(getModuloUniforms)),
            _PostProcessingRenderStep::create(
                StepParameters().shader(ShaderSources::BlurHorizontal).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
            _PostProcessingRenderStep::create(
                StepParameters().shader(ShaderSources::BlurVertical).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
            _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Metaballs)),
            _PostProcessingRenderStep::create(StepParameters()
                                                  .shader(ShaderSources::OrganicSurface)
                                                  .uniformFunc(getOrganicSurfaceUniforms)
                                                  .addUniform("warpStrength", 0.0f)
                                                  .addUniform("warpFrequency", 3.5f)
                                                  .addUniform("smoothingStrength", 0.3f)
                                                  .addUniform("depthStrength", 0.5f)
                                                  .addUniform("roundingStrength", 0.35f)
                                                  .addUniform("reliefStrength", 4.0f)
                                                  .addUniform("shadingStrength", 0.8f)
                                                  .addUniform("membraneStrength", 0.15f)
                                                  .addUniform("scatterStrength", 0.08f)
                                                  .addUniform("specularStrength", 0.08f)
                                                  .addUniform("cavityStrength", 0.5f)
                                                  .addUniform("grainStrength", 0.0f)
                                                  .addUniform("grainFrequency", 25.0f)),
            //_PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Fresnel)),
            //_PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::SubsurfaceScatter)),
        });
    }

    void append(RenderBlocks& blocks, RenderBlocks&& otherBlocks)
    {
        std::ranges::move(otherBlocks, std::back_inserter(blocks));
    }

    // Blurred and merged into a glowing fluid layer
    RenderBlocks createFluidParticleBlocks(bool energyParticlesOnly)
    {
        return RenderBlocks{

            // Render block: Render fluid particles
            RenderBlock{
                RenderSequence().steps({
                    _FluidParticleRenderStep::create(StepParameters()
                                                         .shader(ShaderSources::FluidParticle)
                                                         .addUniform("onBackground", true)
                                                         .addUniform("energyParticlesOnly", energyParticlesOnly)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(getModuloUniforms)),
                }),
            },

            // Render block: Downscale blur for fluid particles
            RenderBlock{
                RenderSequence().repetitions(1).steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::BlurHorizontal).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::BlurVertical).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::DownSampler).addUniform("scale", 0.5f)),
                }),
                RenderSequence().steps({
                    _FluidParticleRenderStep::create(StepParameters()
                                                         .shader(ShaderSources::FluidParticle)
                                                         .addUniform("onBackground", false)
                                                         .addUniform("energyParticlesOnly", energyParticlesOnly)),
                }),
            },

            // Render block: Upscale blur for fluid particles
            RenderBlock{
                RenderSequence().repetitions(1).steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::UpSampler).addUniform("scale", 2.0f)),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::BlurHorizontal).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::BlurVertical).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Metaballs)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                }),
            },

            // Render block: Merge fluid particles for bloom
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::MergeMax).addUniform("colorFactor1", 0.8f)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ZoomBrightnessCorrection).addUniform("strength", 8.0f)),
                }),
            },
        };
    }

    // Detonation flashes, a blurred copy of the bright regions and the original remain for the merge
    RenderBlocks createDetonationAndBloomBlocks()
    {
        return RenderBlocks{

            // Render block: Two outputs: Detonation flashes lighting the scene and the scene
            RenderBlock{
                RenderSequence().steps({
                    _DetonationEventRenderStep::create(StepParameters().shader(ShaderSources::DetonationEvent)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
            },

            // Render block: Add detonation flashes to the scene
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::MergeAdditive).addUniform("colorFactor1", 1.0f).addUniform("colorFactor2", 1.0f)),
                }),
            },

            // Render block: Two outputs: Threshold and original
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Threshold)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
            },

            // Render block: Two outputs: downscale blur and original
            RenderBlock{
                RenderSequence()
                    .repetitions(getBlurRepetitions)
                    .steps({
                        _PostProcessingRenderStep::create(StepParameters()
                                                              .shader(ShaderSources::BlurHorizontal)
                                                              .uniformFunc(getBlurStrength)
                                                              //.addUniform("strength", 0.12f / 8)
                                                              .addUniform("zoomDependent", true)),
                        _PostProcessingRenderStep::create(StepParameters()
                                                              .shader(ShaderSources::BlurVertical)
                                                              .uniformFunc(getBlurStrength)
                                                              //.addUniform("strength", 0.12f / 8)
                                                              .addUniform("zoomDependent", true)),
                        _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::DownSampler).addUniform("scale", 1.0f / 2.0f)),
                    }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                })},

            // Render block: Two outputs: upscale blur and original
            RenderBlock{
                RenderSequence()
                    .repetitions(getBlurRepetitions)
                    .steps({
                        _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::UpSampler).addUniform("scale", 2.0f)),
                        _PostProcessingRenderStep::create(
                            StepParameters().shader(ShaderSources::BlurHorizontal).addUniform("strength", 0.12f / 8).addUniform("zoomDependent", true)),
                        _PostProcessingRenderStep::create(
                            StepParameters().shader(ShaderSources::BlurVertical).addUniform("strength", 0.12f / 8).addUniform("zoomDependent", true)),
                    }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                })},
        };
    }

    // The background is added after the bloom and the tone mapping, so that it keeps its color and does not glow
    RenderBlocks createToneMappingAndBackgroundBlocks(ImFont* labelFont)
    {
        return RenderBlocks{

            // Render block: Merge and tone mapping
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::MergeAdditive).uniformFunc(getBloomMergeUniforms)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ToneMapping)),
                }),
            },

            // Render block: Background
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Background).uniformFunc(getBackgroundUniforms)),
                    _LocationRenderStep::create(StepParameters().shader(ShaderSources::Location).previousTargetSelection(0)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::GridLines).uniformFunc(getGridLinesUniforms)),
                    _SelectedObjectRenderStep::create(StepParameters().shader(ShaderSources::SelectedObject).previousTargetSelection(0)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(getModuloUniforms)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
            },

            // Render block: Merge background and foreground
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::MergeAdditive).addUniform("colorFactor1", 1.0f).addUniform("colorFactor2", 1.0f)),
                    _SelectedConnectionRenderStep::create(StepParameters().shader(ShaderSources::SelectedConnection).previousTargetSelection(0)),
                    _CellTypeOverlayRenderStep::create(StepParameters().shader(ShaderSources::CellTypeOverlay).previousTargetSelection(0), labelFont),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::DeNoise)),
                }),
            },
        };
    }
}

void SimulationRenderer::createRasterGraph(ImFont* labelFont)
{
    auto blocks = createFluidParticleBlocks(false);
    append(
        blocks,
        RenderBlocks{

            // Render block: Render objects in different sequences
            RenderBlock{
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
                createObjectSequence(),
            },

            // Render block: Merge fluid, connections and objects sequence
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::MergeAdditive).addUniform("colorFactor1", 1.0f).uniformFunc(getObjectMergeUniforms)),
                }),
            },
        });
    append(blocks, createDetonationAndBloomBlocks());
    append(blocks, createToneMappingAndBackgroundBlocks(labelFont));
    _rasterGraph = std::make_shared<_RenderGraph>(std::move(blocks), _geometryBuffers);
}

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

    // The ray tracing steps compute into the shared resources, see RayTracingSteps.h. The lights are sorted into the light grid while the
    // acceleration structures are built.
    RenderSequence createRayTracingSequence(RayTracingResources const& resources)
    {
        using enum RayTracingResource;
        using enum RayTracingAccess;

        return RenderSequence().steps({
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
                    .dispatchSize(
                        [](_RayTracingResources const& resources, auto const&) { return IntVector2D{toInt(resources.getLightGrid().getNumCells()), 1}; })
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
                    .shader(std::format("#define VERTICAL\n{}", Shaders::RayTracingGasBlurCS))
                    .workGroupSize({8, 8})
                    .dispatchSize(perGasNode)
                    .accesses({{GasIntermediate, Read}, {GasImage, ReadWrite}})),

            // The light of the cells and of the radiation sources reaches the gas through the gas and the objects. The threads of a work group
            // row share a node, so that the many rays of the nodes close to large sources spread over the GPU.
            _RayTracingComputeStep::create(
                resources,
                ComputeStepParameters()
                    .shader(std::string(Shaders::RayTracingGasIrradianceCS))
                    .workGroupSize({ThreadsPerIrradianceNode, 4})
                    .dispatchSize([](_RayTracingResources const& resources, auto const&) {
                        auto const& size = resources.getGasGrid().irradianceSize;
                        return IntVector2D{size.x * ThreadsPerIrradianceNode, size.y};
                    })
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
            _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(getModuloUniforms)),
        });
    }
}

void SimulationRenderer::createRayTracingGraph(ImFont* labelFont)
{
    _rayTracingResources = _RayTracingResources::create();

    // The ray tracing renders the fluids itself as gas, the energy particles and the cores of glowing fluids keep the look of the rasterized view
    auto blocks = createFluidParticleBlocks(true);
    append(
        blocks,
        RenderBlocks{

            // Render block: Two outputs: The background as ground and the energy particles
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Background).uniformFunc(getBackgroundUniforms)),
                    _LocationRenderStep::create(StepParameters().shader(ShaderSources::Location).previousTargetSelection(0)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::GridLines).uniformFunc(getGridLinesUniforms)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
            },

            // Render block: Four outputs: Ray tracing of the objects on the ground, its attack events softened as in the rasterized view, the
            // rasterized objects, which replace the ray tracing when zoomed out, and the ground, which they absorb like the tissue
            RenderBlock{
                createRayTracingSequence(_rayTracingResources),
                RenderSequence().steps({
                    _AttackEventRenderStep::create(StepParameters().shader(ShaderSources::AttackEvent)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(getModuloUniforms)),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::BlurHorizontal).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::BlurVertical).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Metaballs)),
                }),
                createObjectSequence(),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
            },

            // Render block: Merge ray tracing, attack events and rasterized objects
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::RayTracingMerge).uniformFunc(getRayTracingMergeUniforms)),
                }),
            },
        });
    append(blocks, createDetonationAndBloomBlocks());
    append(blocks, createToneMappingAndBackgroundBlocks(labelFont));
    _rayTracingGraph = std::make_shared<_RenderGraph>(std::move(blocks), _geometryBuffers);
}
