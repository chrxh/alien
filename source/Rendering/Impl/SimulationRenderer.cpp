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

    // The glow spreads the light of the bright regions into their surroundings. At the reference glow, the scene and its glow are added. Any
    // other glow takes its additional light from the bright regions or returns it to them, so that the brightness does not depend on the
    // glow.
    auto constexpr SceneWeight = 1.2f;
    auto constexpr ReferenceGlow = 0.3f;

    // The wide glow carries the light of bright regions into dark ones, where it appears brighter after the tone mapping. It is weakened, so
    // that the overall brightness does not increase.
    auto constexpr GlowWeight = 0.7f;

    // The tissue shows the solids darker than the cells, see Shaders/RayTracingTissueCS.h. The rasterized solids, which replace it when zoomed
    // out, match it.
    auto constexpr RayTracedSolidBrightness = 0.5f;

    // Between these screen zoom factors, the structure of the solids fades in
    auto constexpr SolidDetailZoomStart = 15.0f;
    auto constexpr SolidDetailZoomEnd = 40.0f;

    // Threads that share the work of a node of the irradiance grids, so that its light clusters and its rays to the radiation sources are
    // handled in parallel, see Shaders/RayTracingGasIrradianceCS.h. They lie in one subgroup, see VulkanContext.cpp.
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
        [this, view] { updateGeometry(view); },
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

float SimulationRenderer::getBloomSceneWeight()
{
    return SceneWeight;
}

float SimulationRenderer::getSolidDetailVisibility(RenderView const& view)
{
    auto fraction = std::clamp((view.getScreenZoomFactor() - SolidDetailZoomStart) / (SolidDetailZoomEnd - SolidDetailZoomStart), 0.0f, 1.0f);
    return fraction * fraction * (3.0f - 2.0f * fraction);
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

    updateGeometry(view);
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

void SimulationRenderer::updateGeometry(RenderView const& view)
{
    auto simulationFacade = _SimulationFacade::get();
    if (_rayTracingGraph) {
        _rayTracingLayout = RayTracingLayout::calc(view, simulationFacade->getSimulationParameters(), simulationFacade->getWorldSize());
    }
    simulationFacade->tryCopyBuffersFromCudaToRenderer(
        _geometryBuffers, {.visibleWorldRect = view.visibleWorldRect, .shadowGrid = _rayTracingLayout.shadowGrid});
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
        _rayTracingResources->prepare(view, _rayTracingLayout, numObjects, parameters);
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
    // The levels of the glow are blurred little while they are magnified again
    auto constexpr GlowUpscaleBlurStrength = 0.12f / 8;

    // Number of blur repetitions and blur strengths is based on zoom level to balance performance and quality.
    // The screen zoom factor is used so that a picture rendered at a higher resolution keeps the same appearance.
    float getGlowBlurStrength(RenderView const& view)
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
        return strength;
    }

    UniformValueMap getBlurStrength(SimulationParameters const&, RenderView const& view)
    {
        return UniformValueMap{{"strength", getGlowBlurStrength(view)}};
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

    // The steps pass their input through in these cases, see the shaders
    float getBlurRadius(UniformValueMap const& uniforms, RenderView const& view)
    {
        auto strength = std::get<float>(uniforms.at("strength"));
        return std::get<int>(uniforms.at("zoomDependent")) != 0 ? view.zoomFactor * strength : strength * view.renderScale;
    }

    bool isBlurPassThrough(UniformValueMap const& uniforms, RenderView const& view)
    {
        return getBlurRadius(uniforms, view) < 1.0f;
    }

    bool isModuloCopyPassThrough(UniformValueMap const& uniforms, RenderView const&)
    {
        return std::get<int>(uniforms.at("borderlessRendering")) == 0;
    }

    bool isGridLinesPassThrough(UniformValueMap const& uniforms, RenderView const&)
    {
        return std::get<int>(uniforms.at("gridLines")) == 0;
    }

    bool isZoomBrightnessCorrectionPassThrough(UniformValueMap const& uniforms, RenderView const& view)
    {
        return view.zoomFactor / view.renderScale * std::get<float>(uniforms.at("strength")) >= 1.0f;
    }

    bool isOrganicSurfacePassThrough(UniformValueMap const& uniforms, RenderView const&)
    {
        return std::get<float>(uniforms.at("effectStrength")) <= 0.0f;
    }

    // The levels of the glow lie around the center of their targets, each downsampling halves their extent, see Shaders/DownSamplerFS.h,
    // each upsampling doubles it. The blurs widen them by their radius, see Shaders/BlurHorizontalFS.h. Outside of the extent of a level plus
    // the margin, the targets are black, as the sampled texels are.
    VkRect2D getGlowRect(RenderView const& view, float levelScale, float margin)
    {
        auto const& size = view.viewSize;
        auto getRange = [&](int extent) {
            auto halfExtent = toFloat(extent) * levelScale / 2 + margin;
            auto begin = std::max(0, toInt(std::floor(toFloat(extent) / 2 - halfExtent)));
            auto end = std::min(extent, toInt(std::ceil(toFloat(extent) / 2 + halfExtent)));
            return std::pair(begin, end);
        };
        auto [beginX, endX] = getRange(size.x);
        auto [beginY, endY] = getRange(size.y);
        return VkRect2D{{beginX, beginY}, {toUInt32(endX - beginX), toUInt32(endY - beginY)}};
    }

    float getGlowBlurRadius(RenderView const& view, float strength)
    {
        return std::ceil(view.zoomFactor * strength);
    }

    float getLevelScale(int level)
    {
        return 1.0f / toFloat(1 << level);
    }

    std::optional<VkRect2D> getGlowFirstDownsamplingRect(UniformValueMap const&, RenderView const& view, int)
    {
        return getGlowRect(view, getLevelScale(1), 2.0f);
    }

    // The repetition k of the downscaling blurs the level k + 1 and downsamples it
    std::optional<VkRect2D> getGlowDownsamplingRect(UniformValueMap const&, RenderView const& view, int repetition)
    {
        return getGlowRect(view, getLevelScale(repetition + 2), getGlowBlurRadius(view, getGlowBlurStrength(view)) + 4.0f);
    }

    std::optional<VkRect2D> getGlowBlurRect(UniformValueMap const&, RenderView const& view, int repetition)
    {
        return getGlowRect(view, getLevelScale(repetition + 1), 2.0f * getGlowBlurRadius(view, getGlowBlurStrength(view)) + 4.0f);
    }

    // The repetition j of the upscaling magnifies the level n - j into the level n - j - 1, where n is the number of levels below the original
    std::optional<VkRect2D> getGlowUpsamplingRect(UniformValueMap const&, RenderView const& view, int repetition)
    {
        auto numLevels = getBlurRepetitions(view);
        auto margin = getGlowBlurRadius(view, getGlowBlurStrength(view)) + 4.0f;
        for ([[maybe_unused]] auto previousRepetition : std::views::iota(0, repetition + 1)) {
            margin = 2.0f * margin + 2.0f + getGlowBlurRadius(view, GlowUpscaleBlurStrength);
        }
        return getGlowRect(view, getLevelScale(numLevels - repetition - 1), margin);
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

    RenderStep createBlurStep(ShaderSources::ShaderSource const& shader, StepParameters parameters)
    {
        return _PostProcessingRenderStep::create(parameters.shader(shader).addUniform("zoomDependent", true).passThroughFunc(isBlurPassThrough));
    }

    RenderStep createModuloCopyStep()
    {
        return _PostProcessingRenderStep::create(
            StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(getModuloUniforms).passThroughFunc(isModuloCopyPassThrough));
    }

    // Inputs: glow, scene and its bright regions
    UniformValueMap getBloomMergeUniforms(SimulationParameters const& parameters, RenderView const&)
    {
        auto glow = parameters.glow.value;
        return UniformValueMap{
            {"colorFactor1", glow * GlowWeight},
            {"colorFactor2", SceneWeight},
            {"colorFactor3", (ReferenceGlow - glow) * GlowWeight},
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
    UniformValueMap getRayTracingMergeUniforms(SimulationParameters const&, RenderView const& view)
    {
        return UniformValueMap{
            {"objectGain", getObjectGain(view)},
            {"tissueVisibility", _RayTracingResources::getTissueVisibility(view)},
            {"backgroundScale", _RayTracingResources::getBackgroundScale()},
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
            {"solidDetailVisibility", SimulationRenderer::getSolidDetailVisibility(view)},
        };
    }

    // Connections, polygons, cells and attack events of the rasterized view on a black layer
    RenderSequence createObjectSequence(float solidBrightness)
    {
        return RenderSequence().steps({
            _LineRenderStep::create(StepParameters().shader(ShaderSources::Line).addUniform("solidBrightness", solidBrightness)),
            _TriangleRenderStep::create(
                StepParameters().shader(ShaderSources::Triangle).addUniform("solidBrightness", solidBrightness).previousTargetSelection(0)),
            _NonFluidObjectRenderStep::create(StepParameters()
                                                  .shader(ShaderSources::NonFluidObject)
                                                  .uniformFunc(getCellAppearanceUniforms)
                                                  .addUniform("solidBrightness", solidBrightness)
                                                  .previousTargetSelection(0)),
            _AttackEventRenderStep::create(StepParameters().shader(ShaderSources::AttackEvent).previousTargetSelection(0)),
            createModuloCopyStep(),
            createBlurStep(ShaderSources::BlurHorizontal, StepParameters().addUniform("strength", 0.1f)),
            createBlurStep(ShaderSources::BlurVertical, StepParameters().addUniform("strength", 0.1f)),
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
                                                  .addUniform("grainFrequency", 25.0f)
                                                  .passThroughFunc(isOrganicSurfacePassThrough)),
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
                    createModuloCopyStep(),
                }),
            },

            // Render block: Downscale blur for fluid particles
            RenderBlock{
                RenderSequence().repetitions(1).steps({
                    createBlurStep(ShaderSources::BlurHorizontal, StepParameters().addUniform("strength", 0.1f)),
                    createBlurStep(ShaderSources::BlurVertical, StepParameters().addUniform("strength", 0.1f)),
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
                    createBlurStep(ShaderSources::BlurHorizontal, StepParameters().addUniform("strength", 0.1f)),
                    createBlurStep(ShaderSources::BlurVertical, StepParameters().addUniform("strength", 0.1f)),
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
                    _PostProcessingRenderStep::create(StepParameters()
                                                          .shader(ShaderSources::ZoomBrightnessCorrection)
                                                          .addUniform("strength", 8.0f)
                                                          .passThroughFunc(isZoomBrightnessCorrectionPassThrough)),
                }),
            },
        };
    }

    // Detonation flashes, a blurred copy of the bright regions, the original and its bright regions remain for the merge
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

            // Render block: Three outputs: threshold at half the resolution, original and threshold
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::DownSampler).addUniform("scale", 1.0f / 2.0f).renderAreaFunc(getGlowFirstDownsamplingRect)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                })},

            // Render block: Three outputs: downscale blur, original and threshold. The threshold is not blurred before its first downsampling,
            // since the blurs of the smaller levels widen the glow far more.
            RenderBlock{
                RenderSequence()
                    .repetitions([](RenderView const& view) { return getBlurRepetitions(view) - 1; })
                    .steps({
                        createBlurStep(ShaderSources::BlurHorizontal, StepParameters().uniformFunc(getBlurStrength).renderAreaFunc(getGlowBlurRect)),
                        createBlurStep(ShaderSources::BlurVertical, StepParameters().uniformFunc(getBlurStrength).renderAreaFunc(getGlowBlurRect)),
                        _PostProcessingRenderStep::create(
                            StepParameters().shader(ShaderSources::DownSampler).addUniform("scale", 1.0f / 2.0f).renderAreaFunc(getGlowDownsamplingRect)),
                    }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(2)),
                })},

            // Render block: Three outputs: upscale blur, original and threshold
            RenderBlock{
                RenderSequence()
                    .repetitions(getBlurRepetitions)
                    .steps({
                        _PostProcessingRenderStep::create(
                            StepParameters().shader(ShaderSources::UpSampler).addUniform("scale", 2.0f).renderAreaFunc(getGlowUpsamplingRect)),
                        createBlurStep(
                            ShaderSources::BlurHorizontal,
                            StepParameters().addUniform("strength", GlowUpscaleBlurStrength).renderAreaFunc(getGlowUpsamplingRect)),
                        createBlurStep(
                            ShaderSources::BlurVertical,
                            StepParameters().addUniform("strength", GlowUpscaleBlurStrength).renderAreaFunc(getGlowUpsamplingRect)),
                    }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(2)),
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
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::GridLines).uniformFunc(getGridLinesUniforms).passThroughFunc(isGridLinesPassThrough)),
                    _SelectedObjectRenderStep::create(StepParameters().shader(ShaderSources::SelectedObject).previousTargetSelection(0)),
                    createModuloCopyStep(),
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
                createObjectSequence(1.0f),
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

    IntVector2D perShadowGridNode(_RayTracingResources const& resources, NumRenderObjects const&)
    {
        return resources.getShadowGrid().size;
    }

    IntVector2D perPixel(_RayTracingResources const& resources, NumRenderObjects const&)
    {
        return resources.getViewSize();
    }

    // The levels of an occlusion pyramid are computed one after the other, each from the level before. The pyramid of the gas grid starts with
    // the occlusion of the extracted objects, the pyramid of the shadow grid with the shadow grid itself.
    std::vector<RenderStep> createOcclusionPyramidSteps(RayTracingResources const& resources, bool shadowGrid)
    {
        using enum RayTracingResource;
        using enum RayTracingAccess;

        std::vector<RenderStep> result;
        for (auto level : std::views::iota(shadowGrid ? 1 : 0, RayTracingLayout::OcclusionLevels)) {
            RayTracingAccesses accesses{{shadowGrid ? ShadowGridImage : OcclusionImage, ReadWrite}};
            if (level == 0) {
                accesses.emplace_back(OcclusionAccumulation, Read);
            }
            result.emplace_back(_RayTracingComputeStep::create(
                resources,
                ComputeStepParameters()
                    .shader(std::format("{}#define OCCLUSION_LEVEL {}\n{}", shadowGrid ? "#define SHADOW_GRID\n" : "", level, Shaders::RayTracingOcclusionCS))
                    .workGroupSize({8, 8})
                    .dispatchSize([level, shadowGrid](_RayTracingResources const& resources, NumRenderObjects const&) {
                        auto const& size = shadowGrid ? resources.getShadowGrid().size : resources.getGasGrid().size;
                        auto scale = 1 << level;
                        return IntVector2D{(size.x + scale - 1) / scale, (size.y + scale - 1) / scale};
                    })
                    .reusedInStillPicture(true)
                    .reusedInNextFrame(true)
                    .accesses(accesses)
                    .requiresScene(!shadowGrid)));
        }
        return result;
    }

    // The ray tracing steps compute into the shared resources, see RayTracingSteps.h
    RenderSequence createRayTracingSequence(RayTracingResources const& resources)
    {
        using enum RayTracingResource;
        using enum RayTracingAccess;

        std::vector<RenderStep> steps{
            _RayTracingFrameStep::create(resources),
            _RayTracingComputeStep::create(
                resources,
                ComputeStepParameters()
                    .shader(std::string(Shaders::RayTracingPrepareCS))
                    .workGroupSize({64, 1})
                    .dispatchSize([](auto const&, NumRenderObjects const& numObjects) {
                        return IntVector2D{toInt(std::max({numObjects.objects, numObjects.lineIndices / 2, numObjects.triangleIndices / 3})), 1};
                    })
                    .reusedInStillPicture(true)
                    .accesses({
                        {Spheres, ReadWrite},
                        {SphereAabbs, ReadWrite},
                        {CapsuleAabbs, ReadWrite},
                        {TriangleAabbs, ReadWrite},
                        {ShadowCasters, ReadWrite},
                        {OcclusionAccumulation, Clear},
                    })
                    .requiresScene(true)),
            _RayTracingSceneStep::create(resources),
            _RayTracingComputeStep::create(
                resources,
                ComputeStepParameters()
                    .shader(std::string(Shaders::RayTracingGasSplatCS))
                    .workGroupSize({64, 1})
                    .dispatchSize([](auto const&, NumRenderObjects const& numObjects) { return IntVector2D{toInt(numObjects.fluidParticles), 1}; })
                    .reusedInStillPicture(true)
                    .accesses({{GasAccumulation, Clear}})),
            _RayTracingComputeStep::create(
                resources,
                ComputeStepParameters()
                    .shader(std::string(Shaders::RayTracingGasLightsCS))
                    .workGroupSize({64, 1})
                    .dispatchSize(perObject)
                    .reusedInStillPicture(true)
                    .accesses({{GasAccumulation, ReadWrite}, {GasLightClusters, Clear}})),
            _RayTracingComputeStep::create(
                resources,
                ComputeStepParameters()
                    .shader(std::string(Shaders::RayTracingGasBlurCS))
                    .workGroupSize({8, 8})
                    .dispatchSize(perGasNode)
                    .reusedInStillPicture(true)
                    .accesses({{GasAccumulation, Read}, {GasIntermediate, ReadWrite}})),
            _RayTracingComputeStep::create(
                resources,
                ComputeStepParameters()
                    .shader(std::format("#define VERTICAL\n{}", Shaders::RayTracingGasBlurCS))
                    .workGroupSize({8, 8})
                    .dispatchSize(perGasNode)
                    .reusedInStillPicture(true)
                    .accesses({{GasIntermediate, Read}, {GasImage, ReadWrite}})),
            _RayTracingComputeStep::create(
                resources,
                ComputeStepParameters()
                    .shader(std::string(Shaders::RayTracingShadowGridCS))
                    .workGroupSize({8, 8})
                    .dispatchSize(perShadowGridNode)
                    .reusedInStillPicture(true)
                    .reusedInNextFrame(true)
                    .accesses({{GasIntermediate, ReadWrite}})),
            _RayTracingComputeStep::create(
                resources,
                ComputeStepParameters()
                    .shader(std::format("#define VERTICAL\n{}", Shaders::RayTracingShadowGridCS))
                    .workGroupSize({8, 8})
                    .dispatchSize(perShadowGridNode)
                    .reusedInStillPicture(true)
                    .reusedInNextFrame(true)
                    .accesses({{GasIntermediate, Read}, {ShadowGridImage, ReadWrite}})),
        };
        std::ranges::move(createOcclusionPyramidSteps(resources, true), std::back_inserter(steps));
        std::ranges::move(createOcclusionPyramidSteps(resources, false), std::back_inserter(steps));
        steps.insert(
            steps.end(),
            {
                // The light of the cells reaches the gas through the gas and the cells in it, the light of the radiation sources through the gas
                // and the objects. Both are computed on grids of their own, the threads of a work group row share a node.
                _RayTracingComputeStep::create(
                    resources,
                    ComputeStepParameters()
                        .shader(std::string(Shaders::RayTracingGasIrradianceCS))
                        .workGroupSize({ThreadsPerIrradianceNode, 4})
                        .dispatchSize([](_RayTracingResources const& resources, auto const&) {
                            auto const& size = resources.getGasGrid().irradianceSize;
                            return IntVector2D{size.x * ThreadsPerIrradianceNode, size.y};
                        })
                        .reusedInStillPicture(true)
                        .reusedInNextFrame(true)
                        .accesses({{GasLightClusters, Read}, {GasImage, Read}, {GasIrradiance, ReadWrite}})),
                _RayTracingComputeStep::create(
                    resources,
                    ComputeStepParameters()
                        .shader(std::format("#define SOURCE_LIGHT\n{}", Shaders::RayTracingGasIrradianceCS))
                        .workGroupSize({ThreadsPerIrradianceNode, 4})
                        .dispatchSize([](_RayTracingResources const& resources, auto const&) {
                            auto const& size = resources.getGasGrid().sourceLightSize;
                            return IntVector2D{size.x * ThreadsPerIrradianceNode, size.y};
                        })
                        .reusedInStillPicture(true)
                        .reusedInNextFrame(true)
                        .accesses({
                            {Scene, Read},
                            {Spheres, Read},
                            {ShadowCasters, Read},
                            {GasImage, Read},
                            {ShadowGridImage, Read},
                            {OcclusionImage, Read},
                            {GasIrradiance, ReadWrite},
                        })),
                _RayTracingComputeStep::create(
                    resources,
                    ComputeStepParameters()
                        .shader(std::string(Shaders::RayTracingGasShadingCS) + std::string(Shaders::RayTracingTissueCS))
                        .shaderWithoutTissue(std::format("#define WITHOUT_TISSUE\n{}{}", Shaders::RayTracingGasShadingCS, Shaders::RayTracingTissueCS))
                        .workGroupSize({8, 8})
                        .dispatchSize(perPixel)
                        .accesses({{Scene, Read}, {Spheres, Read}, {GasImage, Read}, {GasIrradiance, Read}, {SceneColor, ReadWrite}})),
                _RayTracingComputeStep::create(
                    resources,
                    ComputeStepParameters()
                        .shader(std::string(Shaders::RayTracingAntiAliasingCS))
                        .workGroupSize({8, 8})
                        .dispatchSize(perPixel)
                        .antiAliasing(true)
                        .accesses({{SceneColor, Read}, {PreviousAntiAliased, Read}, {AntiAliased, ReadWrite}})),
                _RayTracingPresentStep::create(resources),
                createModuloCopyStep(),
            });
        return RenderSequence().steps(steps);
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
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(ShaderSources::GridLines).uniformFunc(getGridLinesUniforms).passThroughFunc(isGridLinesPassThrough)),
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
                    createModuloCopyStep(),
                    createBlurStep(ShaderSources::BlurHorizontal, StepParameters().addUniform("strength", 0.1f)),
                    createBlurStep(ShaderSources::BlurVertical, StepParameters().addUniform("strength", 0.1f)),
                    _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Metaballs)),
                }),
                createObjectSequence(RayTracedSolidBrightness),
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
