#include "SimulationRenderer.h"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <span>

#include <Base/ExitScopeGuard.h>

#include "RenderGraph.h"
#include "RenderStep.h"
#include "VulkanContext.h"
#include "VulkanFrameRenderer.h"

void SimulationRenderer::setup(ImFont* labelFont)
{
    createRenderGraph(labelFont);
}

void SimulationRenderer::shutdown()
{
    _renderGraph.reset();
}

void SimulationRenderer::draw(RenderView const& view)
{
    VulkanFrameRenderer::get().drawScene(
        [this, view] { _renderGraph->updateGeometry(view.visibleWorldRect); },
        [this, view](VkCommandBuffer commandBuffer) -> VulkanImage& { return _renderGraph->execute(commandBuffer, view); });
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

    _renderGraph->updateGeometry(view.visibleWorldRect);
    context.submitAndWait([&](VkCommandBuffer commandBuffer) {
        auto& image = _renderGraph->execute(commandBuffer, view, target);
        VulkanContext::useImage(commandBuffer, image, ImageUsage::TransferSource);
        VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {static_cast<uint32_t>(resolution.x), static_cast<uint32_t>(resolution.y), 1},
        };
        vkCmdCopyImageToBuffer(commandBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer.buffer, 1, &region);
    });

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

void SimulationRenderer::createRenderGraph(ImFont* labelFont)
{
    // Define lambdas for render graph
    auto backgroundUniformFunc = [](SimulationParameters const& parameters, RenderView const&) {
        return UniformValueMap{
            {"background", parameters.backgroundColor.baseValue},
            {"borderlessRendering", parameters.borderlessRendering.value},
        };
    };
    auto gridLinesUniformFunc = [](SimulationParameters const& parameters, RenderView const&) {
        return UniformValueMap{
            {"gridLines", parameters.gridLines.value},
        };
    };
    auto moduloUniformFunc = [](SimulationParameters const& parameters, RenderView const&) {
        return UniformValueMap{{"borderlessRendering", parameters.borderlessRendering.value}};
    };

    // Number of blur repetitions and blur strengths is based on zoom level to balance performance and quality.
    // The screen zoom factor is used so that a picture rendered at a higher resolution keeps the same appearance.
    auto blurStrengthFunc = [](SimulationParameters const&, RenderView const& view) {
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
    };
    auto blurRepetitionsFunc = [](RenderView const& view) {
        auto zoom = view.getScreenZoomFactor();
        auto result = 7;
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
    };

    auto organicFadeIn = [](RenderView const& view) { return std::clamp((view.getScreenZoomFactor() - 4.0f) / 12.0f, 0.0f, 1.0f); };

    auto organicSurfaceUniformFunc = [organicFadeIn](SimulationParameters const&, RenderView const& view) {
        return UniformValueMap{{"effectStrength", organicFadeIn(view)}};
    };

    auto cellAppearanceUniformFunc = [organicFadeIn](SimulationParameters const&, RenderView const& view) {
        auto fadeIn = organicFadeIn(view);
        auto dimmed = 0.54f * std::min(1.0f, view.getScreenZoomFactor() * 0.5f);
        return UniformValueMap{
            {"brightness", std::lerp(dimmed, 1.0f, fadeIn)},
            {"sizeScale", std::lerp(0.87f, 1.0f, fadeIn)},
        };
    };
    auto objectMergeUniformFunc = [organicFadeIn](SimulationParameters const&, RenderView const& view) {
        return UniformValueMap{{"colorFactor2", std::lerp(2.0f, 1.3f, organicFadeIn(view))}};
    };

    // Define render graph
    _renderGraph = std::make_shared<_RenderGraph>(RenderBlocks{

        // Render block: Render fluid particles
        RenderBlock{
            RenderSequence().steps({
                _FluidParticleRenderStep::create(StepParameters().shader(ShaderSources::FluidParticle).addUniform("onBackground", true)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(moduloUniformFunc)),
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
                _FluidParticleRenderStep::create(StepParameters().shader(ShaderSources::FluidParticle).addUniform("onBackground", false)),
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

        // Render block: Render objects in different sequences
        RenderBlock{
            RenderSequence().steps({
                _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
            }),
            RenderSequence().steps({
                _LineRenderStep::create(StepParameters().shader(ShaderSources::Line)),
                _TriangleRenderStep::create(StepParameters().shader(ShaderSources::Triangle).previousTargetSelection(0)),
                _NonFluidObjectRenderStep::create(
                    StepParameters().shader(ShaderSources::NonFluidObject).uniformFunc(cellAppearanceUniformFunc).previousTargetSelection(0)),
                _AttackEventRenderStep::create(StepParameters().shader(ShaderSources::AttackEvent).previousTargetSelection(0)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(moduloUniformFunc)),
                _PostProcessingRenderStep::create(
                    StepParameters().shader(ShaderSources::BlurHorizontal).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                _PostProcessingRenderStep::create(
                    StepParameters().shader(ShaderSources::BlurVertical).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Metaballs)),
                _PostProcessingRenderStep::create(StepParameters()
                                                      .shader(ShaderSources::OrganicSurface)
                                                      .uniformFunc(organicSurfaceUniformFunc)
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
            }),
        },

        // Render block: Merge fluid, connections and objects sequence
        RenderBlock{
            RenderSequence().steps({
                _PostProcessingRenderStep::create(
                    StepParameters().shader(ShaderSources::MergeAdditive).addUniform("colorFactor1", 1.0f).uniformFunc(objectMergeUniformFunc)),
            }),
        },

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
                .repetitions(blurRepetitionsFunc)
                .steps({
                    _PostProcessingRenderStep::create(StepParameters()
                                                          .shader(ShaderSources::BlurHorizontal)
                                                          .uniformFunc(blurStrengthFunc)
                                                          //.addUniform("strength", 0.12f / 8)
                                                          .addUniform("zoomDependent", true)),
                    _PostProcessingRenderStep::create(StepParameters()
                                                          .shader(ShaderSources::BlurVertical)
                                                          .uniformFunc(blurStrengthFunc)
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
                .repetitions(blurRepetitionsFunc)
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

        // Render block: Merge and tone mapping
        RenderBlock{
            RenderSequence().steps({
                _PostProcessingRenderStep::create(
                    StepParameters().shader(ShaderSources::MergeAdditive).uniformFunc([](SimulationParameters const& parameters, RenderView const&) {
                        float bloom = parameters.glow.value;
                        return UniformValueMap{
                            {"colorFactor1", bloom},
                            {"colorFactor2", 1.5f - bloom},
                        };
                    })),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ToneMapping)),
            }),
        },

        // Render block: Background
        RenderBlock{
            RenderSequence().steps({
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Background).uniformFunc(backgroundUniformFunc)),
                _LocationRenderStep::create(StepParameters().shader(ShaderSources::Location).previousTargetSelection(0)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::GridLines).uniformFunc(gridLinesUniformFunc)),
                _SelectedObjectRenderStep::create(StepParameters().shader(ShaderSources::SelectedObject).previousTargetSelection(0)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(moduloUniformFunc)),
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
    });
}
