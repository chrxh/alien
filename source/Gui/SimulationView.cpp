#include "SimulationView.h"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <vector>
#include <span>

#include <imgui.h>

#include <Base/AlienExceptions.h>
#include <Base/ExitScopeGuard.h>
#include <Base/GlobalSettings.h>
#include <Base/Resources.h>

#include <Data/SpaceCalculator.h>

#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "RenderPipeline.h"
#include "RenderStep.h"
#include "Shader.h"
#include "SimulationScrollbars.h"
#include "StyleService.h"
#include "Viewport.h"
#include "VulkanContext.h"
#include "VulkanFrameRenderer.h"

void SimulationView::setup()
{

    _cellDetailOverlayActive = GlobalSettings::get().getValue("settings.simulation view.overlay", _cellDetailOverlayActive);
    _brightness = GlobalSettings::get().getValue("windows.simulation view.brightness", _brightness);
    _contrast = GlobalSettings::get().getValue("windows.simulation view.contrast", _contrast);
    _motionBlur = GlobalSettings::get().getValue("windows.simulation view.motion blur factor", _motionBlur);

    setupRenderPipeline();

    _scrollbars = std::make_shared<_SimulationScrollbars>(true);

    resize(Viewport::get().getViewSize());
}

void SimulationView::shutdown()
{
    GlobalSettings::get().setValue("settings.simulation view.overlay", _cellDetailOverlayActive);
    GlobalSettings::get().setValue("windows.simulation view.brightness", _brightness);
    GlobalSettings::get().setValue("windows.simulation view.contrast", _contrast);
    GlobalSettings::get().setValue("windows.simulation view.motion blur factor", _motionBlur);
}

void SimulationView::releaseGraphicsResources()
{
    _renderPipeline.reset();
}

void SimulationView::resize(IntVector2D const& size)
{
    _renderPipeline->resize(size);

    Viewport::get().setViewSize(size);
}

void SimulationView::draw()
{
    if (_renderSimulation) {
        VulkanFrameRenderer::get().drawScene(
            [this] { _renderPipeline->updateGeometry(); },
            [this](VkCommandBuffer commandBuffer) -> VulkanImage& { return _renderPipeline->execute(commandBuffer); });

        if (_SimulationFacade::get()->getSimulationParameters().markReferenceDomain.value) {
            markReferenceDomain();
        }

    } else {
        VulkanFrameRenderer::get().clearScreen({0, 0, 0});

        auto textWidth = scale(300.0f);
        auto textHeight = scale(80.0f);
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto& styleRep = StyleService::get();
        auto right = ImGui::GetMainViewport()->Pos.x + ImGui::GetMainViewport()->Size.x;
        auto bottom = ImGui::GetMainViewport()->Pos.y + ImGui::GetMainViewport()->Size.y;
        auto maxLength = std::max(right, bottom);

        AlienGui::RotateStart(drawList);
        auto font = styleRep.getReefLargeFont();
        auto text = "Rendering disabled";
        ImVec4 clipRect(-100000.0f, -100000.0f, 100000.0f, 100000.0f);
        for (int i = 0; toFloat(i) * textWidth < maxLength * 2; ++i) {
            for (int j = 0; toFloat(j) * textHeight < maxLength * 2; ++j) {
                font->RenderText(
                    drawList,
                    scale(34.0f),
                    {toFloat(i) * textWidth - maxLength / 2, toFloat(j) * textHeight - maxLength / 2},
                    Const::RenderingDisabledTextColor,
                    clipRect,
                    text,
                    text + strlen(text),
                    0.0f,
                    false);
            }
        }
        AlienGui::RotateEnd(45.0f, drawList);
    }
}

void SimulationView::processSimulationScrollbars()
{
    if (_renderSimulation) {
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        auto mainMenubarHeight = scale(22);

        auto worldCenter = Viewport::get().getCenterInWorldPos();
        auto worldRect = RealRect{{0, 0}, toRealVector2D(_SimulationFacade::get()->getWorldSize())};
        auto visibleWorldRect = Viewport::get().getVisibleWorldRect();
        auto viewRect =
            RealRect{{viewport->Pos.x, viewport->Pos.y + mainMenubarHeight}, {viewport->Pos.x + viewport->Size.x, viewport->Pos.y + viewport->Size.y}};
        _scrollbars->process(worldCenter, worldRect, visibleWorldRect, viewRect);
        Viewport::get().setCenterInWorldPos({worldCenter.x, worldCenter.y});
    }
}

bool SimulationView::isScrollbarDragging() const
{
    return _scrollbars->isHoveredOrDragged();
}

bool SimulationView::isRenderSimulation() const
{
    return _renderSimulation;
}

void SimulationView::setRenderSimulation(bool value)
{
    _renderSimulation = value;
}

bool SimulationView::isOverlayActive() const
{
    return _cellDetailOverlayActive;
}

void SimulationView::setOverlayActive(bool active)
{
    _cellDetailOverlayActive = active;
}

float SimulationView::getBrightness() const
{
    return _brightness;
}

void SimulationView::setBrightness(float value)
{
    _brightness = value;
}

float SimulationView::getContrast() const
{
    return _contrast;
}

void SimulationView::setContrast(float value)
{
    _contrast = value;
}

float SimulationView::getMotionBlur() const
{
    return _motionBlur;
}

void SimulationView::setMotionBlur(float value)
{
    _motionBlur = value;
}

PictureData SimulationView::savePicture(IntVector2D const& resolution)
{
    auto maxTextureSize = toInt(VulkanContext::get().getProperties().limits.maxImageDimension2D);
    if (resolution.x > maxTextureSize || resolution.y > maxTextureSize) {
        throw AlienException("The resolution must not exceed " + std::to_string(maxTextureSize) + " pixels per dimension on this GPU.");
    }

    try {
        return renderPicture(resolution);
    } catch (AlienException const&) {
        throw;
    } catch (std::exception const& exception) {
        throw AlienException(
            std::string("The picture could not be rendered, possibly because the GPU memory does not suffice for this resolution. ") + exception.what());
    }
}

PictureData SimulationView::renderPicture(IntVector2D const& resolution)
{
    auto& context = VulkanContext::get();

    // The rendering resources are shared with the frame that might still be in flight
    context.waitIdle();

    auto origViewSize = Viewport::get().getViewSize();
    auto origZoomFactor = Viewport::get().getZoomFactor();
    auto origRenderScale = Viewport::get().getRenderScale();

    auto target = _TextureTarget::create();
    auto readbackBuffer =
        context.createBuffer(static_cast<VkDeviceSize>(resolution.x) * resolution.y * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VulkanMemory::HostVisible);
    ExitScopeGuard restoreState([&] {
        context.destroyBuffer(readbackBuffer);

        Viewport::get().setViewSize(origViewSize);
        Viewport::get().setZoomFactor(origZoomFactor);
        Viewport::get().setRenderScale(origRenderScale);
        _renderPipeline->resize(origViewSize);
    });

    target->resize(resolution, VK_FORMAT_R8G8B8A8_UNORM);

    // The visible world rect equals view size / zoom factor. Thus, scaling both by the same amount keeps the
    // horizontally visible world range while the vertical range follows from the aspect ratio of the picture.
    // The render scale lets the render steps enlarge all effects with a size in pixels accordingly.
    auto renderScale = origRenderScale * toFloat(resolution.x) / toFloat(origViewSize.x);
    Viewport::get().setViewSize(resolution);
    Viewport::get().setZoomFactor(origZoomFactor * renderScale);
    Viewport::get().setRenderScale(renderScale);

    _renderPipeline->resize(resolution);
    _renderPipeline->updateGeometry();
    context.submitAndWait([&](VkCommandBuffer commandBuffer) {
        auto& image = _renderPipeline->execute(commandBuffer, target);
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

    // The render pipeline provides the rows bottom-up as OpenGL did
    auto bytesPerRow = static_cast<size_t>(resolution.x) * PictureData::NumChannels;
    for (auto row : std::views::iota(0, resolution.y / 2)) {
        auto upperRow = result.pixels.begin() + row * bytesPerRow;
        auto lowerRow = result.pixels.begin() + (resolution.y - 1 - row) * bytesPerRow;
        std::swap_ranges(upperRow, upperRow + bytesPerRow, lowerRow);
    }
    return result;
}

void SimulationView::setupRenderPipeline()
{
    // Define lambdas for render pipeline
    auto backgroundUniformFunc = [this](SimulationParameters const& parameters) {
        return UniformValueMap{
            {"background", parameters.backgroundColor.baseValue},
            {"borderlessRendering", parameters.borderlessRendering.value},
        };
    };
    auto gridLinesUniformFunc = [](SimulationParameters const& parameters) {
        return UniformValueMap{
            {"gridLines", parameters.gridLines.value},
        };
    };
    auto moduloUniformFunc = [](SimulationParameters const& parameters) {
        return UniformValueMap{{"borderlessRendering", parameters.borderlessRendering.value}};
    };

    // Number of blur repetitions and blur strengths is based on zoom level to balance performance and quality.
    // The screen zoom factor is used so that a picture rendered at a higher resolution keeps the same appearance.
    auto blurStrengthFunc = [](SimulationParameters const& parameters) {
        auto zoom = Viewport::get().getScreenZoomFactor();
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
    auto blurRepetitionsFunc = [] {
        auto zoom = Viewport::get().getScreenZoomFactor();
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

    auto organicFadeIn = [] { return std::clamp((Viewport::get().getScreenZoomFactor() - 4.0f) / 12.0f, 0.0f, 1.0f); };

    auto organicSurfaceUniformFunc = [organicFadeIn](SimulationParameters const&) { return UniformValueMap{{"effectStrength", organicFadeIn()}}; };

    auto cellAppearanceUniformFunc = [organicFadeIn](SimulationParameters const&) {
        auto fadeIn = organicFadeIn();
        auto dimmed = 0.54f * std::min(1.0f, Viewport::get().getScreenZoomFactor() * 0.5f);
        return UniformValueMap{
            {"brightness", std::lerp(dimmed, 1.0f, fadeIn)},
            {"sizeScale", std::lerp(0.87f, 1.0f, fadeIn)},
        };
    };
    auto objectMergeUniformFunc = [organicFadeIn](SimulationParameters const&) {
        return UniformValueMap{{"colorFactor2", std::lerp(2.0f, 1.3f, organicFadeIn())}};
    };

    // Define render pipeline
    _renderPipeline = std::make_shared<_RenderPipeline>(RenderBlocks{

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
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::MergeAdditive).uniformFunc([](SimulationParameters const& parameters) {
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
                _CellTypeOverlayRenderStep::create(StepParameters().shader(ShaderSources::CellTypeOverlay).previousTargetSelection(0)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::DeNoise)),
            }),
        },
    });
}

void SimulationView::markReferenceDomain()
{
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    auto p1 = Viewport::get().mapWorldToViewPosition({0, 0}, false);
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    auto p2 = Viewport::get().mapWorldToViewPosition(toRealVector2D(worldSize), false);
    auto color = ImColor::HSV(0.66f, 1.0f, 1.0f, 0.8f);
    auto color2 = ImColor::HSV(0, 0, 0, 0.8f);
    drawList->AddLine({p1.x, p1.y}, {p2.x, p1.y}, color);
    drawList->AddLine({p2.x, p1.y}, {p2.x, p2.y}, color);
    drawList->AddLine({p2.x, p2.y}, {p1.x, p2.y}, color);
    drawList->AddLine({p1.x, p2.y}, {p1.x, p1.y}, color);
    drawList->AddLine({p1.x - 1.0f, p1.y - 1.0f}, {p2.x + 1.0f, p1.y - 1.0f}, color2);
    drawList->AddLine({p2.x + 1.0f, p1.y - 1.0f}, {p2.x + 1.0f, p2.y + 1.0f}, color2);
    drawList->AddLine({p2.x + 1.0f, p2.y + 1.0f}, {p1.x - 1.0f, p2.y + 1.0f}, color2);
    drawList->AddLine({p1.x - 1.0f, p2.y + 1.0f}, {p1.x - 1.0f, p1.y - 1.0f}, color2);
}
