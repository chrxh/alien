#include "SimulationView.h"

#include <algorithm>
#include <glad/glad.h>
#include <imgui.h>

#include "Base/GlobalSettings.h"
#include "Base/Resources.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/SpaceCalculator.h"

#include "AlienGui.h"
#include "RenderPipeline.h"
#include "RenderStep.h"
#include "SimulationScrollbars.h"
#include "Shader.h"
#include "Viewport.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr ZoomFactorForOverlay = 12.0f;
}

void SimulationView::setup(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;

    _cellDetailOverlayActive = GlobalSettings::get().getValue("settings.simulation view.overlay", _cellDetailOverlayActive);
    _brightness = GlobalSettings::get().getValue("windows.simulation view.brightness", _brightness);
    _contrast = GlobalSettings::get().getValue("windows.simulation view.contrast", _contrast);
    _motionBlur = GlobalSettings::get().getValue("windows.simulation view.motion blur factor", _motionBlur);

    _scrollbars = std::make_shared<_SimulationScrollbars>(true);

    // Initialize viewport size
    Viewport::get().setViewSize(Viewport::get().getViewSize());

    // Create overlay texture
    createOverlayTexture();

    // Setup render pipeline (must be after texture creation)
    setupRenderPipeline();

    // Resize everything
    resize(Viewport::get().getViewSize());
}

void SimulationView::shutdown()
{
    GlobalSettings::get().setValue("settings.simulation view.overlay", _cellDetailOverlayActive);
    GlobalSettings::get().setValue("windows.simulation view.brightness", _brightness);
    GlobalSettings::get().setValue("windows.simulation view.contrast", _contrast);
    GlobalSettings::get().setValue("windows.simulation view.motion blur factor", _motionBlur);

    // Clean up overlay texture resources
    if (_overlayTexture != 0) {
        glDeleteTextures(1, &_overlayTexture);
        _overlayTexture = 0;
    }
    if (_overlayFBO != 0) {
        glDeleteFramebuffers(1, &_overlayFBO);
        _overlayFBO = 0;
    }
}

void SimulationView::createOverlayTexture()
{
    auto viewSize = Viewport::get().getViewSize();
    
    // Delete existing texture if any
    if (_overlayTexture != 0) {
        glDeleteTextures(1, &_overlayTexture);
        glDeleteFramebuffers(1, &_overlayFBO);
    }

    // Create texture for overlay
    glGenTextures(1, &_overlayTexture);
    glBindTexture(GL_TEXTURE_2D, _overlayTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, viewSize.x, viewSize.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    // Create framebuffer
    glGenFramebuffers(1, &_overlayFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, _overlayFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _overlayTexture, 0);

    // Check framebuffer status
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        // Log error or handle failure
    }

    // Restore default framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SimulationView::renderOverlayToTexture()
{
    // Only render if overlay is active, zoom is sufficient, and overlay data exists
    if (!_cellDetailOverlayActive || Viewport::get().getZoomFactor() <= ZoomFactorForOverlay || !_overlay) {
        // Clear texture to transparent when not active
        if (_overlayTexture != 0) {
            glBindFramebuffer(GL_FRAMEBUFFER, _overlayFBO);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        return;
    }

    // Ensure texture is created
    if (_overlayTexture == 0) {
        createOverlayTexture();
    }

    auto viewSize = Viewport::get().getViewSize();

    // Save current framebuffer
    GLint currentFbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFbo);

    // Bind our framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, _overlayFBO);
    glViewport(0, 0, viewSize.x, viewSize.y);

    // Clear with transparent background
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // For now, render simple colored rectangles where text would go
    // TODO: Implement proper ImGui text rendering to texture
    // This demonstrates the infrastructure is working
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    auto parameters = _simulationFacade->getSimulationParameters();
    
    for (auto const& overlayElement : _overlay->elements) {
        if (overlayElement.cell) {
            auto viewPos = Viewport::get().mapWorldToViewPosition(
                {overlayElement.pos.x, overlayElement.pos.y + 0.3f}, 
                parameters.borderlessRendering.value);
            
            // Draw a simple colored quad as placeholder
            // In a full implementation, this would render actual text
            float quadSize = 20.0f;
            float x1 = viewPos.x - quadSize;
            float y1 = viewPos.y - quadSize/2;
            float x2 = viewPos.x + quadSize;
            float y2 = viewPos.y + quadSize/2;
            
            // Convert to NDC
            float ndcX1 = (x1 / viewSize.x) * 2.0f - 1.0f;
            float ndcY1 = 1.0f - (y1 / viewSize.y) * 2.0f;
            float ndcX2 = (x2 / viewSize.x) * 2.0f - 1.0f;
            float ndcY2 = 1.0f - (y2 / viewSize.y) * 2.0f;
            
            glBegin(GL_QUADS);
            glColor4f(1.0f, 1.0f, 1.0f, 0.8f);
            glVertex2f(ndcX1, ndcY1);
            glVertex2f(ndcX2, ndcY1);
            glVertex2f(ndcX2, ndcY2);
            glVertex2f(ndcX1, ndcY2);
            glEnd();
        }
    }
    
    glDisable(GL_BLEND);

    // Restore framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, currentFbo);
}

void SimulationView::resize(IntVector2D const& size)
{
    _renderPipeline->resize(size);
    
    Viewport::get().setViewSize(size);

    // Recreate overlay texture with new size
    if (_overlayTexture != 0) {
        createOverlayTexture();
    }
}

void SimulationView::draw()
{
    if (_renderSimulation) {
        // Render overlay to texture before pipeline execution
        renderOverlayToTexture();

        _renderPipeline->execute();

        if (_simulationFacade->getSimulationParameters().markReferenceDomain.value) {
            markReferenceDomain();
        }

        // Draw selected cell circles (keep this separate from cell type overlay)
        if (_overlay) {
            ImDrawList* drawList = ImGui::GetBackgroundDrawList();
            auto parameters = _simulationFacade->getSimulationParameters();
            for (auto const& overlayElement : _overlay->elements) {
                if (overlayElement.selected == 1) {
                    auto viewPos = Viewport::get().mapWorldToViewPosition({overlayElement.pos.x, overlayElement.pos.y}, parameters.borderlessRendering.value);
                    if (Viewport::get().isVisible(viewPos)) {
                        drawList->AddCircle({viewPos.x, viewPos.y}, Viewport::get().getZoomFactor() * 0.45f, Const::SelectedCellOverlayColor, 0, 2.0f);
                    }
                }
            }
        }
    } else {
        glClearColor(0, 0, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        auto textWidth = scale(300.0f);
        auto textHeight = scale(80.0f);
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto& styleRep = StyleRepository::get();
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
        auto worldRect = RealRect{{0,0}, toRealVector2D(_simulationFacade->getWorldSize())};
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

void SimulationView::updateMotionBlur()
{
}

void SimulationView::setupRenderPipeline()
{
    auto currentBackgroundColor = [this] {
        auto params = _simulationFacade->getSimulationParameters();
        FloatColorRGB background = params.backgroundColor.baseValue;
        int gridLines = params.gridLines.value ? 1 : 0;
        return UniformValueMap{{"background", background}, {"gridLines", gridLines}};
    };
    _renderPipeline = std::make_shared<_RenderPipeline>(
        _simulationFacade,
        RenderBlocks{

            // Render block: Render energy particles
            RenderBlock{
                RenderSequence().steps({
                    _EnergyParticleRenderStep::create(
                        StepParameters().shader(Const::EnergyParticleShader).uniforms({{"ballSize", 2.0f}}).preventMoirePatterns(false)/*.previousTargetSelection(0)*/),
                }),
            },

            // Render block: Downscale blur for energy particles
            RenderBlock{
                RenderSequence().repetitions(4).steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BlurHorizontalShader).uniforms({{"strength", 0.1f}})),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BlurVerticalShader).uniforms({{"strength", 0.1f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::DownSamplerShader).textureScale(0.5f)),
                }),
                RenderSequence().steps({
                    _EnergyParticleRenderStep::create(StepParameters().shader(Const::EnergyParticleShader).uniforms({{"ballSize", 0.2f}})),
                }),
            },

            // Render block: Upscale blur for energy particles
            RenderBlock{
                RenderSequence().repetitions(4).steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::UpSamplerShader).textureScale(2.0f)),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BlurHorizontalShader).uniforms({{"strength", 0.1f}})),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BlurVerticalShader).uniforms({{"strength", 0.1f}})),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                }),
            },

            // Render block: Zoom brightness correction for energy particles
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::ZoomBrightnessCorrectionShader).uniforms({{"strength", 0.5f}})),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                }),
            },

            // Render block: Merge (bloom) energy particles
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::MergeMaxShader).uniforms({{"colorFactor1", 0.8f}})),
                }),
            },

            // Render block: Render cells in different sequences
            RenderBlock{
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
                RenderSequence().steps({
                    _LineRenderStep::create(StepParameters().shader(Const::LineShader)),
                    _TriangleRenderStep::create(StepParameters().shader(Const::TriangleShader).previousTargetSelection(0)),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurHorizontalShader).uniforms({{"strength", 0.1f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurVerticalShader).uniforms({{"strength", 0.1f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::MetaballsShader)),
                    //_PostProcessingRenderStep::create(StepParameters().shader(Const::FresnelShader)),
                    //_PostProcessingRenderStep::create(StepParameters().shader(Const::SubsurfaceScatterShader)),
                }),
                RenderSequence().steps({
                    _CellRenderStep::create(StepParameters().shader(Const::CellLargeShader).previousTargetSelection(0)),
                }),
            },

            // Render block: Merge energy, blur cells and cells
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters()
                                                          .shader(Const::MergeAdditiveShader)
                                                          .uniforms({{"colorFactor1", 1.0f}, {"colorFactor2", 0.6f}, {"colorFactor3", 0.5f}})),
                }),
            },

            // Render block: Two outputs: Threshold and original
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::ThresholdShader)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
            },

            // Render block: Two outputs: downscale blur and original
            RenderBlock{
                RenderSequence().repetitions(6).steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BlurHorizontalShader).uniforms({{"strength", 0.25f}})),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BlurVerticalShader).uniforms({{"strength", 0.25f}}).textureScale(1.0f / 1.5f)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                })},

            // Render block: Two outputs: upscale blur and original
            RenderBlock{
                RenderSequence().repetitions(6).steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BlurHorizontalShader).uniforms({{"strength", 0.25f}})),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BlurVerticalShader).uniforms({{"strength", 0.25f}}).textureScale(1.5f)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                })},

            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::ZoomBrightnessCorrectionShader).uniforms({{"strength", 0.5f}})),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                }),
            },

            // Render block: Merge and tone mapping
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::MergeAdditiveShader).uniforms({{"colorFactor1", 0.5f}, {"colorFactor2", 1.0f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::ZoomBrightnessCorrectionShader).uniforms({{"strength", 1.0f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::ToneMappingShader)),
                }),
            },

            // Render block: Background
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BackgroundShader).uniformFunc(currentBackgroundColor)),
                    _LocationRenderStep::create(StepParameters().shader(Const::LocationShader).previousTargetSelection(0)),
                    _SelectedCellRenderStep::create(StepParameters().shader(Const::SelectedCellShader).previousTargetSelection(0)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
            },

            // Render block: Merge background and foreground
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::MergeAdditiveShader).uniforms({{"colorFactor1", 1.0f}, {"colorFactor2", 1.0f}})),
                }),
            },

            // Render block: Cell type overlay (only active when zoom > ZoomFactorForOverlay and overlay is active)
            RenderBlock{
                RenderSequence().steps({
                    _CellTypeOverlayRenderStep::create(
                        StepParameters()
                            .shader(Const::CellTypeOverlayShader)
                            .previousTargetSelection(0)
                            .inputTextures({_overlayTexture})
                            .uniforms({})),
                }),
            },
        });
}

void SimulationView::markReferenceDomain()
{
    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    auto p1 = Viewport::get().mapWorldToViewPosition({0, 0}, false);
    auto worldSize = _simulationFacade->getWorldSize();
    auto p2 = Viewport::get().mapWorldToViewPosition(toRealVector2D(worldSize), false);
    auto color = ImColor::HSV(0.66f, 1.0f, 1.0f, 0.8f);
    drawList->AddLine({p1.x, p1.y}, {p2.x, p1.y}, color);
    drawList->AddLine({p2.x, p1.y}, {p2.x, p2.y}, color);
    drawList->AddLine({p2.x, p2.y}, {p1.x, p2.y}, color);
    drawList->AddLine({p1.x, p2.y}, {p1.x, p1.y}, color);
}

