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

    // Create cell type texture atlas before setting up render pipeline
    createCellTypeTextureAtlas();

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

    // Clean up cell type texture atlas
    if (_cellTypeTextureAtlas != 0) {
        glDeleteTextures(1, &_cellTypeTextureAtlas);
        _cellTypeTextureAtlas = 0;
    }
}

void SimulationView::createCellTypeTextureAtlas()
{
    // Create a texture atlas containing all cell type strings
    // We'll arrange them in a vertical strip, one per row
    
    auto font = StyleRepository::get().getMediumFont();
    float fontSize = 40.0f; // Base font size for rendering
    
    // Calculate dimensions for each cell type label
    int maxWidth = 0;
    int totalHeight = 0;
    std::vector<ImVec2> textSizes;
    
    for (auto const& cellTypeStr : Const::CellTypeStrings) {
        auto textSize = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, cellTypeStr.c_str());
        textSizes.push_back(textSize);
        maxWidth = std::max(maxWidth, toInt(textSize.x) + 10); // Add padding
        totalHeight += toInt(textSize.y) + 5; // Add padding between rows
    }
    
    // Ensure power-of-2 dimensions for better GPU compatibility
    int textureWidth = 256;  // Fixed width
    int textureHeight = 256; // Fixed height, should be enough for all labels
    
    // Create pixel buffer
    std::vector<uint8_t> pixels(textureWidth * textureHeight * 4, 0); // RGBA
    
    // Render each cell type string to the buffer
    // Note: This is a simplified approach. For production, we'd use ImGui's font
    // rendering or a proper text rendering library. For now, we'll create
    // a simple colored texture to demonstrate the infrastructure.
    
    int yOffset = 0;
    for (size_t i = 0; i < Const::CellTypeStrings.size(); ++i) {
        int rowHeight = textureHeight / toInt(Const::CellTypeStrings.size());
        int startY = toInt(i) * rowHeight;
        
        // Fill this row with a gradient (placeholder for actual text rendering)
        for (int y = startY; y < startY + rowHeight - 2; ++y) {
            for (int x = 0; x < textureWidth; ++x) {
                int idx = (y * textureWidth + x) * 4;
                // Create a simple gradient pattern as placeholder
                pixels[idx + 0] = 255;  // R
                pixels[idx + 1] = 255;  // G
                pixels[idx + 2] = 255;  // B
                pixels[idx + 3] = static_cast<uint8_t>(255.0f * (toFloat(x) / toFloat(textureWidth))); // A (gradient)
            }
        }
    }
    
    // Create OpenGL texture
    glGenTextures(1, &_cellTypeTextureAtlas);
    glBindTexture(GL_TEXTURE_2D, _cellTypeTextureAtlas);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    
    glBindTexture(GL_TEXTURE_2D, 0);
}

void SimulationView::resize(IntVector2D const& size)
{
    _renderPipeline->resize(size);
    
    Viewport::get().setViewSize(size);
}

void SimulationView::draw()
{
    if (_renderSimulation) {
        _renderPipeline->execute();

        if (_simulationFacade->getSimulationParameters().markReferenceDomain.value) {
            markReferenceDomain();
        }

        // Draw overlay if activated
        if (_overlay && Viewport::get().getZoomFactor() > ZoomFactorForOverlay) {
            ImDrawList* drawList = ImGui::GetBackgroundDrawList();
            auto parameters = _simulationFacade->getSimulationParameters();
            for (auto const& overlayElement : _overlay->elements) {
                if (_cellDetailOverlayActive && overlayElement.cell) {
                    {
                        auto fontSizeUnit = std::min(scale(40.0f), Viewport::get().getZoomFactor()) / 2;
                        auto viewPos =
                            Viewport::get().mapWorldToViewPosition({overlayElement.pos.x, overlayElement.pos.y + 0.3f}, parameters.borderlessRendering.value);
                        auto text = Const::CellTypeStrings.at(overlayElement.cellType);
                        drawList->AddText(
                            StyleRepository::get().getMediumFont(),
                            fontSizeUnit,
                            {viewPos.x - 1.7f * fontSizeUnit, viewPos.y},
                            Const::CellTypeOverlayShadowColor,
                            text.c_str());
                        drawList->AddText(
                            StyleRepository::get().getMediumFont(),
                            fontSizeUnit,
                            {viewPos.x - 1.7f * fontSizeUnit + 1, viewPos.y + 1},
                            Const::CellTypeOverlayColor,
                            text.c_str());
                    }
                }

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
                            .inputTextures({_cellTypeTextureAtlas})
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

