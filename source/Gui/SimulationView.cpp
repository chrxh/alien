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
#include "Shader.h"
#include "SimulationScrollbars.h"
#include "StyleRepository.h"
#include "Viewport.h"

void SimulationView::setup(SimulationFacade const& simulationFacade)
{
    _simulationFacade = simulationFacade;

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
        auto worldRect = RealRect{{0, 0}, toRealVector2D(_simulationFacade->getWorldSize())};
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

void SimulationView::updateMotionBlur() {}

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
                    _EnergyParticleRenderStep::create(StepParameters()
                                                          .shader(Const::EnergyParticleShader)
                                                          .uniforms({{"ballSize", 2.0f}})
                                                          .preventMoirePatterns(false) /*.previousTargetSelection(0)*/),
                }),
            },

            // Render block: Downscale blur for energy particles
            RenderBlock{
                RenderSequence().repetitions(4).steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurHorizontalShader).uniforms({{"strength", 0.1f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurVerticalShader).uniforms({{"strength", 0.1f}})),
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
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurHorizontalShader).uniforms({{"strength", 0.1f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurVerticalShader).uniforms({{"strength", 0.1f}})),
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
                    _CellRenderStep::create(StepParameters().shader(Const::CellShader).previousTargetSelection(0)),
                }),
            },

            // Render block: Merge energy, blur cells and cells
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::MergeAdditiveShader).uniforms({{"colorFactor1", 1.0f}, {"colorFactor2", 0.6f}, {"colorFactor3", 0.5f}})),
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
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurHorizontalShader).uniforms({{"strength", 0.25f}})),
                    _PostProcessingRenderStep::create(
                        StepParameters().shader(Const::BlurVerticalShader).uniforms({{"strength", 0.25f}}).textureScale(1.0f / 1.5f)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                })},

            // Render block: Two outputs: upscale blur and original
            RenderBlock{
                RenderSequence().repetitions(6).steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurHorizontalShader).uniforms({{"strength", 0.25f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurVerticalShader).uniforms({{"strength", 0.25f}}).textureScale(1.5f)),
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
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BackgroundShader).uniformFunc(currentBackgroundColor)),
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
                    _CellTypeOverlayRenderStep::create(
                        StepParameters().shader(Const::CellTypeOverlayShader).previousTargetSelection(0)),
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