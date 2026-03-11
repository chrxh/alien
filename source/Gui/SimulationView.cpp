#include "SimulationView.h"

#include <algorithm>

#include <glad/glad.h>

#include <imgui.h>

#include <Base/GlobalSettings.h>
#include <Base/Resources.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SpaceCalculator.h>

#include "AlienGui.h"
#include "RenderPipeline.h"
#include "RenderStep.h"
#include "Shader.h"
#include "SimulationScrollbars.h"
#include "StyleRepository.h"
#include "Viewport.h"

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

void SimulationView::resize(IntVector2D const& size)
{
    _renderPipeline->resize(size);

    Viewport::get().setViewSize(size);
}

void SimulationView::draw()
{
    if (_renderSimulation) {
        _renderPipeline->execute();

        if (_SimulationFacade::get()->getSimulationParameters().markReferenceDomain.value) {
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

void SimulationView::updateMotionBlur() {}

void SimulationView::setupRenderPipeline()
{
    // Define lambdas for render pipeline
    auto backgroundUniformFunc = [this](SimulationParameters const& parameters) {
        return UniformValueMap{
            {"background", parameters.backgroundColor.baseValue},
            {"gridLines", parameters.gridLines.value},
            {"borderlessRendering", parameters.borderlessRendering.value},
        };
    };
    auto moduloUniformFunc = [](SimulationParameters const& parameters) {
        return UniformValueMap{{"borderlessRendering", parameters.borderlessRendering.value}};
    };

    // Number of blur repetitions and blur strengths is based on zoom level to balance performance and quality
    auto blurStrengthFunc = [](SimulationParameters const& parameters) {
        auto zoom = Viewport::get().getZoomFactor();
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
        auto zoom = Viewport::get().getZoomFactor();
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
                _AttackEventRenderStep::create(StepParameters().shader(ShaderSources::AttackEvent).previousTargetSelection(0)),
                _DetonationEventRenderStep::create(StepParameters().shader(ShaderSources::DetonationEvent).previousTargetSelection(0)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(moduloUniformFunc)),
                _PostProcessingRenderStep::create(
                    StepParameters().shader(ShaderSources::BlurHorizontal).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                _PostProcessingRenderStep::create(
                    StepParameters().shader(ShaderSources::BlurVertical).addUniform("strength", 0.1f).addUniform("zoomDependent", true)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Metaballs)),
                //_PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::Fresnel)),
                //_PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::SubsurfaceScatter)),
            }),
            RenderSequence().steps({
                _ObjectRenderStep::create(StepParameters().shader(ShaderSources::Object)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ZoomBrightnessCorrection).addUniform("strength", 0.5f)),
                _PostProcessingRenderStep::create(StepParameters().shader(ShaderSources::ModuloCopy).uniformFunc(moduloUniformFunc)),
            }),
        },

        // Render block: Merge fluid, connections and objects sequence
        RenderBlock{
            RenderSequence().steps({
                _PostProcessingRenderStep::create(StepParameters()
                                                      .shader(ShaderSources::MergeAdditive)
                                                      .addUniform("colorFactor1", 1.0f)
                                                      .addUniform("colorFactor2", 2.0f)
                                                      .addUniform("colorFactor3", 1.5f)),
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
