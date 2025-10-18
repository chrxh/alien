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
    
    //if (_areTexturesInitialized) {
    //    glDeleteTextures(1, &_screenBackgroundTexture);
    //    _areTexturesInitialized = true;
    //}

    //// Init textures - use RGBA16F for proper floating point color handling
    //// Create screen background texture (dark blue)
    //glGenTextures(1, &_screenBackgroundTexture);
    //glBindTexture(GL_TEXTURE_2D, _screenBackgroundTexture);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //
    //// Fill with dark blue color (RGB: 0.0, 0.05, 0.15)
    //std::vector<float> darkBlueData(size.x * size.y * 4);
    //for (int i = 0; i < size.x * size.y; ++i) {
    //    darkBlueData[i * 4 + 0] = 0.0f;   // R
    //    darkBlueData[i * 4 + 1] = 0.0f;  // G
    //    darkBlueData[i * 4 + 2] = 0.15f;  // B
    //    darkBlueData[i * 4 + 3] = 1.0f;   // A
    //}
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, darkBlueData.data());

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
        if (_overlay) {
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
    _renderPipeline = std::make_shared<_RenderPipeline>(
        _simulationFacade,
        RenderBlocks{

            // Render block 1: Render foreground and background scene on different render sequences
            RenderBlock{
                RenderSequence().steps({
                    _LineRenderStep::create(StepParameters().shader(Const::LineShader)),
                    _TriangleRenderStep::create(StepParameters().shader(Const::TriangleShader).previousTargetSelection(0)),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurHorizontalShader).uniformValues({{"strength", 0.1f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurVerticalShader).uniformValues({{"strength", 0.1f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::MetaballsShader)),
                    // _PostProcessingRenderStep::create(Const::FresnelShader),
                    // _PostProcessingRenderStep::create(Const::SubsurfaceScatterShader),
                }),
                RenderSequence().steps({
                    _PointRenderStep::create(StepParameters().shader(Const::PointLargeShader)),
                }),
            },

            // Render block 2: Merge foreground and background
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::MergeShader)),
                }),
            },

            // Render block 3: Two outputs: Threshold and original
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::ThresholdShader)),
                }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(0)),
                }),
            },

            // Render block 4: Two outputs: blur in one sequence (10x times) and original
            RenderBlock{
                RenderSequence().repetitions(10).steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurHorizontalShader).uniformValues({{"strength", 1.25f}})),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::BlurVerticalShader).uniformValues({{"strength", 1.25f}})),
                    }),
                RenderSequence().steps({
                    _ForwardRenderStep::create(StepParameters().previousTargetSelection(1)),
                })
            },

            // Render block 5: Merge and tone mapping
            RenderBlock{
                RenderSequence().steps({
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::MergeBlurShader)),
                    _PostProcessingRenderStep::create(StepParameters().shader(Const::ToneMappingShader)),
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

