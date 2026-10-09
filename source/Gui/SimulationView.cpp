#include "SimulationView.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <imgui.h>

#include <Base/ExitScopeGuard.h>
#include <Base/GlobalSettings.h>
#include <Base/Resources.h>

#include <EngineInterface/SimulationFacade.h>

#include <RenderingInterface/RenderingFacade.h>

#include "AlienGui.h"
#include "SimulationScrollbars.h"
#include "StyleService.h"
#include "Viewport.h"

void SimulationView::setup()
{

    _cellDetailOverlayActive = GlobalSettings::get().getValue("settings.simulation view.overlay", _cellDetailOverlayActive);
    _brightness = GlobalSettings::get().getValue("windows.simulation view.brightness", _brightness);
    _contrast = GlobalSettings::get().getValue("windows.simulation view.contrast", _contrast);
    _motionBlur = GlobalSettings::get().getValue("windows.simulation view.motion blur factor", _motionBlur);

    _RenderingFacade::get()->setupSimulationRendering(StyleService::get().getDefaultFont());

    _scrollbars = std::make_shared<_SimulationScrollbars>(true);
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
    Viewport::get().setViewSize(size);
}

void SimulationView::draw()
{
    if (_renderSimulation) {
        _RenderingFacade::get()->drawSimulation(createRenderView());

        if (_SimulationFacade::get()->getSimulationParameters().markReferenceDomain.value) {
            markReferenceDomain();
        }

    } else {
        _RenderingFacade::get()->clearScreen({0, 0, 0});

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
    auto& viewport = Viewport::get();
    auto origViewSize = viewport.getViewSize();
    auto origZoomFactor = viewport.getZoomFactor();
    auto origRenderScale = viewport.getRenderScale();

    // The visible world rect equals view size / zoom factor. Thus, scaling both by the same amount keeps the
    // horizontally visible world range while the vertical range follows from the aspect ratio of the picture.
    // The render scale lets the render steps enlarge all effects with a size in pixels accordingly.
    auto renderScale = origRenderScale * toFloat(resolution.x) / toFloat(origViewSize.x);
    RenderView pictureView;
    {
        ExitScopeGuard restoreViewport([&] {
            viewport.setViewSize(origViewSize);
            viewport.setZoomFactor(origZoomFactor);
            viewport.setRenderScale(origRenderScale);
        });
        viewport.setViewSize(resolution);
        viewport.setZoomFactor(origZoomFactor * renderScale);
        viewport.setRenderScale(renderScale);
        pictureView = createRenderView();
    }
    return _RenderingFacade::get()->renderSimulationPicture(pictureView);
}

RenderView SimulationView::createRenderView() const
{
    auto& viewport = Viewport::get();
    return RenderView{
        .viewSize = viewport.getViewSize(),
        .zoomFactor = viewport.getZoomFactor(),
        .renderScale = viewport.getRenderScale(),
        .visibleWorldRect = viewport.getVisibleWorldRect(),
        .cellDetailOverlay = _cellDetailOverlayActive,
    };
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
