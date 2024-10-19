#include "ShaderWindow.h"

#include "Base/GlobalSettings.h"

#include "AlienImGui.h"
#include "SimulationView.h"

namespace
{
    auto const RightColumnWidth = 140.0f;
}

_ShaderWindow::_ShaderWindow()
: AlienWindow("Shader parameters", "windows.shader", false)
{
}

_ShaderWindow::~_ShaderWindow()
{
}

void _ShaderWindow::processIntern()
{
    auto const defaultBrightness = _SimulationView::DefaultBrightness;
    auto const defaultContrast = _SimulationView::DefaultContrast;
    auto const defaultMotionBlur = _SimulationView::DefaultMotionBlur;

    auto brightness = _SimulationView::get().getBrightness();
    auto contrast = _SimulationView::get().getContrast();
    auto motionBlur = _SimulationView::get().getMotionBlur();
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Brightness").min(0).max(3.0f).textWidth(RightColumnWidth).defaultValue(&defaultBrightness), &brightness)) {
        _SimulationView::get().setBrightness(brightness);
    }
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Contrast").min(0).max(2.0f).textWidth(RightColumnWidth).defaultValue(&defaultContrast), &contrast)) {
        _SimulationView::get().setContrast(contrast);
    }
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters()
                .name("Motion blur")
                .min(0)
                .max(10.0f)
                .textWidth(RightColumnWidth)
                .logarithmic(true)
                .defaultValue(&defaultMotionBlur),
            &motionBlur)) {
        _SimulationView::get().setMotionBlur(motionBlur);
    }
}
