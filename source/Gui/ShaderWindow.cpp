#include "ShaderWindow.h"

#include "Base/GlobalSettings.h"

#include "AlienImGui.h"
#include "SimulationView.h"

namespace
{
    auto const RightColumnWidth = 140.0f;
}

_ShaderWindow::_ShaderWindow(SimulationView const& simulationView)
: _AlienWindow("Shader parameters", "windows.shader", false)
, _simulationView(simulationView)
{
}

_ShaderWindow::~_ShaderWindow()
{
}

void _ShaderWindow::processIntern()
{
    auto const defaultBrightness = 1.0f;
    auto const defaultContrast = 1.0f;
    auto const defaultMotionBlur = 0.5f;

    auto brightness = _simulationView->getBrightness();
    auto contrast = _simulationView->getContrast();
    auto motionBlur = _simulationView->getMotionBlur();
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Brightness").min(0).max(3.0f).textWidth(RightColumnWidth).defaultValue(&defaultBrightness), &brightness)) {
        _simulationView->setBrightness(brightness);
    }
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Contrast").min(0).max(2.0f).textWidth(RightColumnWidth).defaultValue(&defaultContrast), &contrast)) {
        _simulationView->setContrast(contrast);
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
        _simulationView->setMotionBlur(motionBlur);
    }
}
