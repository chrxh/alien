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
    _brightness = GlobalSettings::getInstance().getFloat("windows.shader.brightness", _brightness);
    _contrast = GlobalSettings::getInstance().getFloat("windows.shader.contrast", _contrast);
    _motionBlur = GlobalSettings::getInstance().getFloat("windows.shader.motion blur", _motionBlur);
    _simulationView->setBrightness(_brightness);
    _simulationView->setContrast(_contrast);
    _simulationView->setMotionBlur(_motionBlur);
}

_ShaderWindow::~_ShaderWindow()
{
    GlobalSettings::getInstance().setFloat("windows.shader.brightness", _brightness);
    GlobalSettings::getInstance().setFloat("windows.shader.contrast", _contrast);
    GlobalSettings::getInstance().setFloat("windows.shader.motion blur", _motionBlur);
}

void _ShaderWindow::processIntern()
{
    auto const defaultValue = 1.0f;
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Brightness").min(0).max(3.0f).textWidth(RightColumnWidth).defaultValue(&defaultValue), &_brightness)) {
        _simulationView->setBrightness(_brightness);
    }
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Contrast").min(0).max(2.0f).textWidth(RightColumnWidth).defaultValue(&defaultValue), &_contrast)) {
        _simulationView->setContrast(_contrast);
    }
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Motion blur").min(0).max(10.0f).textWidth(RightColumnWidth).logarithmic(true).defaultValue(&defaultValue),
            &_motionBlur)) {
        _simulationView->setMotionBlur(_motionBlur);
    }
}
