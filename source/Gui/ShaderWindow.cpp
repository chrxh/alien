#include "ShaderWindow.h"

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "SimulationView.h"

namespace
{
    auto const MaxContentTextWidth = 140.0f;
}

_ShaderWindow::_ShaderWindow(SimulationView const& simulationView)
: _AlienWindow("Shader parameters", "windows.shader", false)
, _simulationView(simulationView)
{
    _brightness = GlobalSettings::getInstance().getFloatState("windows.shader.brightness", _brightness);
    _contrast = GlobalSettings::getInstance().getFloatState("windows.shader.contrast", _contrast);
    _motionBlur = GlobalSettings::getInstance().getFloatState("windows.shader.motion blur", _motionBlur);
    _simulationView->setBrightness(_brightness);
    _simulationView->setMotionBlur(_motionBlur);
}

_ShaderWindow::~_ShaderWindow()
{
    GlobalSettings::getInstance().setFloatState("windows.shader.brightness", _brightness);
    GlobalSettings::getInstance().setFloatState("windows.shader.contrast", _contrast);
    GlobalSettings::getInstance().setFloatState("windows.shader.motion blur", _motionBlur);
}

void _ShaderWindow::processIntern()
{
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Brightness").min(0).max(3.0f).textWidth(MaxContentTextWidth).defaultValue(1.0f), _brightness)) {
        _simulationView->setBrightness(_brightness);
    }
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Contrast").min(0).max(2.0f).textWidth(MaxContentTextWidth).defaultValue(1.0f), _contrast)) {
        _simulationView->setContrast(_contrast);
    }
    if (AlienImGui::SliderFloat(
            AlienImGui::SliderFloatParameters().name("Motion blur").min(0).max(10.0f).textWidth(MaxContentTextWidth).logarithmic(true).defaultValue(1.0f), _motionBlur)) {
        _simulationView->setMotionBlur(_motionBlur);
    }
}
