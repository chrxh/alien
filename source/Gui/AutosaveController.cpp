#include "AutosaveController.h"

#include <imgui.h>

#include "Base/Resources.h"
#include "Base/GlobalSettings.h"
#include "EngineInterface/SerializerService.h"
#include "EngineInterface/SimulationController.h"

#include "Viewport.h"
#include "DelayedExecutionController.h"
#include "OverlayMessageController.h"

_AutosaveController::_AutosaveController(SimulationController const& simController)
    : _simController(simController)
{
    _startTimePoint = std::chrono::steady_clock::now();
    _on = GlobalSettings::getInstance().getBool("controllers.auto save.active", true);
}

_AutosaveController::~_AutosaveController()
{
    GlobalSettings::getInstance().setBool("controllers.auto save.active", _on);
}

void _AutosaveController::shutdown()
{
    if (!_on) {
        return;
    }
    onSave();
}

bool _AutosaveController::isOn() const
{
    return _on;
}

void _AutosaveController::setOn(bool value)
{
    _on = value;
}

void _AutosaveController::process()
{
    if (!_on) {
        return;
    }

    auto durationSinceStart = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - *_startTimePoint).count();
    if (durationSinceStart > 0 && durationSinceStart % 20 == 0 && !_alreadySaved) {
        printOverlayMessage("Auto saving ...");
        delayedExecution([=, this] { onSave(); });
        _alreadySaved = true;
    }
    if (durationSinceStart > 0 && durationSinceStart % 20 == 1 && _alreadySaved) {
        _alreadySaved = false;
    }
}

void _AutosaveController::onSave()
{
    DeserializedSimulation sim;
    sim.auxiliaryData.timestep = _simController->getCurrentTimestep();
    sim.auxiliaryData.zoom = Viewport::getZoomFactor();
    sim.auxiliaryData.center = Viewport::getCenterInWorldPos();
    sim.auxiliaryData.generalSettings = _simController->getGeneralSettings();
    sim.auxiliaryData.simulationParameters = _simController->getSimulationParameters();
    sim.mainData = _simController->getClusteredSimulationData();
    sim.statistics = _simController->getStatisticsHistory().getCopiedData();
    SerializerService::serializeSimulationToFiles(Const::AutosaveFile, sim);
}
