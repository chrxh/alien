#include "AutosaveController.h"

#include "imgui.h"

#include "EngineInterface/Serializer.h"
#include "Resources.h"
#include "GlobalSettings.h"

_AutosaveController::_AutosaveController(SimulationController const& simController)
    : _simController(simController)
{
    _startTimePoint = std::chrono::steady_clock::now();
    _on = GlobalSettings::getInstance().getBoolState("controllers.auto save.active", true);
}

_AutosaveController::~_AutosaveController()
{
    GlobalSettings::getInstance().setBoolState("controllers.auto save.active", _on);
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

    auto durationSinceStart =
        std::chrono::duration_cast<std::chrono::minutes>(std::chrono::steady_clock::now() - *_startTimePoint).count();
    if (durationSinceStart > 0 && durationSinceStart % 20 == 0) {
        onSave();
    }
}

void _AutosaveController::onSave()
{
    DeserializedSimulation sim;
    sim.timestep = static_cast<uint32_t>(_simController->getCurrentTimestep());
    sim.settings = _simController->getSettings();
    sim.symbolMap = _simController->getSymbolMap();
    sim.content = _simController->getSimulationData(
        {-1000, -1000}, {_simController->getWorldSize().x + 1000, _simController->getWorldSize().y + 1000});
    Serializer serializer = boost::make_shared<_Serializer>();
    auto serializedSim = serializer->serializeSimulation(sim);
    serializer->saveSimulationDataToFile(Const::AutosaveFile, serializedSim);
}
