#include "SerializationHelperService.h"
#include "EngineInterface/SimulationController.h"

#include "Viewport.h"


DeserializedSimulation SerializationHelperService::getDeserializedSerialization(SimulationController const& simController)
{
    DeserializedSimulation result;
    result.auxiliaryData.timestep = static_cast<uint32_t>(simController->getCurrentTimestep());
    result.auxiliaryData.realTime = simController->getRealTime();
    result.auxiliaryData.zoom = Viewport::getZoomFactor();
    result.auxiliaryData.center = Viewport::getCenterInWorldPos();
    result.auxiliaryData.generalSettings = simController->getGeneralSettings();
    result.auxiliaryData.simulationParameters = simController->getSimulationParameters();
    result.statistics = simController->getStatisticsHistory().getCopiedData();
    result.mainData = simController->getClusteredSimulationData();
    return result;
}
