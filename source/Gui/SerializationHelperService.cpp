#include "SerializationHelperService.h"
#include "EngineInterface/SimulationFacade.h"

#include "Viewport.h"


DeserializedSimulation SerializationHelperService::getDeserializedSerialization(SimulationFacade const& simulationFacade)
{
    DeserializedSimulation result;
    result.auxiliaryData.timestep = static_cast<uint32_t>(simulationFacade->getCurrentTimestep());
    result.auxiliaryData.realTime = simulationFacade->getRealTime();
    result.auxiliaryData.zoom = Viewport::getZoomFactor();
    result.auxiliaryData.center = Viewport::getCenterInWorldPos();
    result.auxiliaryData.generalSettings = simulationFacade->getGeneralSettings();
    result.auxiliaryData.simulationParameters = simulationFacade->getSimulationParameters();
    result.statistics = simulationFacade->getStatisticsHistory().getCopiedData();
    result.mainData = simulationFacade->getClusteredSimulationData();
    return result;
}
