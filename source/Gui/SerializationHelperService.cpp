#include "SerializationHelperService.h"
#include "EngineInterface/SimulationFacade.h"

#include "Viewport.h"


DeserializedSimulation SerializationHelperService::getDeserializedSerialization(SimulationFacade const& simulationFacade)
{
    DeserializedSimulation result;
    result.auxiliaryData.timestep = static_cast<uint32_t>(simulationFacade->getCurrentTimestep());
    result.auxiliaryData.realTime = simulationFacade->getRealTime();
    result.auxiliaryData.zoom = Viewport::get().getZoomFactor();
    result.auxiliaryData.center = Viewport::get().getCenterInWorldPos();
    result.auxiliaryData.generalSettings = simulationFacade->getGeneralSettings();
    result.auxiliaryData.simulationParameters = simulationFacade->getSimulationParameters();
    result.statistics = simulationFacade->getStatisticsHistory().getCopiedData();
    result.mainData = simulationFacade->getClusteredSimulationData();
    return result;
}
