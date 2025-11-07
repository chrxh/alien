#include "SimulationParametersBaseWidget.h"

#include <imgui.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/ParametersEditService.h>
#include <EngineInterface/ParametersValidationService.h>
#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParametersUpdateConfig.h>

#include "AlienGui.h"
#include "SpecificationGuiService.h"
#include "SimulationFacadeProvider.h"

void _SimulationParametersBaseWidget::init(SimulationFacade const& simulationFacade)
{

}

void _SimulationParametersBaseWidget::process(ParametersFilter const& filter)
{
    auto parameters = SimulationFacadeProvider::getSimulationFacade()->getSimulationParameters();
    auto origParameters = SimulationFacadeProvider::getSimulationFacade()->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, SimulationFacadeProvider::getSimulationFacade(), 0, filter);

    if (parameters != lastParameters) {
        ParametersValidationService::get().validateAndCorrect({SimulationFacadeProvider::getSimulationFacade()->getWorldSize()}, parameters);
        SimulationFacadeProvider::getSimulationFacade()->setSimulationParameters(parameters, SimulationParametersUpdateConfig::AllExceptChangingPositions);
    }
}

std::string _SimulationParametersBaseWidget::getLocationName()
{
    return "Simulation parameters for 'Base'";
}

int _SimulationParametersBaseWidget::getOrderNumber() const
{
    return 0;
}

void _SimulationParametersBaseWidget::setOrderNumber(int orderNumber)
{
    // do nothing
}
