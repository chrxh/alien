#include "SimulationParametersBaseWidget.h"

#include <imgui.h>

#include <EngineInterface/Description.h>
#include <EngineInterface/ParametersEditService.h>
#include <EngineInterface/ParametersValidationService.h>
#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParametersUpdateConfig.h>

#include "AlienGui.h"
#include "SpecificationGuiService.h"
#include <EngineInterface/SimulationFacade.h>

void _SimulationParametersBaseWidget::init(SimulationFacade const& simulationFacade)
{

}

void _SimulationParametersBaseWidget::process(ParametersFilter const& filter)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, _SimulationFacade::get(), 0, filter);

    if (parameters != lastParameters) {
        ParametersValidationService::get().validateAndCorrect({_SimulationFacade::get()->getWorldSize()}, parameters);
        _SimulationFacade::get()->setSimulationParameters(parameters, SimulationParametersUpdateConfig::AllExceptChangingPositions);
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
