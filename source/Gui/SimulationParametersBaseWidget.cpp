#include "SimulationParametersBaseWidget.h"

#include <imgui.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/ParametersEditService.h>
#include <EngineInterface/ParametersValidationService.h>
#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SimulationParametersUpdateConfig.h>

#include "AlienGui.h"
#include "SpecificationGuiService.h"

void _SimulationParametersBaseWidget::init() {}

void _SimulationParametersBaseWidget::process(ParametersFilter const& filter)
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();
    auto lastParameters = parameters;

    SpecificationGuiService::get().createWidgetsForParameters(parameters, origParameters, 0, filter);

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
