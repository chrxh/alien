#include "MultiplierService.h"

#include "SimulationFacade.h"

namespace
{
    void replaceSelection(ContentDesc&& content)
    {
        _SimulationFacade::get()->removeSelectedObjects(true);
        _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
    }
}

MultiplierService::Result MultiplierService::multiplyInGrid(DescEditService::GridMultiplyParameters const& parameters) const
{
    auto origSelection = _SimulationFacade::get()->getSelectedSimulationData(true);
    replaceSelection(DescEditService::get().gridMultiply(origSelection, parameters));
    return {.origSelection = std::move(origSelection)};
}

MultiplierService::Result MultiplierService::multiplyRandomly(DescEditService::RandomMultiplyParameters const& parameters) const
{
    auto origSelection = _SimulationFacade::get()->getSelectedSimulationData(true);
    auto overlappingCheckSuccessful = true;
    auto multiplicationResult = DescEditService::get().randomMultiply(
        origSelection, parameters, _SimulationFacade::get()->getWorldSize(), _SimulationFacade::get()->getSimulationData(), overlappingCheckSuccessful);
    replaceSelection(std::move(multiplicationResult));
    return {.origSelection = std::move(origSelection), .overlappingCheckSuccessful = overlappingCheckSuccessful};
}

void MultiplierService::undo(ContentDesc const& origSelection) const
{
    replaceSelection(ContentDesc(origSelection));
}
