#include "SimulatedPreviewWidget.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "CreatureTabEditData.h"

SimulatedPreviewWidget _SimulatedPreviewWidget::create(SimulationFacade const& simulationFacade, CreatureTabEditData const& editData)
{
    return SimulatedPreviewWidget(new _SimulatedPreviewWidget(simulationFacade, editData));
}

void _SimulatedPreviewWidget::process()
{
    if (!_lastGenome.has_value() || _lastGenome.value() != _editData->genome) {
        CollectionDescription preview;
        _simulationFacade->newPreview(preview);
    }
    _lastGenome = _editData->genome;
}

_SimulatedPreviewWidget::_SimulatedPreviewWidget(SimulationFacade const& simulationFacade, CreatureTabEditData const& editData)
    : _simulationFacade(simulationFacade), _editData(editData)
{
}
