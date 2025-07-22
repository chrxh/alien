#include "SimulatedPreviewWidget.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "GenomeTabEditData.h"
#include "WindowController.h"

SimulatedPreviewWidget _SimulatedPreviewWidget::create(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData)
{
    return SimulatedPreviewWidget(new _SimulatedPreviewWidget(simulationFacade, editData));
}

void _SimulatedPreviewWidget::process()
{
    if (!_lastGenome.has_value() || _lastGenome.value() != _editData->genome) {
        auto preview = CollectionDescription().creatures({
            CreatureDescription()
                .genome(_editData->genome)
                .cells({
                    CellDescription()
                        .pos({0, 0})
                        .stiffness(1.0f)
                        .cellTypeData(ConstructorDescription().geneIndex(0)),
                }),
        });
        
        _simulationFacade->newPreview(preview);
    }
    auto fps = WindowController::get().getFps();
    auto duration = std::chrono::milliseconds(1000 / fps / 2);
    _simulationFacade->calcTimestepsForPreview(duration);

    _lastGenome = _editData->genome;
}

_SimulatedPreviewWidget::_SimulatedPreviewWidget(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData)
    : _simulationFacade(simulationFacade), _editData(editData)
{
}
