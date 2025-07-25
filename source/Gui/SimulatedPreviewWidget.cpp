#include "SimulatedPreviewWidget.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/PreviewDescriptionConverterService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "PreviewDescriptionWidget.h"
#include "WindowController.h"

SimulatedPreviewWidget _SimulatedPreviewWidget::create(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData)
{
    return SimulatedPreviewWidget(new _SimulatedPreviewWidget(simulationFacade, editData));
}

void _SimulatedPreviewWidget::process()
{
    if (!_lastGenome.has_value() || _lastGenome.value() != _editData->genome) {
        auto castratedGenome = _editData->genome;

        GenomeDescriptionEditService::get().adaptDescriptionForPreview(castratedGenome);
        auto preview = CollectionDescription().creatures({
            CreatureDescription()
                .genome(castratedGenome)
                .cells({
                    CellDescription()
                        .pos({100.0f, 100.0f})
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

    auto desc = _simulationFacade->getPreviewData();
    auto previewDesc = PreviewDescriptionConverterService::get().convert(desc);
    
    _previewWidget->process(previewDesc);
}

_SimulatedPreviewWidget::_SimulatedPreviewWidget(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData)
    : _simulationFacade(simulationFacade), _editData(editData)
{
    _previewWidget = _PreviewDescriptionWidget::create();
}
