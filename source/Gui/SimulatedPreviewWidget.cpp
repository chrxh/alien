#include "SimulatedPreviewWidget.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/PreviewDescriptionConverterService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeWindowEditData.h"
#include "PreviewDescriptionWidget.h"
#include "PreviewDescriptionWidgetSettings.h"
#include "WindowController.h"

SimulatedPreviewWidget _SimulatedPreviewWidget::create(
    SimulationFacade const& simulationFacade,
    PreviewDescriptionSettings const& settings,
    GenomeWindowEditData const& genomeEditData,
    GenomeTabEditData const& editData)
{
    return SimulatedPreviewWidget(new _SimulatedPreviewWidget(simulationFacade, settings, genomeEditData, editData));
}

void _SimulatedPreviewWidget::process()
{
    if (!_genomeFromPreviousFrame.has_value() || _genomeFromPreviousFrame.value() != _editData->genome) {
        createGenomeForPreview();
        setPreview();
    }
    if (_genomeEditData->currentPreviewId.has_value() && _genomeEditData->currentPreviewId.value() != _editData->id) {
        setPreview();
    }
    calcPreview();
    showPreview();

    _genomeFromPreviousFrame = _editData->genome;
}

_SimulatedPreviewWidget::_SimulatedPreviewWidget(
    SimulationFacade const& simulationFacade,
    PreviewDescriptionSettings const& settings,
    GenomeWindowEditData const& genomeEditData,
    GenomeTabEditData const& editData)
    : _simulationFacade(simulationFacade)
    , _settings(settings)
    , _genomeEditData(genomeEditData)
    , _editData(editData)
{
    _previewWidget = _PreviewDescriptionWidget::create(settings);
}

void _SimulatedPreviewWidget::createGenomeForPreview()
{
    _genomeForPreview = _editData->genome;
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(_genomeForPreview);

}

void _SimulatedPreviewWidget::setPreview()
{
    CollectionDescription preview;

    // Cache the preview data to avoid recalculating it if the genome hasn't changed
    auto findResult = _genomeEditData->genotypeToPhenotype.find(_genomeForPreview);
    if (findResult != _genomeEditData->genotypeToPhenotype.end()) {
        preview = findResult->second;
    } else {
        preview = CollectionDescription().creatures({
            CreatureDescription()
                .genome(_genomeForPreview)
                .cells({
                    CellDescription().stiffness(1.0f).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f}),
                }),
        });
    }

    _simulationFacade->setPreviewData(preview);
    _genomeEditData->currentPreviewId = _editData->id;
}

void _SimulatedPreviewWidget::calcPreview()
{
    auto fps = WindowController::get().getFps();
    auto duration = _settings->maxSpeed ? std::chrono::milliseconds(1000 / fps) : std::chrono::milliseconds(1000 / fps / 2);
    _simulationFacade->calcTimestepsForPreview(duration);
}

namespace
{
    int calcTimestepsPerSecond(uint64_t lastTimestep, uint64_t timestep, std::chrono::milliseconds const& duration)
    {
        if (duration.count() == 0) {
            return 0;
        }
        uint64_t deltaTimesteps = (timestep > lastTimestep) ? (timestep - lastTimestep) : 0;
        double seconds = static_cast<double>(duration.count()) / 1000.0;
        if (seconds == 0.0) {
            return 0;
        }
        return static_cast<int>(static_cast<double>(deltaTimesteps) / seconds);
    }
}

void _SimulatedPreviewWidget::showPreview()
{
    auto now = std::chrono::steady_clock::now();
    auto previewData = _simulationFacade->getPreviewData();
    _genomeEditData->genotypeToPhenotype.insert_or_assign(_genomeForPreview, previewData);

    auto timestep = _simulationFacade->getCurrentTimestepForPreview();

    int tps = 0;
    if (_previewTimestepFromPreviousMeasure.has_value() && _timepointFromPreviousMeasure.has_value()) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_timepointFromPreviousMeasure);
        if (duration.count() > 300) {
            tps = calcTimestepsPerSecond(_previewTimestepFromPreviousMeasure.value(), timestep, duration);
            _previewTimestepFromPreviousMeasure = timestep;
            _timepointFromPreviousMeasure = now;
            _tpsFromPreviousMeasure = tps;
        } else {
            tps = _tpsFromPreviousMeasure.value();
        }
    } else {
        _previewTimestepFromPreviousMeasure = timestep;
        _timepointFromPreviousMeasure = now;
        _tpsFromPreviousMeasure = tps;
    }
    auto previewDesc = PreviewDescriptionConverterService::get().convert(_editData->genome, std::move(previewData));
    _previewWidget->setSelectedGene(_editData->selectedGeneIndex);
    _previewWidget->setSelectedNode(_editData->getSelectedNodeIndex());
    _previewWidget->process(tps, previewDesc);
    _editData->selectedGeneIndex = _previewWidget->getSelectedGene();
    _editData->setSelectedNodeIndex(_previewWidget->getSelectedNode());
}

