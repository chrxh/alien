#include "SimulatedPreviewWidget.h"

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/PreviewDescriptionConverterService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeWindowEditData.h"
#include "PreviewDescriptionWidget.h"
#include "WindowController.h"

SimulatedPreviewWidget
_SimulatedPreviewWidget::create(SimulationFacade const& simulationFacade, GenomeWindowEditData const& genomeEditData, GenomeTabEditData const& editData)
{
    return SimulatedPreviewWidget(new _SimulatedPreviewWidget(simulationFacade, genomeEditData, editData));
}

void _SimulatedPreviewWidget::process()
{
    if (!_genomeFromPreviousFrame.has_value() || _genomeFromPreviousFrame.value() != _editData->genome) {
        initPreview();
    }
    if (_genomeEditData->previewId.has_value() && _genomeEditData->previewId.value() != _editData->id) {
        continuePreview();
    }
    calcPreview();
    showPreview();

    _genomeFromPreviousFrame = _editData->genome;
}

_SimulatedPreviewWidget::_SimulatedPreviewWidget(
    SimulationFacade const& simulationFacade,
    GenomeWindowEditData const& genomeEditData,
    GenomeTabEditData const& editData)
    : _simulationFacade(simulationFacade)
    , _genomeEditData(genomeEditData)
    , _editData(editData)
{
    _previewWidget = _PreviewDescriptionWidget::create();
}

void _SimulatedPreviewWidget::initPreview()
{
    auto castratedGenome = _editData->genome;
    GenomeDescriptionEditService::get().adaptDescriptionForPreview(castratedGenome);

    _previewData = CollectionDescription().creatures({
        CreatureDescription()
            .genome(castratedGenome)
            .cells({
            CellDescription().stiffness(1.0f).cellTypeData(ConstructorDescription().geneIndex(0)).pos({100.0f, 100.0f}),
        }),
    });

    _simulationFacade->setPreviewData(_previewData);
    _genomeEditData->previewId = _editData->id;
}

void _SimulatedPreviewWidget::continuePreview()
{
    _simulationFacade->setPreviewData(_previewData);
    _genomeEditData->previewId = _editData->id;
}

void _SimulatedPreviewWidget::calcPreview()
{
    auto fps = WindowController::get().getFps();
    auto duration = std::chrono::milliseconds(1000 / fps / 2);
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
    _previewData = _simulationFacade->getPreviewData();
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
    auto copy = _previewData;
    auto previewDesc = PreviewDescriptionConverterService::get().convert(std::move(copy));
    _previewWidget->process(tps, previewDesc);

}
