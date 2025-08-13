#include "SimulatedPreviewWidget.h"

#include <boost/range/adaptors.hpp>
#include <boost/range/combine.hpp>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionInfoService.h"
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
    if (!_genomeFromPreviousFrame.has_value() || _genomeFromPreviousFrame.value() != _editData->genome || _selectedGeneIndexFromPreviousFrame != _editData->selectedGeneIndex) {
        //createGenomeForPreview();
        //setPreview();
    }
    if (_genomeEditData->currentPreviewId.has_value() && _genomeEditData->currentPreviewId.value() != _editData->id) {
        //setPreview();
    }
    calcPreview();
    showPreview();

    _genomeFromPreviousFrame = _editData->genome;
    _selectedGeneIndexFromPreviousFrame = _editData->selectedGeneIndex;
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

void _SimulatedPreviewWidget::createSubGenomesForPreview()
{
    auto creatureGenesVec = GenomeDescriptionInfoService::get().getGeneIndicesForSubGenomes(_editData->genome);
    _subGenomesForPreview = GenomeDescriptionEditService::get().createSubGenomesForPreview(_editData->genome, creatureGenesVec);
}

void _SimulatedPreviewWidget::setPreviewData()
{
    CollectionDescription preview;

    auto const& editService = DescriptionEditService::get();
    auto const& genomeEditService = GenomeDescriptionEditService::get();

    RealVector2D currentPos{PREVIEW_HEIGHT / 2, PREVIEW_HEIGHT / 2};

    for (auto const& subGenome : _subGenomesForPreview) {
        auto findResult = _genomeEditData->genotypeToPhenotype.find(subGenome);
        if (findResult != _genomeEditData->genotypeToPhenotype.end()) {
            auto phenotype = findResult->second;
            editService.setCenter(phenotype, currentPos);
            preview.add(std::move(phenotype));
        } else {
            auto seed = genomeEditService.createSeedForPreview(subGenome, currentPos);
            preview.add(std::move(seed));
        }
        currentPos.x += PREVIEW_HEIGHT / 2;  // Adjust position for the next sub-genome
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
    auto previewRawData = _simulationFacade->getPreviewData();

    auto phenotypes = GenomeDescriptionEditService::get().extractPhenotypesFromPreview(std::move(previewRawData), _subGenomesForPreview);
    for (auto const& [subGenome, phenotype] : boost::combine(_subGenomesForPreview, phenotypes)) {
        _genomeEditData->genotypeToPhenotype.insert_or_assign(subGenome, phenotype);
    }

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

    // Show first creature for the moment
    GenomeDescriptionEditService::get().removeSeedFromPhenotype(phenotypes.front());
    auto previewDesc = PreviewDescriptionConverterService::get().convert(_editData->genome, std::move(phenotypes.front()), _rootGeneIndex);
    
    auto selectedGene = _editData->selectedGeneIndex;
    auto selectedNode = _editData->getSelectedNodeIndex();
    _previewWidget->process(tps, previewDesc, selectedGene, selectedNode);
    _editData->selectedGeneIndex = selectedGene;
    _editData->setSelectedNodeIndex(selectedNode);
}

