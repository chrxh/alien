#include "PreviewWidget.h"

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/combine.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <EngineInterface/Desc.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/GenomeDescEditService.h>
#include <EngineInterface/GenomeDescInfoService.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "CreaturePreviewWidget.h"
#include "GenomeTabEditData.h"
#include "GenomeWindowEditData.h"
#include "StyleRepository.h"
#include "WindowController.h"

PreviewWidget _PreviewWidget::create(GenomeWindowEditData const& genomeEditData, GenomeTabEditData const& editData)
{
    return PreviewWidget(new _PreviewWidget(genomeEditData, editData));
}

void _PreviewWidget::process()
{
    // Has genome changed?
    auto sessionId = _SimulationFacade::get()->getSessionId();
    if (!_genomeFromPreviousFrame.has_value() || _genomeFromPreviousFrame.value() != _editData->genome || !_sessionIdFromPreviousFrame.has_value()
        || sessionId != _sessionIdFromPreviousFrame.value()) {
        createSubGenomesForPreview();
        setupPreviewData();
        _editData->run = true;
        _savepoints.clear();
    }

    // Has tab changed?
    if (_genomeEditData->currentPreviewId.has_value() && _genomeEditData->currentPreviewId.value() != _editData->id) {
        setupPreviewData();
    }
    calcPreview();
    processCreaturePreviews();

    processActionBar();

    _genomeFromPreviousFrame = _editData->genome;
    _sessionIdFromPreviousFrame = sessionId;
}

_PreviewWidget::_PreviewWidget(GenomeWindowEditData const& genomeEditData, GenomeTabEditData const& editData)
    : _genomeEditData(genomeEditData)
    , _editData(editData)
{}

void _PreviewWidget::createSubGenomesForPreview()
{
    auto geneIndicesForSubGenomes = GenomeDescInfoService::get().getGeneIndicesForSubGenomes(_editData->genome);
    auto subGenomesForPreview = GenomeDescEditService::get().createSubGenomesForPreview(_editData->genome, geneIndicesForSubGenomes, _editData->detailSimulation);

    if (_creatureWidgets.size() != subGenomesForPreview.size()) {
        _creatureWidgets.clear();
        for (auto const& [geneIndices, subGenome] : boost::combine(geneIndicesForSubGenomes, subGenomesForPreview)) {
            _creatureWidgets.emplace_back(_CreaturePreviewWidget::create(_editData, geneIndices, subGenome));
        }
    } else {

        for (int i = 0, size = _creatureWidgets.size(); i < size; ++i) {
            auto& creatureWidget = _creatureWidgets.at(i);
            creatureWidget->setGeneIndices(geneIndicesForSubGenomes.at(i));
            creatureWidget->setGenomeWithStartIndex(subGenomesForPreview.at(i));
            creatureWidget->resetVisualFrontAngle();
        }
    }
}

void _PreviewWidget::setupPreviewData(bool useCache)
{
    auto const& genomeEditService = GenomeDescEditService::get();

    std::vector<SubGenomeDesc> subGenomesForPreview;
    for (auto const& creatureWidget : _creatureWidgets) {
        subGenomesForPreview.emplace_back(creatureWidget->getGenomeWithStartIndex());
    }

    auto preview = genomeEditService.createSeedCollectionForPreview(
        subGenomesForPreview,
        useCache ? std::optional<std::reference_wrapper<GenotypeToPhenotypeCache const>>(_genomeEditData->genotypeToPhenotypeCache) : std::nullopt);

    _SimulationFacade::get()->setPreviewData(preview.description);
    _SimulationFacade::get()->setCurrentTimestepForPreview(_currentTimestep);
    _genomeEditData->currentPreviewId = _editData->id;

    setSeedCreatureIds(preview.seedCreatureIds);
}

void _PreviewWidget::calcPreview()
{
    if (!_editData->run) {
        return;
    }

    auto fps = WindowController::get().getFps();
    auto duration = std::chrono::milliseconds(1000 / fps * _editData->simulationSpeed / 100);
    _SimulationFacade::get()->calcTimestepsForPreview(duration, _editData->detailSimulation);
    _currentTimestep = _SimulationFacade::get()->getCurrentTimestepForPreview();
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

void _PreviewWidget::processCreaturePreviews()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Preview").highlighted(true));

    auto previewRawData = _SimulationFacade::get()->getPreviewData();

    // Get phenotypes for all sub-genomes
    auto seedCreatureIds = getSeedCreatureIds();
    auto subGenomesForPreview = getSubGenomes();
    auto phenotypes = GenomeDescEditService::get().extractPhenotypesFromPreview(std::move(previewRawData), seedCreatureIds);

    // Display and edit previews
    auto phenotypeChanged = false;
    if (ImGui::BeginChild("Sandboxes", ImVec2(0, -scale(47.0f)), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto space = ImGui::GetContentRegionAvail();
        auto width = std::max(space.x / _creatureWidgets.size() - scale(7.0f), space.y);
        for (int i = 0, size = toInt(phenotypes.size()); i < size; ++i) {
            processCreaturePreview(phenotypeChanged, i, phenotypes.at(i), width);
            if (i < size - 1) {
                ImGui::SameLine();
            }
        }
    }

    // Update cache for new phenotypes
    for (auto const& [subGenome, phenotype] : boost::combine(subGenomesForPreview, phenotypes)) {
        _genomeEditData->genotypeToPhenotypeCache.insertOrAssign(subGenome, phenotype);
    }
    if (phenotypeChanged) {
        setupPreviewData();
    }
    ImGui::EndChild();
}

void _PreviewWidget::processCreaturePreview(bool& phenotypeChanged, int subGenomeIndex, Desc& phenotype, float width)
{
    ImGui::PushID(subGenomeIndex);
    auto& creatureWidget = _creatureWidgets.at(subGenomeIndex);
    creatureWidget->process(phenotypeChanged, phenotype, width);
    ImGui::PopID();
}

void _PreviewWidget::processActionBar()
{
    AlienGui::Separator();
    // Alternatives: ICON_FA_GEM, ICON_FA_FIRE, ICON_FA_CUDA
    if (AlienGui::SelectableButton(
            AlienGui::SelectableButtonParameters().name(ICON_FA_DICE_D20).tooltip("Activates a more detail simulation including signals and muscles"),
            _editData->detailSimulation)) {
        onRestart();
    }

    ImGui::SameLine();
    AlienGui::VerticalSeparator(20.0f);

    ImGui::SameLine();
    ImGui::BeginDisabled(_editData->run);
    if (AlienGui::Button(ICON_FA_PLAY)) {
        onRun();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!_editData->run);
    if (AlienGui::Button(ICON_FA_PAUSE)) {
        _editData->run = false;
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    AlienGui::VerticalSeparator(20.0f);

    ImGui::SameLine();
    ImGui::BeginDisabled(_editData->run || _savepoints.empty());
    if (AlienGui::Button(ICON_FA_CHEVRON_LEFT)) {
        onStepBackward();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(_editData->run);
    if (AlienGui::Button(ICON_FA_CHEVRON_RIGHT)) {
        onStepForward();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    AlienGui::VerticalSeparator(20.0f);

    ImGui::SameLine();
    if (AlienGui::Button(ICON_FA_REDO)) {
        onRestart();
    }

    ImGui::SameLine();
    ImGui::SetNextItemWidth(scale(90.0f));
    ImGui::SliderInt("##TPSRestriction", &_editData->simulationSpeed, 1, 100, "%d%% speed", ImGuiSliderFlags_None);
    _editData->simulationSpeed = std::clamp(_editData->simulationSpeed, 1, 100);

    ImGui::SameLine();
    AlienGui::VerticalSeparator(20.0f);

    ImGui::SameLine();
    auto tps = calcTpsForPreview();
    AlienGui::Text("TPS: " + StringHelper::format(tps));
}

int _PreviewWidget::calcTpsForPreview()
{
    auto now = std::chrono::steady_clock::now();
    _currentTimestep = _SimulationFacade::get()->getCurrentTimestepForPreview();
    int tps = 0;
    if (_previewTimestepFromPreviousMeasure.has_value() && _timepointFromPreviousMeasure.has_value()) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - *_timepointFromPreviousMeasure);
        if (duration.count() > 300) {
            tps = calcTimestepsPerSecond(_previewTimestepFromPreviousMeasure.value(), _currentTimestep, duration);
            _previewTimestepFromPreviousMeasure = _currentTimestep;
            _timepointFromPreviousMeasure = now;
            _tpsFromPreviousMeasure = tps;
        } else {
            tps = _tpsFromPreviousMeasure.value();
        }
    } else {
        _previewTimestepFromPreviousMeasure = _currentTimestep;
        _timepointFromPreviousMeasure = now;
        _tpsFromPreviousMeasure = tps;
    }
    return tps;
}

void _PreviewWidget::onRun()
{
    _editData->run = true;
    _savepoints.clear();
}

void _PreviewWidget::onStepBackward()
{
    auto lastSavepoint = _savepoints.back();
    _savepoints.pop_back();
    _SimulationFacade::get()->setCurrentTimestepForPreview(lastSavepoint.timestep);
    _SimulationFacade::get()->setPreviewData(lastSavepoint.description);
    setSeedCreatureIds(lastSavepoint.seedCreatureIds);
}

void _PreviewWidget::onStepForward()
{
    auto timestep = _SimulationFacade::get()->getCurrentTimestepForPreview();
    auto data = _SimulationFacade::get()->getPreviewData();
    auto seedCreatureIds = getSeedCreatureIds();
    _savepoints.emplace_back(timestep, data, seedCreatureIds);

    _SimulationFacade::get()->calcTimestepsForPreview(TIMESTEPS_PER_CELL_FUNCTION, _editData->detailSimulation);
}

void _PreviewWidget::onRestart()
{
    createSubGenomesForPreview();
    setupPreviewData(false);
    _editData->run = true;
    _savepoints.clear();
}

std::vector<uint64_t> _PreviewWidget::getSeedCreatureIds() const
{
    std::vector<uint64_t> result;
    for (auto const& creatureWidget : _creatureWidgets) {
        result.emplace_back(creatureWidget->getCreatureId());
    }
    return result;
}

void _PreviewWidget::setSeedCreatureIds(std::vector<uint64_t> const& value)
{
    for (auto const& [creatureWidget, creatureId] : boost::combine(_creatureWidgets, value)) {
        creatureWidget->setCreatureId(creatureId);
    }
}

std::vector<SubGenomeDesc> _PreviewWidget::getSubGenomes() const
{
    std::vector<SubGenomeDesc> result;
    for (auto const& creatureWidget : _creatureWidgets) {
        result.emplace_back(creatureWidget->getGenomeWithStartIndex());
    }
    return result;
}
