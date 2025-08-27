#include "PreviewWidget.h"

#include <boost/algorithm/string/join.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/combine.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/StringHelper.h"

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionInfoService.h"
#include "EngineInterface/PreviewDescriptionConverterService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeWindowEditData.h"
#include "CreaturePreviewWidget.h"
#include "StyleRepository.h"
#include "WindowController.h"

PreviewWidget _PreviewWidget::create(
    SimulationFacade const& simulationFacade,
    GenomeWindowEditData const& genomeEditData,
    GenomeTabEditData const& editData)
{
    return PreviewWidget(new _PreviewWidget(simulationFacade, genomeEditData, editData));
}

void _PreviewWidget::process()
{
    if (!_genomeFromPreviousFrame.has_value() || _genomeFromPreviousFrame.value() != _editData->genome) {
        createSubGenomesForPreview();
        setupPreviewData();
        _run = true;
    }
    if (_genomeEditData->currentPreviewId.has_value() && _genomeEditData->currentPreviewId.value() != _editData->id) {
        setupPreviewData();
    }
    calcPreview();
    processSandboxes();

    processActionBar();

    _genomeFromPreviousFrame = _editData->genome;
}

_PreviewWidget::_PreviewWidget(
    SimulationFacade const& simulationFacade,
    GenomeWindowEditData const& genomeEditData,
    GenomeTabEditData const& editData)
    : _simulationFacade(simulationFacade)
    , _genomeEditData(genomeEditData)
    , _editData(editData)
{
}

void _PreviewWidget::createSubGenomesForPreview()
{
    auto geneIndicesForSubGenomes = GenomeDescriptionInfoService::get().getGeneIndicesForSubGenomes(_editData->genome);
    auto subGenomesForPreview = GenomeDescriptionEditService::get().createSubGenomesForPreview(_editData->genome, geneIndicesForSubGenomes);

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
    auto const& genomeEditService = GenomeDescriptionEditService::get();

    std::vector<GenomeDescriptionWithStartGeneIndex> subGenomesForPreview;
    for (auto const& creatureWidget : _creatureWidgets) {
        subGenomesForPreview.emplace_back(creatureWidget->getGenomeWithStartIndex());
    }
    auto preview = genomeEditService.createSeedCollectionForPreview(
        subGenomesForPreview,
        useCache ? _genomeEditData->genotypeToPhenotypeCache : std::unordered_map<GenomeDescriptionWithStartGeneIndex, CollectionDescription>());

    _simulationFacade->setPreviewData(preview.data);
    _genomeEditData->currentPreviewId = _editData->id;

    for (auto const& [index, creatureWidget] : _creatureWidgets | boost::adaptors::indexed(0)) {
        creatureWidget->setCreatureId(preview.seedCreatureIds.at(index));
    }
}

void _PreviewWidget::calcPreview()
{
    if (!_run) {
        return;
    }

    auto fps = WindowController::get().getFps();
    auto duration = std::chrono::milliseconds(1000 / fps * _simulationSpeed / 100);
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

void _PreviewWidget::processSandboxes()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Preview").highlighted(true));

    auto previewRawData = _simulationFacade->getPreviewData();

    std::vector<uint64_t> seedCreatureIds;
    std::vector<GenomeDescriptionWithStartGeneIndex> subGenomesForPreview;
    for (auto const& creatureWidget : _creatureWidgets) {
        seedCreatureIds.emplace_back(creatureWidget->getCreatureId());
        subGenomesForPreview.emplace_back(creatureWidget->getGenomeWithStartIndex());
    }
    auto phenotypes = GenomeDescriptionEditService::get().extractPhenotypesFromPreview(std::move(previewRawData), seedCreatureIds);
    for (auto const& [subGenome, phenotype] : boost::combine(subGenomesForPreview, phenotypes)) {
        _genomeEditData->genotypeToPhenotypeCache.insert_or_assign(subGenome, phenotype);
    }

    if (ImGui::BeginChild("Sandboxes", ImVec2(0, -scale(47.0f)), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto space = ImGui::GetContentRegionAvail();
        auto width = std::max(space.x / _creatureWidgets.size() - scale(7.0f), space.y);
        for (int i = 0, size = toInt(phenotypes.size()); i < size; ++i) {
            processSandbox(i, std::move(phenotypes.at(i)), width);
            if (i < size - 1) {
                ImGui::SameLine();
            }
        }
    }
    ImGui::EndChild();
}

void _PreviewWidget::processSandbox(int subGenomeIndex, CollectionDescription&& phenotype, float width)
{
    ImGui::PushID(subGenomeIndex);
    if (ImGui::BeginChild("Sandbox", ImVec2(width, 0), 0, 0)) {
        auto& creatureWidget = _creatureWidgets.at(subGenomeIndex);

        auto multipleSandboxes = _creatureWidgets.size() > 1;
        if (multipleSandboxes) {
            AlienGui::MoveTickUp();
            AlienGui::MoveTickUp();

            std::vector<std::string> geneIndexStrings;
            auto geneIndices = creatureWidget->getGeneIndices();
            geneIndexStrings.emplace_back(std::to_string(geneIndices.front() + 1) + " (start)");
            for (auto const& geneIndex : geneIndices | boost::adaptors::sliced(1, geneIndices.size())) {
                geneIndexStrings.emplace_back(std::to_string(geneIndex + 1));
            }
            auto title = "Genes: " + boost::join(geneIndexStrings, ", ");
            AlienGui::Group(AlienGui::GroupParameters().text(title));
        }
        GenomeDescriptionEditService::get().removeSeedFromPhenotype(phenotype);

        creatureWidget->process(std::move(phenotype));
    }
    ImGui::EndChild();
    ImGui::PopID();
}

void _PreviewWidget::processActionBar()
{
    AlienGui::Separator();
    AlienGui::SelectableButton(AlienGui::SelectableButtonParameters().name(/*ICON_FA_CUBES*/ICON_FA_GEM), _fullSimulation);

    ImGui::SameLine();
    AlienGui::VerticalSeparator(20.0f);

    ImGui::SameLine();
    if (AlienGui::Button(ICON_FA_UNDO)) {
        setupPreviewData(false);
        _run = true;
    }

    ImGui::SameLine();
    ImGui::BeginDisabled(_run);
    if (AlienGui::Button(ICON_FA_PLAY)) {
        _run = true;
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!_run);
    if (AlienGui::Button(ICON_FA_PAUSE)) {
        _run = false;
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(_run);
    AlienGui::Button(ICON_FA_CHEVRON_RIGHT);
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::SetNextItemWidth(scale(90.0f));
    ImGui::SliderInt("##TPSRestriction", &_simulationSpeed, 5, 100, "%d%% speed", ImGuiSliderFlags_None);
    _simulationSpeed = std::clamp(_simulationSpeed, 5, 100);

    ImGui::SameLine();
    AlienGui::VerticalSeparator(20.0f);

    ImGui::SameLine();
    auto tps = calcTpsForPreview();
    AlienGui::Text("TPS: " + StringHelper::format(tps));
}

int _PreviewWidget::calcTpsForPreview()
{
    auto now = std::chrono::steady_clock::now();
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
    return tps;
}

