#include "SimulatedPreviewWidget.h"

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
#include "PreviewDescriptionWidget.h"
#include "StyleRepository.h"
#include "WindowController.h"

SimulatedPreviewWidget _SimulatedPreviewWidget::create(
    SimulationFacade const& simulationFacade,
    GenomeWindowEditData const& genomeEditData,
    GenomeTabEditData const& editData)
{
    return SimulatedPreviewWidget(new _SimulatedPreviewWidget(simulationFacade, genomeEditData, editData));
}

void _SimulatedPreviewWidget::process()
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

_SimulatedPreviewWidget::_SimulatedPreviewWidget(
    SimulationFacade const& simulationFacade,
    GenomeWindowEditData const& genomeEditData,
    GenomeTabEditData const& editData)
    : _simulationFacade(simulationFacade)
    , _genomeEditData(genomeEditData)
    , _editData(editData)
{
}

void _SimulatedPreviewWidget::createSubGenomesForPreview()
{
    _geneIndicesForSubGenomes = GenomeDescriptionInfoService::get().getGeneIndicesForSubGenomes(_editData->genome);
    _subGenomesForPreview = GenomeDescriptionEditService::get().createSubGenomesForPreview(_editData->genome, _geneIndicesForSubGenomes);

    _visualFrontAngles.clear();
    _visualFrontAngles.resize(_subGenomesForPreview.size(), std::nullopt);
}

void _SimulatedPreviewWidget::setupPreviewData(bool useCache)
{
    auto const& genomeEditService = GenomeDescriptionEditService::get();

    auto preview = genomeEditService.createSeedCollectionForPreview(
        _subGenomesForPreview,
        useCache ? _genomeEditData->genotypeToPhenotypeCache : std::unordered_map<GenomeDescriptionWithStartGeneIndex, CollectionDescription>());
    _seedCreatureIdsForPreview = preview.seedCreatureIds;

    _simulationFacade->setPreviewData(preview.data);
    _genomeEditData->currentPreviewId = _editData->id;

    // Create preview widgets
    if (_previewWidgets.size() != _subGenomesForPreview.size()) {
        _previewWidgets.clear();
        for (int i = 0, size = toInt(_subGenomesForPreview.size()); i < size; ++i) {
            _previewWidgets.emplace_back(_PreviewDescriptionWidget::create(_editData));
        }
    }
}

void _SimulatedPreviewWidget::calcPreview()
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

void _SimulatedPreviewWidget::processSandboxes()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Preview").highlighted(true));

    auto previewRawData = _simulationFacade->getPreviewData();
    auto phenotypes = GenomeDescriptionEditService::get().extractPhenotypesFromPreview(std::move(previewRawData), _seedCreatureIdsForPreview);
    for (auto const& [subGenome, phenotype] : boost::combine(_subGenomesForPreview, phenotypes)) {
        _genomeEditData->genotypeToPhenotypeCache.insert_or_assign(subGenome, phenotype);
    }

    if (ImGui::BeginChild("Sandboxes", ImVec2(0, -scale(47.0f)), 0, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto space = ImGui::GetContentRegionAvail();
        auto width = std::max(space.x / _previewWidgets.size(), space.y);
        for (int i = 0, size = toInt(phenotypes.size()); i < size; ++i) {
            processSandbox(i, std::move(phenotypes.at(i)), _subGenomesForPreview.at(i).startIndex, width);
            if (i < size - 1) {
                ImGui::SameLine();
            }
        }
    }
    ImGui::EndChild();
}

void _SimulatedPreviewWidget::processSandbox(int subGenomeIndex, CollectionDescription&& phenotype, int geneStartIndex, float width)
{
    ImGui::PushID(subGenomeIndex);
    if (ImGui::BeginChild("Sandbox", ImVec2(width, 0), 0, 0)) {
        auto multipleSandboxes = _previewWidgets.size() > 1;
        if (multipleSandboxes) {
            AlienGui::MoveTickUp();
            AlienGui::MoveTickUp();

            std::vector<std::string> geneIndexStrings;
            auto geneIndices = _geneIndicesForSubGenomes.at(subGenomeIndex);
            geneIndexStrings.emplace_back(std::to_string(geneIndices.front() + 1) + " (start)");
            for (auto const& geneIndex : geneIndices | boost::adaptors::sliced(1, geneIndices.size())) {
                geneIndexStrings.emplace_back(std::to_string(geneIndex + 1));
            }
            auto title = "Genes: " + boost::join(geneIndexStrings, ", ");
            AlienGui::Group(AlienGui::GroupParameters().text(title));
        }
        GenomeDescriptionEditService::get().removeSeedFromPhenotype(phenotype);
        auto conversionResult =
            PreviewDescriptionConverterService::get().convert(_editData->genome, std::move(phenotype), geneStartIndex, _visualFrontAngles.at(subGenomeIndex));
        _previewWidgets.at(subGenomeIndex)->process(conversionResult.description);
        _visualFrontAngles.at(subGenomeIndex) = conversionResult.visualFrontAngle;
    }
    ImGui::EndChild();
    ImGui::PopID();
}

void _SimulatedPreviewWidget::processActionBar()
{
    AlienGui::Separator();
    AlienGui::SelectableButton(AlienGui::SelectableButtonParameters().name(ICON_FA_CUBES), _fullSimulation);

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

int _SimulatedPreviewWidget::calcTpsForPreview()
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

