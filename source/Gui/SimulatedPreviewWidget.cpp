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
    if (!_genomeFromPreviousFrame.has_value()
        || _genomeFromPreviousFrame.value() != _editData->genome /* || _selectedGeneIndexFromPreviousFrame != _editData->selectedGeneIndex*/) {
        createSubGenomesForPreview();
        setPreviewData();
    }
    if (_genomeEditData->currentPreviewId.has_value() && _genomeEditData->currentPreviewId.value() != _editData->id) {
        setPreviewData();
    }
    calcPreview();
    drawPreview();

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
}

void _SimulatedPreviewWidget::setPreviewData()
{
    CollectionDescription preview;

    auto const& editService = DescriptionEditService::get();
    auto const& genomeEditService = GenomeDescriptionEditService::get();

    RealVector2D currentPos{toFloat(PREVIEW_HEIGHT) / 2, toFloat(PREVIEW_HEIGHT) / 2};

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
        currentPos.x += toFloat(PREVIEW_HEIGHT) / 2;  // Adjust position for the next sub-genome
    }
    _creatureIdsForPreview.clear();
    for (auto const& creature : preview._creatures) {
        _creatureIdsForPreview.emplace_back(creature._id);
    }

    _simulationFacade->setPreviewData(preview);
    _genomeEditData->currentPreviewId = _editData->id;

    // Create preview widgets
    if (_previewWidgets.size() != _creatureIdsForPreview.size()) {
        _previewWidgets.clear();
        for (int i = 0, size = toInt(_creatureIdsForPreview.size()); i < size; ++i) {
            _previewWidgets.emplace_back(_PreviewDescriptionWidget::create(_editData));
        }
    }
}

void _SimulatedPreviewWidget::calcPreview()
{
    auto fps = WindowController::get().getFps();
    auto duration = _fullSpeed ? std::chrono::milliseconds(1000 / fps) : std::chrono::milliseconds(1000 / fps / 2);
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

void _SimulatedPreviewWidget::drawPreview()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Preview").highlighted(true));

    auto previewRawData = _simulationFacade->getPreviewData();
    auto phenotypes = GenomeDescriptionEditService::get().extractPhenotypesFromPreview(std::move(previewRawData), _creatureIdsForPreview);
    for (auto const& [subGenome, phenotype] : boost::combine(_subGenomesForPreview, phenotypes)) {
        _genomeEditData->genotypeToPhenotype.insert_or_assign(subGenome, phenotype);
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

            auto title = "Sandbox " + std::to_string(subGenomeIndex + 1);
            std::vector<std::string> geneIndexStrings;
            for (auto const& geneIndex : _geneIndicesForSubGenomes.at(subGenomeIndex)) {
                geneIndexStrings.emplace_back("Gene " + std::to_string(geneIndex));
            }
            title += ": " + boost::join(geneIndexStrings, ", ");
            AlienGui::Group(AlienGui::GroupParameters().text(title));
        }
        //GenomeDescriptionEditService::get().removeSeedFromPhenotype(phenotype);
        auto previewDesc = PreviewDescriptionConverterService::get().convert(_editData->genome, std::move(phenotype), geneStartIndex);
        _previewWidgets.at(subGenomeIndex)->process(previewDesc);
    }
    ImGui::EndChild();
    ImGui::PopID();
}

void _SimulatedPreviewWidget::processActionBar()
{
    AlienGui::Separator();
    AlienGui::SelectableButton(AlienGui::SelectableButtonParameters().name(ICON_FA_GAMEPAD), _fullSimulation);
    ImGui::SameLine();
    AlienGui::SelectableButton(AlienGui::SelectableButtonParameters().name(ICON_FA_FAN), _fullSpeed);
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

