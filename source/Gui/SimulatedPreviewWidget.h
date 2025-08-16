#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescription.h"

#include "Definitions.h"

class _SimulatedPreviewWidget
{
public:
    static SimulatedPreviewWidget create(
        SimulationFacade const& simulationFacade,
        GenomeWindowEditData const& genomeEditData,
        GenomeTabEditData const& editData);

    void process();

private:
    _SimulatedPreviewWidget(
        SimulationFacade const& simulationFacade,
        GenomeWindowEditData const& genomeEditData,
        GenomeTabEditData const& editData);

    void createSubGenomesForPreview();
    void setPreviewData();
    void calcPreview();
    void drawPreview();

private:
    void processSandbox(int previewWidgetIndex, CollectionDescription&& phenotype, int geneStartIndex, float width);

    int calcTpsForPreview();

    SimulationFacade _simulationFacade;
    std::vector<PreviewDescriptionWidget> _previewWidgets;

    GenomeWindowEditData _genomeEditData;
    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    std::vector<GenomeDescriptionWithStartGeneIndex> _subGenomesForPreview;
    std::vector<uint64_t> _creatureIdsForPreview;

    GenomeDescription _genomeForPreview;
    int _rootGeneIndex = 0;

    std::optional<GenomeDescription> _genomeFromPreviousFrame;

    std::optional<uint64_t> _previewTimestepFromPreviousMeasure;
    std::optional<std::chrono::steady_clock::time_point> _timepointFromPreviousMeasure;
    std::optional<int> _tpsFromPreviousMeasure;
};
