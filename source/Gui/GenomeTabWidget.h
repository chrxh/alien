#pragma once

#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/SimulationFacade.h>

#include "Definitions.h"
#include "GenomeTabLayoutData.h"

class _GenomeTabWidget
{
public:
    static GenomeTabWidget create(GenomeWindowEditData const& genomeEditData, GenomeDesc const& genome, GenomeTabLayoutData const& layoutData = nullptr);

    void process();

    int getTabId() const;
    std::string getName() const;

    GenomeTabEditData const& getEditData() const;
    GenomeTabLayoutData const& getLayoutData() const;
    GenomeDesc const& getGenomeDesc() const;

    void setGenomeDesc(GenomeDesc const& genome);

    bool hasGenomeChanged() const;

    bool isEmpty() const;
    void resetOriginal();
    void revertChanges();

private:
    _GenomeTabWidget(
        GenomeWindowEditData const& genomeEditData,
        GenomeDesc const& genome,
        GenomeTabLayoutData const& layoutData = nullptr);

    void processEditors();
    void processPreview();

    void doLayout();

    // Widgets
    GenomeEditorWidget _genomeEditorWidget;
    GeneEditorWidget _geneEditorWidget;
    NodeEditorWidget _nodeEditorWidget;
    PreviewWidget _simulatedPreviewWidget;

    // Creature data
    GenomeTabEditData _editData;

    // Layout data
    GenomeTabLayoutData _origLayoutData;
    GenomeTabLayoutData _layoutData;
    std::optional<RealVector2D> _lastWindowSize;

    float _previewZoom = 30.0f;
};
