#pragma once

#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/SimulationFacade.h>

#include "Definitions.h"
#include "GenomeTabLayoutData.h"

class _GenomeTabWidget
{
public:
    static GenomeTabWidget create(
        GenomeWindowEditData const& genomeEditData,
        GenomeDesc const& genome,
        GenomeTabLayoutData const& layoutData = nullptr,
        std::optional<int> lineageId = std::nullopt);

    // Validated genome with normalized derived geometry, as it is maintained by the editor widgets
    static GenomeDesc normalizeForEditor(GenomeDesc genome);

    void process();

    int getTabId() const;
    std::string getName() const;

    // Set if the genome was opened from an inspected creature
    std::optional<int> getLineageId() const;
    void setLineageId(std::optional<int> value);

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
        GenomeTabLayoutData const& layoutData = nullptr,
        std::optional<int> lineageId = std::nullopt);

    void processEditors();
    void processPreview();
    void processStatusBar();

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
    std::optional<int> _lineageId;
    float _statusBarHeight = 0;

    float _previewZoom = 30.0f;
};
