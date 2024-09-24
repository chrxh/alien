#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"

class _EditorController
{
public:
    _EditorController(SimulationController const& simController);

    void registerCyclicReferences(UploadSimulationDialogWeakPtr const& uploadSimulationDialog);

    bool isOn() const;
    void setOn(bool value);

    void process();

    SelectionWindow getSelectionWindow() const;
    PatternEditorWindow getPatternEditorWindow() const;
    CreatorWindow getCreatorWindow() const;
    MultiplierWindow getMultiplierWindow() const;
    GenomeEditorWindow getGenomeEditorWindow() const;
    EditorModel getEditorModel() const;

    bool areInspectionWindowsActive() const;
    void onCloseAllInspectorWindows();

    bool isObjectInspectionPossible() const;
    bool isGenomeInspectionPossible() const;
    void onInspectSelectedObjects();
    void onInspectSelectedGenomes();
    void onInspectObjects(std::vector<CellOrParticleDescription> const& entities, bool selectGenomeTab);

    bool isCopyingPossible() const;
    void onCopy();
    bool isPastingPossible() const;
    void onPaste();
    bool isDeletingPossible() const;
    void onDelete();

    void selectObjects(RealVector2D const& viewPos, bool modifierKeyPressed);
    void moveSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);
    void fixateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& initialViewPos, RealVector2D const& selectionPositionOnClick);
    void updateSelectionRect(RealRect const& rect);
    void applyForces(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);
    void accelerateSelectedObjects(RealVector2D const& viewPos, RealVector2D const& prevWorldPos);

private:
    void processInspectorWindows();

private:
    EditorModel _editorModel;
    SelectionWindow _selectionWindow;
    PatternEditorWindow _patternEditorWindow;
    CreatorWindow _creatorWindow; 
    MultiplierWindow _multiplierWindow;
    GenomeEditorWindow _genomeEditorWindow;

    SimulationController _simController;

    bool _on = false;

    std::vector<InspectorWindow> _inspectorWindows;
    DataDescription _drawing;
};
