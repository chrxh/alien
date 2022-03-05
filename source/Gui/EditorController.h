#pragma once

#include "Base/Definitions.h"
#include "EngineInterface/Descriptions.h"

#include "Definitions.h"

class _EditorController
{
public:
    _EditorController(SimulationController const& simController, Viewport const& viewport);

    bool isOn() const;
    void setOn(bool value);

    void process();

    SelectionWindow getSelectionWindow() const;
    PatternEditorWindow getPatternEditorWindow() const;
    CreatorWindow getCreatorWindow() const;
    MultiplierWindow getMultiplierWindow() const;
    SymbolsWindow getSymbolsWindow() const;

    bool areInspectionWindowsActive() const;
    void onCloseAllInspectorWindows();

    bool isInspectionPossible() const;
    void onInspectEntities() const;

    bool isCopyingPossible() const;
    void onCopy();
    bool isPastingPossible() const;
    void onPaste();
    bool isDeletingPossible() const;
    void onDelete();

private:
    void processSelectionRect();
    void processInspectorWindows();

    void newEntitiesToInspect(std::vector<CellOrParticleDescription> const& entities);

    void selectEntities(RealVector2D const& viewPos, bool modifierKeyPressed);
    void moveSelectedEntities(RealVector2D const& viewPos, RealVector2D const& prevViewPos);
    void applyForces(RealVector2D const& viewPos, RealVector2D const& prevViewPos);

    void createSelectionRect(RealVector2D const& viewPos);
    void resizeSelectionRect(RealVector2D const& viewPos, RealVector2D const& prevViewPos);
    void removeSelectionRect();

private:
    EditorModel _editorModel;
    SelectionWindow _selectionWindow;
    PatternEditorWindow _patternEditorWindow;
    CreatorWindow _creatorWindow; 
    MultiplierWindow _multiplierWindow; 
    SymbolsWindow _symbolsWindow;

    SimulationController _simController;
    Viewport _viewport;

    bool _on = false;

    std::optional<RealVector2D> _prevMousePosInt;

    struct SelectionRect
    {
        RealVector2D startPos;
        RealVector2D endPos;
    };
    std::optional<SelectionRect> _selectionRect;
    std::vector<InspectorWindow> _inspectorWindows;
    DataDescription _drawing;
};