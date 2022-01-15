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
    ManipulatorWindow getManipulatorWindow() const;
    CreatorWindow getCreatorWindow() const;

    bool areInspectionWindowsActive() const;
    void onCloseAllInspectorWindows();

    bool isInspectionPossible() const;
    void onInspectEntities() const;

private:
    void processSelectionRect();
    void processInspectorWindows();

    void newEntitiesToInspect(std::vector<CellOrParticleDescription> const& entities);

    void leftMouseButtonPressed(RealVector2D const& viewPos, bool modifierKeyPressed);
    void leftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos, bool modifierKeyPressed);
    void leftMouseButtonReleased();

    void rightMouseButtonPressed(RealVector2D const& viewPos);
    void rightMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos);
    void rightMouseButtonReleased();

    void drawing();

private:
    EditorModel _editorModel;
    SelectionWindow _selectionWindow;
    ManipulatorWindow _manipulatorWindow;
    CreatorWindow _creatorWindow; 

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