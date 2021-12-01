#pragma once

#include "Base/Definitions.h"
#include "EngineImpl/Definitions.h"

#include "Definitions.h"

class _EditorController
{
public:
    _EditorController(
        SimulationController const& simController,
        Viewport const& viewport,
        StyleRepository const& styleRepository);

    bool isOn() const;
    void setOn(bool value);

    void process();

    SelectionWindow getSelectionWindow() const;
    ActionsWindow getActionsWindow() const;

private:
    void synchronizeModelWithSimulation();

    void leftMouseButtonPressed(RealVector2D const& viewPos);
    void leftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos, bool modifierKeyPressed);
    void leftMouseButtonReleased();

    void rightMouseButtonPressed(RealVector2D const& viewPos);
    void rightMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos);
    void rightMouseButtonReleased();

private:
    EditorModel _editorModel;
    SelectionWindow _selectionWindow;
    ActionsWindow _actionsWindow;

    SimulationController _simController;
    Viewport _viewport;
    StyleRepository _styleRepository;

    bool _on = false;

    boost::optional<RealVector2D> _prevMousePosInt;

    struct SelectionRect
    {
        RealVector2D startPos;
        RealVector2D endPos;
    };
    boost::optional<SelectionRect> _selectionRect;
};