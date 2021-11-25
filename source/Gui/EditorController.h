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
        SelectionWindow const selectionWindow);

    bool isOn() const;
    void setOn(bool value);

    void process();

private:
    void leftMouseButtonPressed(RealVector2D const& viewPos);
    void leftMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos);
    void leftMouseButtonReleased();

    void rightMouseButtonPressed(RealVector2D const& viewPos);
    void rightMouseButtonHold(RealVector2D const& viewPos, RealVector2D const& prevViewPos);
    void rightMouseButtonReleased();

private:
    SimulationController _simController;
    Viewport _viewport;
    SelectionWindow _selectionWindow;

    bool _on = false;

    boost::optional<RealVector2D> _prevMousePosInt;

    struct SelectionRect
    {
        RealVector2D startPos;
        RealVector2D endPos;
    };
    boost::optional<SelectionRect> _selectionRect;
};