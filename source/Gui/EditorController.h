#pragma once

#include "Base/Definitions.h"
#include "EngineImpl/Definitions.h"

#include "Definitions.h"

class _EditorController
{
public:
    _EditorController(SimulationController const& simController, Viewport const& viewport);

    bool isOn() const;
    void setOn(bool value);

    void process();

private:
    void leftMouseButtonPressed(IntVector2D const& viewPos);
    void leftMouseButtonHold(IntVector2D const& viewPos, IntVector2D const& prevViewPos);
    void leftMouseButtonReleased();

private:
    SimulationController _simController;
    Viewport _viewport;

    bool _on = true;

    boost::optional<IntVector2D> _prevMousePosInt;

    struct SelectionRect
    {
        RealVector2D startPos;
        RealVector2D endPos;
    };
    boost::optional<SelectionRect> _selectionRect;
};