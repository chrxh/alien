#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "Base/Singleton.h"

class _SimulationInteractionController
{
    MAKE_SINGLETON(_SimulationInteractionController);

public:
    void init(SimulationFacade const& simulationFacade, SimulationView const& simulationView);

    void process();

    bool isEditMode() const;
    void setEditMode(bool value);

    bool isDrawMode() const;
    void setDrawMode(bool value);

    bool isPositionSelectionMode() const;
    void setPositionSelectionMode(bool value);
    std::optional<RealVector2D> getPositionSelectionData() const;

private:
    void processEditWidget();
    void processEvents();

    void leftMouseButtonPressed(IntVector2D const& mousePos);
    void leftMouseButtonHold(IntVector2D const& mousePos, IntVector2D const& prevMousePos);
    void mouseWheelUp(IntVector2D const& mousePos, float strongness);
    void leftMouseButtonReleased(IntVector2D const& mousePos, IntVector2D const& prevMousePos);

    void rightMouseButtonPressed(IntVector2D const& mousePos);
    void rightMouseButtonHold(IntVector2D const& mousePos, IntVector2D const& prevMousePos);
    void mouseWheelDown(IntVector2D const& mousePos, float strongness);
    void rightMouseButtonReleased();

    void processMouseWheel(IntVector2D const& mousePos);

    void middleMouseButtonPressed(IntVector2D const& mousePos);
    void middleMouseButtonHold(IntVector2D const& mousePos);
    void middleMouseButtonReleased();

    void drawCursor();

    void processSelectionRect();

    float calcZoomFactor(std::chrono::steady_clock::time_point const& lastTimepoint);

    SimulationFacade _simulationFacade;
    SimulationView _simulationView;

    TextureData _editorOn;
    TextureData _editorOff;

    struct Modes
    {
        bool editMode = false;
        bool drawMode = false;
        bool positionSelectionMode = false;
    };
    Modes _modes;
    Modes _modesAtClick;

    //navigation
    std::optional<RealVector2D> _worldPosForMovement;
    std::optional<RealVector2D> _worldPosOnClick;
    std::optional<IntVector2D> _prevMousePosInt;
    std::optional<RealVector2D> _selectionPositionOnClick;
    std::optional<RealRect> _selectionRect;

    std::optional<std::chrono::steady_clock::time_point> _lastZoomTimepoint;

    struct MouseWheelAction
    {
        bool up;  //false=down
        float strongness;
        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point lastTime;
    };
    std::optional<MouseWheelAction> _mouseWheelAction;
};
