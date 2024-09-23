#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "Definitions.h"

class _SimulationInteractionController
{
public:
    _SimulationInteractionController(SimulationController const& simController, EditorController const& editorController, SimulationView const& simulationView);

    void process();

    bool isEditMode() const;
    void setEditMode(bool value);

    bool isDrawMode() const;
    void setDrawMode(bool value);

    bool isPositionSelectionMode() const;
    void setPositionSelectionMode(bool value);
    std::optional<RealVector2D> getPositionSelectionData() const;

private:
    void processEvents();

    void leftMouseButtonPressed(IntVector2D const& viewPos);
    void leftMouseButtonHold(IntVector2D const& viewPos, IntVector2D const& prevViewPos);
    void mouseWheelUp(IntVector2D const& viewPos, float strongness);
    void leftMouseButtonReleased();

    void rightMouseButtonPressed();
    void rightMouseButtonHold(IntVector2D const& viewPos);
    void mouseWheelDown(IntVector2D const& viewPos, float strongness);
    void rightMouseButtonReleased();

    void processMouseWheel(IntVector2D const& viewPos);

    void middleMouseButtonPressed(IntVector2D const& viewPos);
    void middleMouseButtonHold(IntVector2D const& viewPos);
    void middleMouseButtonReleased();

    void drawCursor();

    float calcZoomFactor(std::chrono::steady_clock::time_point const& lastTimepoint);

    SimulationController _simController;
    EditorController _editorController;
    SimulationView _simulationView;

    TextureData _editorOn;
    TextureData _editorOff;
    
    bool _editMode = false;
    bool _drawMode = false;
    bool _positionSelectionMode = false;

    //navigation
    std::optional<RealVector2D> _worldPosForMovement;
    std::optional<IntVector2D> _prevMousePosInt;
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
