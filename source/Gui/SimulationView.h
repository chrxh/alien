#pragma once

#include <chrono>

#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "Definitions.h"

class _SimulationView
{
public:
    _SimulationView(
        SimulationController const& simController,
        ModeController const& modeWindow,
        EditorModel const& editorModel);
    ~_SimulationView();

    void resize(IntVector2D const& viewportSize);

    void draw(bool renderSimulation);
    void processControls(bool renderSimulation);

    bool isOverlayActive() const;
    void setOverlayActive(bool active);

    void setBrightness(float value);
    void setContrast(float value);
    void setMotionBlur(float value);

    bool getMousePickerEnabled() const;
    void setMousePickerEnabled(bool value);
    std::optional<RealVector2D> getMousePickerPosition() const;

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

    void updateImageFromSimulation();
    void updateMotionBlur();

    void drawCursor();
    void markReferenceDomain();
    float calcZoomFactor(std::chrono::steady_clock::time_point const& lastTimepoint);

    //widgets
    SimulationScrollbar _scrollbarX;
    SimulationScrollbar _scrollbarY;

    //overlay
    bool _isCellDetailOverlayActive = false;
    float _motionBlurFactor = 1.0f;
    enum class NavigationState {
        Static, Moving
    };
    NavigationState _navigationState = NavigationState::Static;
    std::optional<OverlayDescription> _overlay;
    
    //shader data
    unsigned int _vao, _vbo, _ebo;
    unsigned int _fbo1, _fbo2;
    Shader _shader;

    bool _areTexturesInitialized = false;
    unsigned int _textureSimulationId = 0;
    unsigned int _textureFramebufferId1 = 0;
    unsigned int _textureFramebufferId2 = 0;

    //navigation
    std::optional<RealVector2D> _worldPosForMovement;
    std::optional<IntVector2D> _prevMousePosInt;
    std::optional<std::chrono::steady_clock::time_point> _lastZoomTimepoint;

    struct MouseWheelAction
    {
        bool up;    //false=down
        float strongness;
        std::chrono::steady_clock::time_point start;
        std::chrono::steady_clock::time_point lastTime;
    };
    std::optional<MouseWheelAction> _mouseWheelAction;
    bool _mousePickerEnabled = false;

    ModeController _modeWindow;
    SimulationController _simController;
    EditorModel _editorModel;
};
