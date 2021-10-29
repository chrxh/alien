#pragma once

#include "Base/Definitions.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _SimulationView
{
public:
    _SimulationView(SimulationController const& simController, ModeWindow const& modeWindow, Viewport const& viewport);
    void resize(IntVector2D const& viewportSize);

    void processContent();
    void processControls();

private:
    void processEvents();

    void leftMouseButtonPressed();
    void leftMouseButtonHold(IntVector2D const& viewPos, IntVector2D const& prevViewPos);
    void leftMouseButtonReleased();

    void rightMouseButtonPressed();
    void rightMouseButtonHold(IntVector2D const& viewPos);
    void rightMouseButtonReleased();

    void middleMouseButtonPressed(IntVector2D const& viewPos);
    void middleMouseButtonHold(IntVector2D const& viewPos);
    void middleMouseButtonReleased();

    void requestImageFromSimulation();

    //widgets
    Viewport _viewport;
    SimulationScrollbar _scrollbarX;
    SimulationScrollbar _scrollbarY;
    ModeWindow _modeWindow;

    //shader data
    unsigned int _vao, _vbo, _ebo;
    unsigned int _fbo;
    Shader _shader;

    bool _areTexturesInitialized = false;
    unsigned int _textureId = 0;
    unsigned int _textureFramebufferId = 0;

    //navigation
    boost::optional<RealVector2D> _worldPosForMovement;
    boost::optional<IntVector2D> _prevMousePosInt;

    SimulationController _simController;
};
