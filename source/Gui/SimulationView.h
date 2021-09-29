#pragma once

#include "Base/Definitions.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _SimulationView
{
public:
    _SimulationView(SimulationController const& simController, ModeWindow const& modeWindow, Viewport const& viewport);
    void resize(IntVector2D const& viewportSize);
    void leftMouseButtonHold(IntVector2D const& viewPos, IntVector2D const& prevViewPos);
    void rightMouseButtonHold(IntVector2D const& viewPos);
    void middleMouseButtonPressed(IntVector2D const& viewPos);
    void middleMouseButtonHold(IntVector2D const& viewPos);
    void middleMouseButtonReleased();

    void processContent();
    void processControls();

private:
    void processEvents();
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

    SimulationController _simController;
};
