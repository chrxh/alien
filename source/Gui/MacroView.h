#pragma once

#include "Base/Definitions.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class MacroView
{
public:
    void init(SimulationController* simController, IntVector2D const& viewportSize, float zoomFactor);
    void resize(IntVector2D const& viewportSize);
    void continuousZoomIn(IntVector2D const& viewPos);
    void continuousZoomOut(IntVector2D const& viewPos);

    void render();

private:
    void requestImageFromSimulation();

    void centerTo(RealVector2D const& worldPosition, IntVector2D const& viewPos);
    RealVector2D mapViewToWorldPosition(RealVector2D const& viewPos) const;


    //shader data
    unsigned int _vao, _vbo, _ebo;
    unsigned int _fbo;
    Shader* _shader = nullptr;
    void* _cudaResource = nullptr;

    bool _areTexturesInitialized = false;
    unsigned int _textureId = 0;
    unsigned int _textureFramebufferId = 0;

    //simulation view data
    float _zoomFactor = 0.0f;
    RealVector2D _worldCenter;

    SimulationController* _simController = nullptr;
    IntVector2D _viewportSize;
};
