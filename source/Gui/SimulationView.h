#pragma once

#include <chrono>

#include "Base/Singleton.h"
#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/OverlayDescriptions.h"

#include "Definitions.h"

class SimulationView
{
    MAKE_SINGLETON(SimulationView);

public:
    void setup(SimulationFacade const& simulationFacade);
    void shutdown();

    void resize(IntVector2D const& viewportSize);

    void draw();
    void processSimulationScrollbars();

    bool isRenderSimulation() const;
    void setRenderSimulation(bool value);

    bool isOverlayActive() const;
    void setOverlayActive(bool active);

    float getBrightness() const;
    void setBrightness(float value);
    float getContrast() const;
    void setContrast(float value);
    float getMotionBlur() const;
    void setMotionBlur(float value);

    void updateMotionBlur();

    static auto constexpr DefaultBrightness = 1.0f;
    static auto constexpr DefaultContrast = 1.0f;
    static auto constexpr DefaultMotionBlur = 0.25f;

private:
    void updateImageFromSimulation();

    void markReferenceDomain();

    SimulationFacade _simulationFacade;

    //widgets
    SimulationScrollbar _scrollbarX;
    SimulationScrollbar _scrollbarY;

    //overlay
    bool _isCellDetailOverlayActive = false;
    std::optional<OverlayDescription> _overlay;

    //shader data
    unsigned int _vao, _vbo, _ebo;
    unsigned int _fbo1, _fbo2;
    Shader _shader;

    bool _areTexturesInitialized = false;
    unsigned int _textureSimulationId = 0;
    unsigned int _textureFramebufferId1 = 0;
    unsigned int _textureFramebufferId2 = 0;

    float _brightness = DefaultBrightness;
    float _contrast = DefaultContrast;
    float _motionBlur = DefaultMotionBlur;

    //settings
    bool _renderSimulation = true;
};
