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

    bool isScrollbarDragging() const;

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
    void setupObjectShader();
    void setupBlurHorizontalShader();
    void setupBlurVerticalShader();
    void setupMetaballsShader();
    void setupSubsurfaceShader();

    void markReferenceDomain();

    SimulationFacade _simulationFacade;

    // Widgets
    SimulationScrollbars _scrollbars;

    // Overlay
    bool _cellDetailOverlayActive = false;
    std::optional<OverlayDescription> _overlay;

    // Shader data for object rendering
    Shader _objectShader;
    uint64_t _numObjects = 0;
    unsigned int _objectTexture;
    unsigned int _objectFbo;

    // Shader data for blur preprocessing
    Shader _blurHorizontalShader;
    Shader _blurVerticalShader;
    unsigned int _blurHorizontalTexture;
    unsigned int _blurHorizontalFbo;
    unsigned int _blurVerticalTexture;
    unsigned int _blurVerticalFbo;

    // Shader data for metaballs post-processing
    Shader _metaballsShader;
    unsigned int _metaballsTexture;
    unsigned int _metaballsFbo;

    // Shader data for subsurface scattering post-processing
    Shader _subsurfaceShader;

    bool _areTexturesInitialized = false;

    float _brightness = DefaultBrightness;
    float _contrast = DefaultContrast;
    float _motionBlur = DefaultMotionBlur;

    // Settings
    bool _renderSimulation = true;
};
