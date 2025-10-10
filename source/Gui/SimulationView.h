#pragma once

#include <chrono>

#include "Base/Singleton.h"
#include "Base/Definitions.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/OverlayDescriptions.h"
#include "EngineInterface/RenderData.h"

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
    void setupRenderPipeline();

    void markReferenceDomain();

    SimulationFacade _simulationFacade;

    // Widgets
    SimulationScrollbars _scrollbars;

    // Overlay
    bool _cellDetailOverlayActive = false;
    std::optional<OverlayDescription> _overlay;

    RenderPipeline _renderPipeline;

    NumRenderObjects _numObjects;

    // Shader data for background object rendering
    Shader _objectBackgroundShader;
    unsigned int _objectBackgroundTexture;
    unsigned int _objectBackgroundFbo;

    // Shader data for foreground object rendering
    Shader _objectForegroundShader;
    unsigned int _objectForegroundTexture;
    unsigned int _objectForegroundFbo;

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

    // Shader data for subsurface scattering post-processing (new separated shaders)
    Shader _subsurfaceScatterShader;
    unsigned int _subsurfaceScatterTexture;
    unsigned int _subsurfaceScatterFbo;
    
    // Shader data for Fresnel effect post-processing
    Shader _fresnelShader;
    unsigned int _fresnelTexture;
    unsigned int _fresnelFbo;
    
    // Shader data for merge post-processing
    Shader _mergeShader;
    unsigned int _mergeTexture;
    unsigned int _mergeFbo;
    
    // Screen background texture (dark blue background)
    unsigned int _screenBackgroundTexture;

    bool _areTexturesInitialized = false;

    float _brightness = DefaultBrightness;
    float _contrast = DefaultContrast;
    float _motionBlur = DefaultMotionBlur;

    // Settings
    bool _renderSimulation = true;
};
