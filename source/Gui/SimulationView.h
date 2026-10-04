#pragma once

#include <chrono>

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>

#include "Definitions.h"
#include "PictureData.h"

class SimulationView
{
    MAKE_SINGLETON(SimulationView);

public:
    void setup();
    void shutdown();
    void releaseGraphicsResources();

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

    PictureData savePicture(IntVector2D const& resolution);

    static auto constexpr DefaultBrightness = 1.0f;
    static auto constexpr DefaultContrast = 1.0f;
    static auto constexpr DefaultMotionBlur = 0.25f;

private:
    void setupRenderPipeline();

    PictureData renderPicture(IntVector2D const& resolution);

    void markReferenceDomain();

    // Widgets
    SimulationScrollbars _scrollbars;

    // Overlay
    bool _cellDetailOverlayActive = false;

    RenderPipeline _renderPipeline;

    float _brightness = DefaultBrightness;
    float _contrast = DefaultContrast;
    float _motionBlur = DefaultMotionBlur;

    // Settings
    bool _renderSimulation = true;
};
