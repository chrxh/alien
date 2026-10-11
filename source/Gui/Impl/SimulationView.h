#pragma once

#include <chrono>

#include <Base/Interface/Definitions.h>
#include <Base/Interface/Singleton.h>

#include <Engine/Interface/Definitions.h>

#include <Rendering/Interface/PictureData.h>
#include <Rendering/Interface/RenderView.h>

#include "Definitions.h"

class SimulationView
{
    MAKE_SINGLETON(SimulationView);

public:
    void setup();
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

    PictureData savePicture(IntVector2D const& resolution);

    static auto constexpr DefaultBrightness = 1.0f;
    static auto constexpr DefaultContrast = 1.0f;
    static auto constexpr DefaultMotionBlur = 0.25f;

private:
    RenderView createRenderView() const;

    void markReferenceDomain();

    // Widgets
    SimulationScrollbars _scrollbars;

    // Overlay
    bool _cellDetailOverlayActive = false;

    float _brightness = DefaultBrightness;
    float _contrast = DefaultContrast;
    float _motionBlur = DefaultMotionBlur;

    // Settings
    bool _renderSimulation = true;
};
